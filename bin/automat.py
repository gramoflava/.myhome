#!/usr/bin/env python3
import argparse
import subprocess
import sys
import logging
import tempfile
import json
import shutil
import os
from pathlib import Path

# Global settings
LOG_FILE = "/tmp/automat.log"
ENABLE_LOGGING = False
TRASH_MODE = False
DEBUG_MODE = False
DRY_RUN = False
SUFFIX = "-re"
DEFAULT_CODEC = "h264"
DEFAULT_FORMAT = "mov"

#
# --- Video encoding quality/bitrate logic ---
# CPU (libx264/libx265) CRF settings:
#   - h264: default CRF 24
#   - hevc: default CRF 29
#   - If source is already compressed (bitrate < 1.2M for <=1080p, <2.5M for >1080p):
#       - h264: CRF 26
#       - hevc: CRF 31
# GPU (videotoolbox) bitrate settings:
#   - For SD/720p: min_gpu_bitrate = 450k
#   - For HD and above: min_gpu_bitrate = 600k
#   - If source bitrate < 1M: dynamic_bitrate = max(int(src_bitrate * 0.85), min_gpu_bitrate)
#   - Otherwise, use previous logic
DEFAULT_PRESET = "slow"  # encoding speed vs compression efficiency
DEFAULT_CRF_H264 = 24
DEFAULT_CRF_HEVC = 29

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def setup_logging():
    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    if ENABLE_LOGGING:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

def notify(title, message):
    """Show a macOS notification"""
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True)

def display_error(message):
    logger.error(message)
    
def display_info(message):
    logger.info(message)

def display_debug(message):
    if DEBUG_MODE:
        logger.debug(message)

def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    if DRY_RUN:
        display_info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)

def is_video_file(path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    res = run_command(["file", "--mime-type", "-b", str(path)])
    return res.stdout.strip().startswith("video/")

def is_image_file(path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    res = run_command(["file", "--mime-type", "-b", str(path)])
    return res.stdout.strip().startswith("image/")

def is_already_processed(path):
    """Check if file was already processed by looking for the suffix"""
    path = Path(path)
    return path.stem.endswith(SUFFIX)

def move_to_trash(path):
    if DRY_RUN:
        display_info(f"[DRY RUN] Would move to trash: {path}")
        return
        
    path = Path(path)
    if not path.exists():
        logger.error("File not found for trashing: %s", path)
        return
    script = f'''
    tell application "Finder"
        move POSIX file "{path.as_posix()}" to trash
    end tell
    '''
    res = run_command(["osascript", "-e", script])
    if res.returncode == 0:
        logger.info(f"Moved to trash: {path}")
        display_debug(f"Moved to trash: {path}")
    else:
        display_error(f"Trash failed: {res.stderr.strip()}")

def get_video_info(source):
    source = Path(source)
    logger.debug("Getting info for: %s", source)
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(source)]
    res = run_command(cmd)
    if res.returncode != 0:
        display_error(f"ffprobe error on {source}")
        return 0,0,0.0,0,0
    info = json.loads(res.stdout)
    stream = next((s for s in info.get("streams",[]) if s.get("codec_type")=="video"), {})
    width = stream.get("width", 0) or 0
    height = stream.get("height",0) or 0
    duration = float(info.get("format",{}).get("duration",0.0)) or 0.0
    filesize = source.stat().st_size
    bitrate = int(info.get("format",{}).get("bit_rate",0) or 0)
    logger.info(f"Info: {width}x{height}, {duration}s, {bitrate}b/s, {filesize} bytes")
    return width, height, duration, bitrate, filesize

def calculate_optimal_bitrate(w, h, curr, size, dur):
    display_debug(f"Calc bitrate for {w}x{h}, curr={curr}, size={size}, dur={dur}")
    pix = w*h
    if pix >= 8294400:   base = 6_000_000
    elif pix > 2073600:  base = 4_000_000
    elif pix >  921600:  base = 2_500_000
    else:                base = 1_200_000
    chosen = min(curr or base, base)
    if dur > 0:
        actual = int(size * 8 / dur)
        targ = int(actual * 0.8)
        if 0 < targ < chosen:
            chosen = targ
    chosen = max(chosen, 100_000)
    chosen = round(chosen/100_000)*100_000
    logger.info(f"Optimal bitrate: {chosen}")
    return chosen

def is_videotoolbox_available():
    """Check if videotoolbox hardware encoder is available via ffmpeg."""
    res = run_command(["ffmpeg", "-encoders"])
    return "videotoolbox" in res.stdout

def build_ffmpeg_command(src, codec, fmt, bitrate=None, is_cpu=None, crf_value=None, dynamic_bitrate=None):
    src = Path(src)
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    out = src.parent / f"{src.stem}{SUFFIX}.{fmt}"

    # Determine if we should use CPU (libx264/libx265) or GPU (videotoolbox).
    if is_cpu is None:
        # Default to CPU if not specified.
        is_cpu = True

    if not is_cpu:
        # Hardware-accelerated bitrate-based encoding (videotoolbox)
        # Use dynamic_bitrate if provided, else fallback to defaults.
        if codec == "h264":
            # dynamic_bitrate is in bits per second, ffmpeg expects k (e.g. 2000k)
            bval = f"{int(dynamic_bitrate // 1000)}k" if dynamic_bitrate else "4000k"
            vopt = f"-c:v h264_videotoolbox -b:v {bval} -tag:v avc1"
        elif codec == "hevc":
            bval = f"{int(dynamic_bitrate // 1000)}k" if dynamic_bitrate else "2500k"
            vopt = f"-c:v hevc_videotoolbox -b:v {bval} -tag:v hvc1"
        elif codec == "av1":
            vopt = "-c:v libaom-av1 -crf 30 -b:v 0 -strict experimental"
        else:
            display_error(f"Invalid codec: {codec}")
            sys.exit(1)
    else:
        # CPU-based CRF encoding for better quality/size tradeoff
        # Use crf_value if provided, else fallback to defaults.
        if codec == "h264":
            crf = crf_value if crf_value is not None else 24
            vopt = f"-c:v libx264 -preset slow -crf {crf}"
        elif codec == "hevc":
            crf = crf_value if crf_value is not None else 29
            vopt = f"-c:v libx265 -preset slow -crf {crf} -tag:v hvc1 -x265-params \"psy-rd=2.0:psy-rdoq=1.0:aq-mode=3:aq-strength=1.0:ref=5:bframes=8:rc-lookahead=60\""
        elif codec == "av1":
            vopt = "-c:v libaom-av1 -crf 30 -b:v 0 -strict experimental"
        else:
            display_error(f"Invalid codec: {codec}")
            sys.exit(1)

    if fmt == "mkv":
        vopt += " -f matroska"
    elif fmt == "webm":
        if codec != "av1":
            vopt = "-c:v libvpx-vp9 -crf 30 -b:v 0 -f webm"
        else:
            vopt += " -f webm"

    cmd = ["ffmpeg"]
    if not is_cpu and (codec == "h264" or codec == "hevc"):
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", str(src)]
    cmd += vopt.split()
    cmd += [
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a", "aac", "-b:a", "96k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        str(out)
    ]
    return cmd, out

def process_video_refine(src, codec, fmt, is_cpu=None):
    """
    Refine a video file by re-encoding it
    - For CPU: dynamically choose CRF based on source bitrate and resolution.
    - For GPU: dynamically choose bitrate based on source bitrate and resolution.
    """
    src = Path(src)
    w, h, dur, br, sz = get_video_info(src)
    # --- Aggressive compression logic for CPU/GPU ---
    # For CPU:
    #   - h264: default CRF 24, if already compressed (low bitrate): CRF 26
    #   - hevc: default CRF 29, if already compressed: CRF 31
    # For GPU:
    #   - Use dynamic bitrate (see below)
    #
    # Thresholds:
    #   - For <=1080p: low_bitrate_thresh = 1_200_000, gpu_bitrate_thresh = 2_500_000
    #   - For >1080p:  low_bitrate_thresh = 2_500_000, gpu_bitrate_thresh = 4_000_000
    #
    # Minimal GPU bitrate: 450_000 (SD/720p), 600_000 (HD and above)
    #
    # For GPU: if src_bitrate < 1_000_000: dynamic_bitrate = max(int(src_bitrate * 0.85), min_gpu_bitrate)
    #
    # If AMV or loop_audio, keep old behavior (handled outside)

    is_hd = max(w, h) > 1080
    low_bitrate_thresh = 2_500_000 if is_hd else 1_200_000
    gpu_bitrate_thresh = 4_000_000 if is_hd else 2_500_000
    min_gpu_bitrate = 600_000 if is_hd else 450_000

    crf_value = None
    dynamic_bitrate = None

    if is_cpu:
        # Aggressive CRF selection for CPU encoder
        if codec == "h264":
            if br > 0 and br < low_bitrate_thresh:
                crf_value = 28
            else:
                crf_value = 24
        elif codec == "hevc":
            if br > 0 and br < low_bitrate_thresh:
                crf_value = 32  
            else:
                crf_value = 28
        # For av1, keep as before (CRF 30)
    else:
        # Dynamic bitrate selection for GPU encoder (videotoolbox)
        src_bitrate = br if br > 0 else gpu_bitrate_thresh
        if src_bitrate < 1_000_000:
            dynamic_bitrate = max(int(src_bitrate * 0.85), min_gpu_bitrate)
        else:
            if src_bitrate > gpu_bitrate_thresh:
                chosen = int(max(0.8 * src_bitrate, gpu_bitrate_thresh))
                dynamic_bitrate = min(src_bitrate, chosen)
            else:
                dynamic_bitrate = src_bitrate
            if dynamic_bitrate < min_gpu_bitrate:
                dynamic_bitrate = min_gpu_bitrate

    # For AMV and loop_audio, behavior is unchanged (handled outside)
    # For refine: pass new crf_value/dynamic_bitrate
    cmd, out = build_ffmpeg_command(
        src, codec, fmt, dynamic_bitrate if not is_cpu else None,
        is_cpu=is_cpu, crf_value=crf_value, dynamic_bitrate=dynamic_bitrate
    )
    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not DRY_RUN:
        display_error("ffmpeg failed")
        return False
    if not DRY_RUN and (not out.is_file() or out.stat().st_size == 0):
        display_error(f"Output missing: {out}")
        return False
    orig_sz = src.stat().st_size
    new_sz = out.stat().st_size if not DRY_RUN else int(orig_sz * 0.7)  # Estimate for dry run
    red = (1 - new_sz / orig_sz) * 100 if orig_sz > 0 else 0
    display_info(f"{orig_sz/1e6:.2f}→{new_sz/1e6:.2f} MB ({red:.1f}% reduction)")
    if TRASH_MODE:
        move_to_trash(src)
    return True

def process_video_amv(src, audio_track, codec, fmt):
    """Add or replace audio track in video (AMV operation)"""
    src = Path(src)
    audio_track = Path(audio_track)
    out = src.parent / f"{src.stem}{SUFFIX}.{fmt}"
    
    if not audio_track.exists():
        display_error(f"Audio track not found: {audio_track}")
        return False
    
    # Build FFmpeg command for AMV operation
    cmd = ["ffmpeg"]
    if USE_GPU:
        cmd += ["-hwaccel", "videotoolbox"]
    
    cmd += [
        "-i", str(src),        # Video input
        "-i", str(audio_track), # Audio input
        "-c:v", "copy",        # Copy video stream as-is
        "-c:a", "aac",         # Re-encode audio to AAC
        "-b:a", "128k",        # Audio bitrate
        "-map", "0:v:0",       # Map first video stream from first input
        "-map", "1:a:0",       # Map first audio stream from second input
        "-shortest",           # End when shortest stream ends
        "-movflags", "+faststart",
        "-y",
        str(out)
    ]
    
    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not DRY_RUN:
        display_error("ffmpeg AMV failed")
        return False
    if not DRY_RUN and (not out.is_file() or out.stat().st_size == 0):
        display_error(f"Output missing: {out}")
        return False
    
    display_info(f"AMV created: {out}")
    if TRASH_MODE:
        move_to_trash(src)
    return True

def process_video_loop_audio(src, codec, fmt):
    """Loop audio to match video duration"""
    src = Path(src)
    out = src.parent / f"{src.stem}{SUFFIX}.{fmt}"
    
    # Get video and audio duration info
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(src)]
    res = run_command(cmd)
    if res.returncode != 0:
        display_error(f"ffprobe error on {src}")
        return False
    
    info = json.loads(res.stdout)
    video_stream = next((s for s in info.get("streams",[]) if s.get("codec_type")=="video"), {})
    audio_stream = next((s for s in info.get("streams",[]) if s.get("codec_type")=="audio"), {})
    
    if not video_stream or not audio_stream:
        display_error(f"Missing video or audio stream in {src}")
        return False
    
    video_duration = float(video_stream.get("duration", 0) or info.get("format",{}).get("duration", 0))
    
    # Build FFmpeg command to loop audio
    cmd = ["ffmpeg"]
    if USE_GPU:
        cmd += ["-hwaccel", "videotoolbox"]
    
    cmd += [
        "-stream_loop", "-1",  # Loop audio indefinitely
        "-i", str(src),
        "-c:v", "copy",        # Copy video as-is
        "-c:a", "aac",         # Re-encode audio
        "-b:a", "128k",
        "-t", str(video_duration),  # Stop at video duration
        "-movflags", "+faststart",
        "-y",
        str(out)
    ]
    
    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not DRY_RUN:
        display_error("ffmpeg loop_audio failed")
        return False
    if not DRY_RUN and (not out.is_file() or out.stat().st_size == 0):
        display_error(f"Output missing: {out}")
        return False
    
    display_info(f"Audio looped: {out}")
    if TRASH_MODE:
        move_to_trash(src)
    return True

def process_video_audiofy(src, codec, fmt):
    """Extract audio from video and save as audio file"""
    src = Path(src)
    out = src.parent / f"{src.stem}{SUFFIX}.mp3"
    
    cmd = ["ffmpeg", "-i", str(src), "-vn", "-c:a", "libmp3lame", "-b:a", "192k", "-y", str(out)]
    
    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not DRY_RUN:
        display_error("ffmpeg audiofy failed")
        return False
    if not DRY_RUN and (not out.is_file() or out.stat().st_size == 0):
        display_error(f"Output missing: {out}")
        return False
    
    display_info(f"Audio extracted: {out}")
    if TRASH_MODE:
        move_to_trash(src)
    return True

def process_image(src):
    src = Path(src)
    out = src.parent / f"{src.stem}{SUFFIX}.heic"
    cmd = ["sips","-s","format","heic",str(src),"--out",str(out)]
    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not DRY_RUN:
        display_error("sips failed")
        return False
    if not DRY_RUN and (not out.is_file() or out.stat().st_size == 0):
        display_error(f"Output missing: {out}")
        return False
    orig_sz = src.stat().st_size
    new_sz = out.stat().st_size if not DRY_RUN else int(orig_sz * 0.5)  # Estimate for dry run
    red = (1-new_sz/orig_sz)*100 if orig_sz>0 else 0
    display_info(f"{orig_sz/1e3:.2f}→{new_sz/1e3:.2f} KB ({red:.1f}% reduction)")
    if TRASH_MODE:
        move_to_trash(src)
    return True

def process_single_file(file_path, operations, audio_track, codec, fmt, is_cpu=None):
    """Process a single file with the specified operations"""
    file_path = Path(file_path)
    
    # Convert to absolute path if not already
    if not file_path.is_absolute():
        file_path = Path(os.getcwd()) / file_path
    
    if not file_path.exists():
        display_error(f"File not found: {file_path}")
        return False
    
    # Skip if already processed
    if is_already_processed(file_path):
        display_info(f"Skipping already processed file: {file_path}")
        return True
    
    success = True
    # Process based on file type and operations
    if is_video_file(file_path):
        display_info(f"Processing video: {file_path}")
        for operation in operations:
            if operation == "refine":
                if not process_video_refine(file_path, codec, fmt, is_cpu=is_cpu):
                    success = False
            elif operation == "amv":
                if not audio_track:
                    display_error("AMV operation requires audio track (-a parameter)")
                    success = False
                elif not process_video_amv(file_path, audio_track, codec, fmt):
                    success = False
            elif operation == "loop_audio":
                if not process_video_loop_audio(file_path, codec, fmt):
                    success = False
            elif operation == "audiofy":
                if not process_video_audiofy(file_path, codec, fmt):
                    success = False

    elif is_image_file(file_path):
        display_info(f"Processing image: {file_path}")
        for operation in operations:
            if operation == "refine":
                if not process_image(file_path):
                    success = False
            elif operation == "audiofy" and audio_track:
                if not process_image_audiofy(file_path, audio_track, codec, fmt):
                    success = False
            else:
                if operation not in ("refine", "audiofy"):
                    display_info(f"Skipping image (no supported operations): {file_path}")
    else:
        display_error(f"Unsupported file type: {file_path}")
        success = False
    return success
def process_image_audiofy(src, audio_track, codec, fmt):
    """
    Given an image and an audio track, create a video of the image with the audio.
    If audio_track is a video, extract its audio stream first.
    """
    import uuid
    src = Path(src)
    audio_track = Path(audio_track)
    tmp_audio = None
    try:
        # Determine if audio_track is a video file
        if is_video_file(audio_track):
            # extract audio to temp file
            tmpdir = tempfile.gettempdir()
            tmp_audio_name = f"automat_tmp_audio_{uuid.uuid4().hex}.mp3"
            tmp_audio = Path(tmpdir) / tmp_audio_name
            cmd_extract = [
                "ffmpeg", "-i", str(audio_track), "-vn", "-c:a", "libmp3lame", "-b:a", "192k", "-y", str(tmp_audio)
            ]
            logger.info("Extracting audio from video: " + " ".join(cmd_extract))
            res = run_command(cmd_extract)
            if res.returncode != 0 and not DRY_RUN:
                display_error("Failed to extract audio from video for image+audio operation")
                return False
            audio_input = tmp_audio
        else:
            audio_input = audio_track

        # Get duration of audio
        cmd_probe = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_input)
        ]
        res = run_command(cmd_probe)
        if res.returncode != 0 or not res.stdout.strip():
            display_error(f"Could not determine audio duration: {audio_input}")
            return False
        duration = float(res.stdout.strip())

        # Output filename
        out = src.parent / f"{src.stem}{SUFFIX}.{fmt}"

        # Build ffmpeg command to create video from image + audio
        # Use -loop 1 to repeat image, -framerate 30, -t for duration, -shortest for audio, -r 30 before output
        cmd = [
            "ffmpeg",
            "-loop", "1",
            "-framerate", "30",
            "-i", str(src),
            "-i", str(audio_input),
            "-c:v", "libx264" if codec == "h264" else ("libx265" if codec == "hevc" else "libaom-av1"),
            "-t", str(duration),
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            "-r", "30",
            "-y",
            str(out)
        ]
        logger.info("Running: " + " ".join(str(x) for x in cmd))
        res = run_command(cmd)
        if res.returncode != 0 and not DRY_RUN:
            display_error("ffmpeg failed to create video from image and audio")
            return False
        if not DRY_RUN and (not out.is_file() or out.stat().st_size == 0):
            display_error(f"Output missing: {out}")
            return False
        display_info(f"Created video from image and audio: {out}")
        if TRASH_MODE:
            move_to_trash(src)
        return True
    finally:
        # Clean up temp audio file if it was created
        if tmp_audio is not None and Path(tmp_audio).exists():
            try:
                Path(tmp_audio).unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp audio file: {tmp_audio} ({e})")

def expand_paths_recursively(paths):
    """Recursively expand all given paths into a flat list of media files"""
    expanded_files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        # Convert to absolute path
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
            
        if path.is_file():
            # Single file - add if it's a media file and not already processed
            if not is_already_processed(path) and (is_video_file(path) or is_image_file(path)):
                expanded_files.append(path)
        elif path.is_dir():
            # Directory - recursively find all media files
            for file_path in path.rglob("*"):
                if file_path.is_file() and not is_already_processed(file_path):
                    if is_video_file(file_path) or is_image_file(file_path):
                        expanded_files.append(file_path)
        else:
            display_error(f"Path not found: {path}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file_path in expanded_files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)
    
    return unique_files

def get_base_path(paths):
    """Get the base path for progress/error log files from the first path argument"""
    if not paths:
        return Path.cwd()
    first_path = Path(paths[0])
    if not first_path.is_absolute():
        first_path = Path.cwd() / first_path
    if first_path.is_file():
        return first_path.parent
    elif first_path.is_dir():
        return first_path
    else:
        return Path.cwd()

def update_progress_file(progress_file, total, processed, errors, progress_created):
    """
    Update the progress tracking file (.automat-progress.log) if it exists.
    Only create or update after 3 minutes (see main for logic).
    """
    if DRY_RUN:
        display_info(f"[DRY RUN] Would update progress: {total}/{processed}/{errors}")
        return
    if not progress_created:
        # Do not create the file if it doesn't exist yet
        return
    try:
        with open(progress_file, 'w') as f:
            f.write(f"{total}/{processed}/{errors}\n")
    except Exception as e:
        display_error(f"Failed to update progress file: {e}")

def log_error_file(error_file, file_path, error_created):
    """
    Append failed file path to error log (.automat-errors.log) if it exists.
    Only create or update after 3 minutes (see main for logic).
    """
    if DRY_RUN:
        display_info(f"[DRY RUN] Would log error: {file_path}")
        return
    if not error_created:
        # Do not create the file if it doesn't exist yet
        return
    try:
        with open(error_file, 'a') as f:
            f.write(f"{file_path}\n")
    except Exception as e:
        display_error(f"Failed to write to error file: {e}")

def refine_recursively(directory, codec, fmt):
    """Recursively process all files in directory (legacy function for compatibility)"""
    display_info(f"Recursively refining: {directory}")
    files = expand_paths_recursively([directory])
    
    total = len(files)
    display_info(f"Found {total} files")
    
    # Set up progress tracking
    base_path = get_base_path([directory])
    progress_file = base_path / "automat-progress.txt"
    error_file = base_path / "automat-errors.txt"
    
    # Initialize progress file
    update_progress_file(progress_file, total, 0, 0)
    
    proc = fail = 0
    
    for idx, path in enumerate(files, 1):
        display_info(f"[{idx}/{total}] Processing: {path}")
        if process_single_file(path, ["refine"], None, codec, fmt):
            display_info(f"✓ {path}")
            proc += 1
        else:
            display_error(f"✗ {path}")
            fail += 1
            log_error_file(error_file, path)
        
        # Update progress after each file
        update_progress_file(progress_file, total, proc, fail)
    
    display_info(f"Done. Processed: {proc}, Failed: {fail}")
    if not DRY_RUN:
        notify("Automat", f"Processed {proc} files, {fail} failed")

def main():
    """
    Main entry point for Automat.

    Progress and error logs:
      - Progress and error logs are only created and updated for long runs (over 3 minutes).
      - Progress log: .automat-progress.log in the first input's directory.
      - Error log: .automat-errors.log in the first input's directory.
      - These files are NOT created at start. They are only created/updated if the run exceeds 3 minutes.
      - At the end, if these files exist and there were no errors, they are deleted.
    """
    import time
    global ENABLE_LOGGING, TRASH_MODE, DEBUG_MODE, DRY_RUN, SUFFIX
    p = argparse.ArgumentParser(
        description=(
            "Automat: refine videos and images using hardware-accelerated encoding.\n\n"
            "This tool optimizes media files by re-encoding them with efficient codecs.\n"
            "Videos are processed with ffmpeg using hardware acceleration when available.\n"
            "Images are converted to HEIC format using sips for better compression.\n\n"
            "Progress and error logs (.automat-progress.log, .automat-errors.log) are only created for long runs (over 3 minutes) and are always located in the first input's directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Refine multiple videos with HEVC codec and move to trash:\n"
            "  automat.py -t --refine video1.mp4 video2.avi video3.mkv\n\n"
            "  # Create AMV with new audio track:\n"
            "  automat.py --amv -a new_track.mp3 -c h264 -f mp4 video.avi\n\n"
            "  # Use CPU instead of GPU (better compression, slower):\n"
            "  automat.py --force-cpu --refine myvideo.mp4\n\n"
            "  # Multiple operations on multiple files:\n"
            "  automat.py -t --amv --refine -a track.mp3 file1.avi file2.mkv\n\n"
            ""
            "Automator Quick Action Example (macOS):\n"
            "  1. Open Automator and create a new \"Quick Action\".\n"
            "  2. Set \"Workflow receives current: files or folders\" in \"Finder.app\".\n"
            "  3. Add a \"Run Shell Script\" action.\n"
            "  4. Configure:\n"
            "     - Shell: /bin/zsh\n"
            "     - Pass input: as arguments\n"
            "     - Script:\n"
            "```\n"
            "source $HOME/.zprofile\n"
            "automat.py -t --refine \"$@\"\n"
            "```\n"
        )
    )

    # Operation flags
    p.add_argument("--refine", action="store_true",
                   help="Refine/optimize video or image files")
    p.add_argument("--amv", action="store_true",
                   help="Add or replace audio track in video (requires -a)")
    p.add_argument("--loop-audio", dest="loop_audio", action="store_true",
                   help="Loop audio to match video duration")
    p.add_argument("--audiofy", action="store_true",
                   help="Extract audio from video as MP3, or if an image and an audio are given, create a video from the image with the audio track. For best macOS compatibility, use -c h264.")

    # Configuration options
    p.add_argument("-a", dest="audio", metavar="AUDIO_FILE",
                   help="Audio file for AMV operation")
    p.add_argument("-c", default=DEFAULT_CODEC, metavar="CODEC",
                   help="Video codec (h264, hevc, av1) [default: h264]. h264 — most compatible; hevc — better compression, slower, not as compatible. av1 - experimental, best compression but slowest and least compatible.")
    p.add_argument("-f", default=DEFAULT_FORMAT, metavar="FORMAT",
                   help=f"Output format (mov, mp4, mkv, webm) [default: {DEFAULT_FORMAT}]")
    p.add_argument("-s", dest="suffix", default=SUFFIX,
                   help=f"Custom suffix for output files [default: {SUFFIX}]")

    # Behavior flags
    p.add_argument("--force-cpu", action="store_true",
                   help="Force CPU-encoder (libx264/libx265) for better compression. Default: HW-accelerated (videotoolbox) for faster results.")
    p.add_argument("-t", action="store_true",
                   help="Move original files to trash after processing")
    p.add_argument("-v", action="store_true",
                   help="Verbose output")
    p.add_argument("-d", action="store_true",
                   help="Debug mode (extra logging)")
    p.add_argument("-l", action="store_true",
                   help="Enable logging to file")
    p.add_argument("-n", action="store_true",
                   help="Dry-run mode (show what would happen without processing)")

    # File arguments
    p.add_argument("files", nargs="+",
                   help="Files or directories to process")

    args = p.parse_args()

    # Set global flags
    ENABLE_LOGGING = args.l or args.v or args.d
    DEBUG_MODE = args.d
    TRASH_MODE = args.t
    DRY_RUN = args.n
    SUFFIX = args.suffix
    codec = args.c
    fmt = args.f

    setup_logging()

    # Determine if we should use CPU or GPU (videotoolbox)
    if args.force_cpu:
        is_cpu = True
        display_info("Forced CPU-encoding (libx264/libx265).")
    else:
        if is_videotoolbox_available():
            is_cpu = False
            display_info("GPU-accelerated encoding (videotoolbox).")
        else:
            is_cpu = True
            display_info("Cannot identify GPU-acceleration (videotoolbox), default to CPU.")

    # Determine which operations to perform
    operations = []
    if args.refine:
        operations.append("refine")
    if args.amv:
        operations.append("amv")
    if args.loop_audio:
        operations.append("loop_audio")
    if args.audiofy:
        operations.append("audiofy")

    # Default to refine if no operations specified
    if not operations:
        operations.append("refine")
        display_info("No operations specified, defaulting to --refine")

    # First, recursively expand all paths into a flat list of media files
    display_info("Expanding paths and discovering media files...")
    all_files = expand_paths_recursively(args.files)

    if not all_files:
        display_error("No media files found to process")
        return 1

    total_files = len(all_files)
    display_info(f"Found {total_files} media files to process")

    # Set up progress and error tracking
    base_path = get_base_path(args.files)
    progress_file = base_path / ".automat-progress.log"
    error_file = base_path / ".automat-errors.log"

    display_info(f"Progress will be tracked in: {progress_file}")
    display_info(f"Errors will be logged in: {error_file}")

    # Do not create progress or error files yet
    progress_created = False
    error_created = False

    processed = 0
    failed = 0

    # Remove existing error file to start fresh (only if it exists)
    if not DRY_RUN and error_file.exists():
        try:
            error_file.unlink()
        except Exception as e:
            display_error(f"Failed to remove existing error file: {e}")

    # Remove existing progress file if it exists (for clean start)
    if not DRY_RUN and progress_file.exists():
        try:
            progress_file.unlink()
        except Exception as e:
            display_error(f"Failed to remove existing progress file: {e}")

    import time
    process_start = time.time()

    # Process each file
    for idx, file_path in enumerate(all_files, 1):
        display_info(f"[{idx}/{total_files}] Processing: {file_path}")

        # Process single file
        if process_single_file(file_path, operations, args.audio, codec, fmt, is_cpu=is_cpu):
            display_info(f"✓ {file_path}")
            processed += 1
        else:
            display_error(f"✗ {file_path}")
            failed += 1

        elapsed = time.time() - process_start

        # Only create and update progress/error files if elapsed > 180s (3 minutes)
        if elapsed > 180:
            if not progress_created:
                progress_created = True
            if not error_created:
                error_created = True
        # Update progress file if it exists
        update_progress_file(progress_file, total_files, processed, failed, progress_created)
        # Log error if needed and error file exists
        if failed > 0:
            log_error_file(error_file, file_path, error_created)

    display_info(f"Processing complete. Success: {processed}, Failed: {failed}")

    # Final progress update if file exists
    update_progress_file(progress_file, total_files, processed, failed, progress_created)

    # Show notification when complete
    if not DRY_RUN and total_files > 1:
        notify("Automat", f"Processed {processed} files, {failed} failed")

    # If .automat-progress.log exists and there were no errors, delete it
    if not DRY_RUN and progress_file.exists() and failed == 0:
        try:
            progress_file.unlink()
        except Exception as e:
            display_error(f"Failed to remove progress log: {e}")
    # If .automat-errors.log exists and there were no errors, delete it
    if not DRY_RUN and error_file.exists() and failed == 0:
        try:
            error_file.unlink()
        except Exception as e:
            display_error(f"Failed to remove error log: {e}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())