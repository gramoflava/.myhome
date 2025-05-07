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
USE_GPU = False
TRASH_MODE = False
DEBUG_MODE = False
DRY_RUN = False
SUFFIX = "-re"
DEFAULT_CODEC = "hevc"
DEFAULT_FORMAT = "mp4"

# Default CPU-based CRF encoding settings
DEFAULT_PRESET = "slow"  # encoding speed vs compression efficiency
DEFAULT_CRF_H264 = 23      # quality level for H.264 (lower is higher quality)
DEFAULT_CRF_HEVC = 28      # quality level for HEVC

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

def build_ffmpeg_command(src, codec, fmt, bitrate):
    src = Path(src)
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    out = src.parent / f"{src.stem}{SUFFIX}.{fmt}"

    # Choose encoding strategy based on GPU flag
    if USE_GPU:
        # Hardware-accelerated bitrate-based encoding
        if codec == "h264":
            vopt = f"-c:v h264_videotoolbox -b:v {bitrate//1024}k -tag:v avc1"
        elif codec == "hevc":
            vopt = f"-c:v hevc_videotoolbox -b:v {bitrate//1024}k -tag:v hvc1"
        elif codec == "av1":
            vopt = "-c:v libaom-av1 -crf 30 -b:v 0 -strict experimental"
        else:
            display_error(f"Invalid codec: {codec}")
            sys.exit(1)
    else:
        # CPU-based CRF encoding for better quality/size tradeoff
        if codec == "h264":
            vopt = f"-c:v libx264 -preset {DEFAULT_PRESET} -crf {DEFAULT_CRF_H264}"
        elif codec == "hevc":
            vopt = f"-c:v libx265 -preset {DEFAULT_PRESET} -crf {DEFAULT_CRF_HEVC} -tag:v hvc1 -x265-params \"psy-rd=2.0:psy-rdoq=1.0:aq-mode=3:aq-strength=1.0:ref=5:bframes=8:rc-lookahead=60\""
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

    # Build FFmpeg command with optional GPU acceleration and faststart
    cmd = ["ffmpeg"]
    if USE_GPU:
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", str(src)]
    cmd += vopt.split()
    cmd += [
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        str(out)
    ]
    return cmd, out

def process_video(src, codec, fmt):
    src = Path(src)
    w,h,dur,br,sz = get_video_info(src)
    new_br = calculate_optimal_bitrate(w,h,br,sz,dur)
    cmd, out = build_ffmpeg_command(src, codec, fmt, new_br)
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
    red = (1-new_sz/orig_sz)*100 if orig_sz>0 else 0
    display_info(f"{orig_sz/1e6:.2f}→{new_sz/1e6:.2f} MB ({red:.1f}% reduction)")
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

def refine_recursively(directory, codec, fmt):
    display_info(f"Recursively refining: {directory}")
    files = []
    directory = Path(directory)
    for path in directory.rglob("*"):
        if path.stem.endswith(SUFFIX):
            continue
        if is_video_file(path):
            kind = "video"
        elif is_image_file(path):
            kind = "image"
        else:
            continue
        files.append((path, kind))
    total = len(files)
    display_info(f"Found {total} files")
    proc = fail = 0
    for idx, (path, kind) in enumerate(files,1):
        display_info(f"[{idx}/{total}] Processing {kind}: {path}")
        ok = process_video(path, codec, fmt) if kind=="video" else process_image(path)
        if ok:
            display_info(f"✓ {path}")
        else:
            fail += 1
            display_error(f"✗ {path}")
        proc += 1
    display_info(f"Done. Processed: {proc}, Failed: {fail}")
    # Show notification when complete
    if not DRY_RUN:
        notify("Automat", f"Processed {proc} files, {fail} failed")

def main():
    global ENABLE_LOGGING, USE_GPU, TRASH_MODE, DEBUG_MODE, DRY_RUN, SUFFIX
    p = argparse.ArgumentParser(
        description=(
            "Automat: refine videos and images using hardware-accelerated encoding.\n\n"
            "This tool optimizes media files by re-encoding them with efficient codecs.\n"
            "Videos are processed with ffmpeg using hardware acceleration when available.\n"
            "Images are converted to HEIC format using sips for better compression."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Refine a single video with HEVC codec and move to trash:\n"
            "  automat.py -t refine myvideo.mp4\n\n"
            "  # Refine all media in a directory with H264 codec and .mp4 output:\n"
            "  automat.py -c h264 -f mp4 refine /path/to/directory\n\n"
            "  # Process with debug info and GPU acceleration:\n"
            "  automat.py -d -g refine myvideo.mp4\n\n"
            "Automator Quick Action Example (macOS):\n"
            "  1. Open Automator and create a new \"Quick Action\".\n"
            "  2. Set \"Workflow receives current: files or folders\" in \"Finder.app\".\n"
            "  3. Add a \"Run Shell Script\" action.\n"
            "  4. Configure:\n"
            "     - Shell: /bin/zsh\n"
            "     - Pass input: as arguments\n"
            "     - Script:\n"
            "```\n"
            "source $HOME/.zshrc\n"
            "for f in \"$@\"; do\n"
            "    $HOME/bin/automat.py -t refine \"$f\"\n"
            "done\n"
            "```\n"
        )
    )
    
    # Command options
    p.add_argument("-v", action="store_true", 
                   help="Verbose output")
    p.add_argument("-c", default=DEFAULT_CODEC, metavar="CODEC",
                   help=f"Video codec (h264, hevc, av1) [default: {DEFAULT_CODEC}]")
    p.add_argument("-g", action="store_true", 
                   help="Enable GPU acceleration")
    p.add_argument("-f", default=DEFAULT_FORMAT, metavar="FORMAT",
                   help=f"Output format (mov, mp4, mkv, webm) [default: {DEFAULT_FORMAT}]")
    p.add_argument("-l", action="store_true", 
                   help="Enable logging to file")
    p.add_argument("-t", action="store_true", 
                   help="Move original files to trash after processing")
    p.add_argument("-d", action="store_true", 
                   help="Debug mode (extra logging)")
    p.add_argument("-s", dest="suffix", default=SUFFIX,
                   help=f"Custom suffix for output files [default: {SUFFIX}]")
    p.add_argument("-n", action="store_true",
                   help="Dry-run mode (show what would happen without processing)")
    
    # Positional arguments
    p.add_argument("operation", 
                   choices=["refine", "amv", "loop_audio", "audiofy"],
                   help="Operation to perform (currently only 'refine' is fully implemented)")
    p.add_argument("source", 
                   help="Source file or directory to process")
    p.add_argument("param", nargs="?", 
                   help="Optional parameter (depends on operation)")
    
    args = p.parse_args()

    ENABLE_LOGGING = args.l or args.v or args.d
    DEBUG_MODE = args.d
    USE_GPU = args.g
    TRASH_MODE = args.t
    DRY_RUN = args.n
    SUFFIX = args.suffix
    codec = args.c
    fmt = args.f

    setup_logging()

    source_path = Path(args.source)
    
    # Check working directory
    display_debug(f"Current working directory: {os.getcwd()}")
    display_debug(f"Source path: {source_path}")
    
    # Convert to absolute path if not already
    if not source_path.is_absolute():
        source_path = Path(os.getcwd()) / source_path
        display_debug(f"Converted to absolute path: {source_path}")

    if args.operation == "refine":
        if source_path.is_dir():
            refine_recursively(source_path, codec, fmt)
        else:
            if not source_path.exists():
                logger.error(f"Not found: {source_path}")
                return 1
            if is_video_file(source_path):
                process_video(source_path, codec, fmt)
            elif is_image_file(source_path):
                process_image(source_path)
            else:
                logger.error(f"Unsupported type: {source_path}")
                return 1
    else:
        logger.error(f"Operation '{args.operation}' not fully implemented yet")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())