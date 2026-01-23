#!/usr/bin/env python3
"""
automat.py - Self-contained media processing tool for macOS

A single-file script for optimizing videos and images with ffmpeg/sips.
No external Python dependencies required (only stdlib).

ARCHITECTURE:
  - Configuration & Presets: Quality presets (meme/share/archive) + Config dataclass
  - Media File Abstraction: MediaFile class with cached metadata + VideoInfo
  - Codec Strategies: Pluggable codec strategies (H264/HEVC CPU/GPU, AV1)
  - Operations: Base Operation class for refine/amv/audiofy/loop_audio
  - Interactive Mode: Guided prompts for rare operations
  - CLI Interface: Simplified argument parsing with preset support

USAGE:
  Most common: automat.py video.mp4                    # Quick refine with defaults
  Memes:       automat.py --preset meme funny.mp4      # Smallest size
  Batch:       automat.py -t ~/Downloads/videos/       # Process folder, trash originals
  Interactive: automat.py -i video.mp4                 # Guided prompts
  Auto:        automat.py --auto large_file.mov        # Auto-detect best preset

PRESETS:
  - meme:    Smallest size for sharing (CRF 28/32, 64k audio, max 1080p)
  - share:   Balanced for messaging apps (CRF 24/28, 96k audio, max 1920p) [DEFAULT]
  - archive: Best quality for storage (CRF 20/24, 192k audio, no limit)

AUTHOR: lava
LICENSE: Public domain
"""

import argparse
import subprocess
import sys
import logging
import tempfile
import json
import shutil
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

# ============================================================================
# CONFIGURATION & PRESETS
# ============================================================================

@dataclass
class Preset:
    """Quality preset definition"""
    name: str
    description: str
    h264_crf: int
    hevc_crf: int
    max_resolution: Optional[int]
    audio_bitrate: str

PRESETS = {
    "meme": Preset("meme", "Smallest size for sharing", 28, 32, 1080, "64k"),
    "share": Preset("share", "Balanced for messaging apps", 24, 28, 1920, "96k"),
    "archive": Preset("archive", "Best quality for storage", 20, 24, None, "192k"),
}

@dataclass
class Config:
    """Runtime configuration"""
    preset: Preset
    codec: str = "h264"
    format: str = "mov"
    use_gpu: bool = False
    trash: bool = False
    dry_run: bool = False
    interactive: bool = False
    suffix: str = "-re"
    audio_file: Optional[Path] = None
    debug: bool = False

# Global settings (to be deprecated in favor of Config)
RAW_LOG_FILE = ".automat-raw.log"
TRASH_MODE = False
DEBUG_MODE = False
DRY_RUN = False
SUFFIX = "-re"
DEFAULT_CODEC = "h264"
DEFAULT_FORMAT = "mov"

# ANSI colors for stderr tags
COLOR_RESET = "\x1b[0m"
COLOR_RED = "\x1b[31m"
COLOR_YELLOW = "\x1b[33m"

def color_tag(tag: str, color: str) -> str:
    """Color only when stderr is a TTY."""
    try:
        if sys.stderr.isatty():
            return f"{color}{tag}{COLOR_RESET}"
    except Exception:
        pass
    return tag

# ============================================================================
# MEDIA FILE ABSTRACTION
# ============================================================================

@dataclass
class VideoInfo:
    """Video metadata from ffprobe"""
    width: int
    height: int
    duration: float
    bitrate: int
    filesize: int

    @property
    def is_hd(self) -> bool:
        """Check if resolution is HD or higher (>1080p)"""
        return max(self.width, self.height) > 1080

    @property
    def is_low_bitrate(self) -> bool:
        """Check if video is already compressed (low bitrate)"""
        threshold = 2_500_000 if self.is_hd else 1_200_000
        return self.bitrate > 0 and self.bitrate < threshold

class MediaFile:
    """Represents a video or image file with cached metadata"""

    def __init__(self, path: Path):
        self.path = path if isinstance(path, Path) else Path(path)
        self._mime_type: Optional[str] = None
        self._video_info: Optional[VideoInfo] = None

    @property
    def mime_type(self) -> str:
        """Cached MIME type detection"""
        if self._mime_type is None:
            if not self.path.is_file():
                self._mime_type = ""
            else:
                res = run_command(["file", "--mime-type", "-b", str(self.path)])
                self._mime_type = res.stdout.strip()
        return self._mime_type

    def is_video(self) -> bool:
        """Check if file is a video"""
        return self.mime_type.startswith("video/")

    def is_image(self) -> bool:
        """Check if file is an image"""
        return self.mime_type.startswith("image/")

    def is_already_processed(self, suffix: str) -> bool:
        """Check if file was already processed by looking for the suffix"""
        return self.path.stem.endswith(suffix)

    def get_video_info(self) -> VideoInfo:
        """Get cached video metadata from ffprobe"""
        if self._video_info is None:
            logger.debug("Getting info for: %s", self.path)
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                   "-show_format", "-show_streams", str(self.path)]
            res = run_command(cmd)
            if res.returncode != 0:
                display_error(f"ffprobe error on {self.path}")
                self._video_info = VideoInfo(0, 0, 0.0, 0, 0)
            else:
                info = json.loads(res.stdout)
                stream = next((s for s in info.get("streams",[]) if s.get("codec_type")=="video"), {})
                width = stream.get("width", 0) or 0
                height = stream.get("height", 0) or 0
                duration = float(info.get("format", {}).get("duration", 0.0)) or 0.0
                filesize = self.path.stat().st_size
                bitrate = int(info.get("format", {}).get("bit_rate", 0) or 0)
                self._video_info = VideoInfo(width, height, duration, bitrate, filesize)
                logger.info(f"Info: {width}x{height}, {duration}s, {bitrate}b/s, {filesize} bytes")
        return self._video_info

    def recommend_preset(self) -> str:
        """Auto-recommend preset based on file size"""
        size_mb = self.path.stat().st_size / 1_000_000
        if size_mb < 10:
            return "share"  # Already small
        elif size_mb < 100:
            return "meme"   # Make it smaller
        else:
            return "archive"  # Preserve quality for large files

# ============================================================================
# UTILITY FUNCTIONS (Logging, Display, Commands)
# ============================================================================
#
# NOTE: The quality/CRF logic is now in CodecStrategy classes (see CODEC STRATEGIES section)
#       Presets are defined above in PRESETS dict
#
DEFAULT_PRESET = "slow"  # encoding speed vs compression efficiency
DEFAULT_CRF_H264 = 24    # Legacy - now defined in Preset dataclass
DEFAULT_CRF_HEVC = 29    # Legacy - now defined in Preset dataclass

logger = logging.getLogger(__name__)

def setup_logging(log_path=None):
    logger.handlers = []            # Remove any existing handlers
    logger.propagate = False        # Don't propagate messages to root logger
    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
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

# ============================================================================
# CODEC STRATEGIES
# ============================================================================

class CodecStrategy(ABC):
    """Base class for codec-specific encoding strategies"""

    @abstractmethod
    def build_video_options(self, config: Config, video_info: VideoInfo) -> str:
        """Build codec-specific ffmpeg video options string"""
        pass

    def get_crf_for_quality(self, base_crf: int, video_info: VideoInfo) -> int:
        """Adjust CRF based on source video bitrate (more aggressive for already-compressed)"""
        if video_info.is_low_bitrate:
            return base_crf + 4  # More aggressive compression
        return base_crf

    def get_dynamic_bitrate(self, video_info: VideoInfo) -> int:
        """Calculate dynamic bitrate for GPU encoding"""
        min_gpu_bitrate = 600_000 if video_info.is_hd else 450_000
        gpu_bitrate_thresh = 4_000_000 if video_info.is_hd else 2_500_000

        src_bitrate = video_info.bitrate if video_info.bitrate > 0 else gpu_bitrate_thresh

        if src_bitrate < 1_000_000:
            return max(int(src_bitrate * 0.85), min_gpu_bitrate)
        else:
            if src_bitrate > gpu_bitrate_thresh:
                chosen = int(max(0.8 * src_bitrate, gpu_bitrate_thresh))
                dynamic_bitrate = min(src_bitrate, chosen)
            else:
                dynamic_bitrate = src_bitrate
            if dynamic_bitrate < min_gpu_bitrate:
                dynamic_bitrate = min_gpu_bitrate
            return dynamic_bitrate

class H264CPUStrategy(CodecStrategy):
    """H.264 CPU encoding with libx264"""

    def build_video_options(self, config: Config, video_info: VideoInfo) -> str:
        crf = self.get_crf_for_quality(config.preset.h264_crf, video_info)
        return f"-c:v libx264 -preset slow -crf {crf}"

class H264GPUStrategy(CodecStrategy):
    """H.264 GPU encoding with videotoolbox"""

    def build_video_options(self, config: Config, video_info: VideoInfo) -> str:
        bitrate = self.get_dynamic_bitrate(video_info)
        bval = f"{int(bitrate // 1000)}k"
        return f"-c:v h264_videotoolbox -b:v {bval} -tag:v avc1"

class HEVCCPUStrategy(CodecStrategy):
    """HEVC/H.265 CPU encoding with libx265"""

    def build_video_options(self, config: Config, video_info: VideoInfo) -> str:
        crf = self.get_crf_for_quality(config.preset.hevc_crf, video_info)
        return f'-c:v libx265 -preset slow -crf {crf} -tag:v hvc1 -x265-params "psy-rd=2.0:psy-rdoq=1.0:aq-mode=3:aq-strength=1.0:ref=5:bframes=8:rc-lookahead=60"'

class HEVCGPUStrategy(CodecStrategy):
    """HEVC/H.265 GPU encoding with videotoolbox"""

    def build_video_options(self, config: Config, video_info: VideoInfo) -> str:
        bitrate = self.get_dynamic_bitrate(video_info)
        bval = f"{int(bitrate // 1000)}k"
        return f"-c:v hevc_videotoolbox -b:v {bval} -tag:v hvc1"

class AV1Strategy(CodecStrategy):
    """AV1 encoding with libaom-av1 (experimental)"""

    def build_video_options(self, config: Config, video_info: VideoInfo) -> str:
        return "-c:v libaom-av1 -crf 30 -b:v 0 -strict experimental"

# Codec strategy registry
CODEC_REGISTRY = {
    ("h264", True): H264CPUStrategy(),
    ("h264", False): H264GPUStrategy(),
    ("hevc", True): HEVCCPUStrategy(),
    ("hevc", False): HEVCGPUStrategy(),
    ("av1", True): AV1Strategy(),
    ("av1", False): AV1Strategy(),  # AV1 doesn't have GPU variant yet
}

def is_videotoolbox_available():
    """Check if videotoolbox hardware encoder is available via ffmpeg."""
    res = run_command(["ffmpeg", "-encoders"])
    return "videotoolbox" in res.stdout

def build_ffmpeg_command(src, codec, fmt, is_cpu=None, crf_value=None, dynamic_bitrate=None):
    src = Path(src)
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

# ============================================================================
# OPERATIONS BASE CLASS
# ============================================================================

@dataclass
class Result:
    """Result of a processing operation"""
    success: bool
    original_size: int = 0
    new_size: int = 0
    output_path: Optional[Path] = None

    @property
    def size_reduction_pct(self) -> float:
        """Calculate size reduction percentage"""
        if self.original_size > 0:
            return (1 - self.new_size / self.original_size) * 100
        return 0.0

    @property
    def original_size_mb(self) -> float:
        """Original size in MB"""
        return self.original_size / 1_000_000

    @property
    def new_size_mb(self) -> float:
        """New size in MB"""
        return self.new_size / 1_000_000

class Operation(ABC):
    """Base class for all media processing operations"""

    @abstractmethod
    def supports(self, media: MediaFile) -> bool:
        """Check if this operation supports the given media file"""
        pass

    @abstractmethod
    def execute(self, media: MediaFile, config: Config) -> Result:
        """Execute the operation on the media file"""
        pass

    def _validate_ffmpeg_output(self, output: Path, dry_run: bool = False) -> bool:
        """Common validation logic for FFmpeg outputs"""
        if dry_run:
            return True
        if not output.is_file() or output.stat().st_size == 0:
            display_error(f"Output missing or empty: {output}")
            return False
        return True

    def _cleanup_source(self, media: MediaFile, config: Config):
        """Common cleanup logic - move to trash if requested"""
        if config.trash and not config.dry_run:
            move_to_trash(media.path)

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
        src, codec, fmt,
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

def process_video_amv(src, audio_track, codec, fmt, is_cpu=None):
    """Add or replace audio track in video (AMV operation)"""
    src = Path(src)
    audio_track = Path(audio_track)
    out = src.parent / f"{src.stem}{SUFFIX}.{fmt}"

    if not audio_track.exists():
        display_error(f"Audio track not found: {audio_track}")
        return False

    # Build FFmpeg command for AMV operation
    cmd = ["ffmpeg"]
    if is_cpu is False:
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

def process_video_loop_audio(src, codec, fmt, is_cpu=None):
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
    if is_cpu is False:
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
                elif not process_video_amv(file_path, audio_track, codec, fmt, is_cpu=is_cpu):
                    success = False
            elif operation == "loop_audio":
                if not process_video_loop_audio(file_path, codec, fmt, is_cpu=is_cpu):
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
            # Single file - add if it's a media file (include already-processed too)
            if is_video_file(path) or is_image_file(path):
                expanded_files.append(path)
        elif path.is_dir():
            # Directory - recursively find all media files (include already-processed too)
            for file_path in path.rglob("*"):
                if file_path.is_file():
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
    """Get the base path for progress/error log files from the first path argument.
    If the first argument is a directory, use its parent directory.
    If it's a file, use its parent directory as well.
    Otherwise, default to the current working directory.
    """
    if not paths:
        return Path.cwd()

    first_path = Path(paths[0])
    if not first_path.is_absolute():
        first_path = Path.cwd() / first_path

    try:
        if first_path.is_file():
            return first_path.parent
        elif first_path.is_dir():
            return first_path.parent
        else:
            return Path.cwd()
    except Exception:
        # In case of any filesystem access issues, fall back to CWD
        return Path.cwd()

def update_progress_file(progress_file, total, done, skipped, failed):
    """
    Update the progress tracking file (.automat-progress.log) with new structure.
    """
    if DRY_RUN:
        display_info(f"[DRY RUN] Would update progress: {total}/{done}/{skipped}/{failed}")
        return
    try:
        with open(progress_file, 'w') as f:
            f.write(f"Total: {total}\n")
            f.write(f"Done: {done}\n")
            f.write(f"Skipped: {skipped}\n")
            f.write(f"Failed: {failed}\n")
    except Exception as e:
        display_error(f"Failed to update progress file: {e}")

def log_error_file(error_file, file_path):
    """
    Append failed file path to error log (.automat-failed.log).
    """
    if DRY_RUN:
        display_info(f"[DRY RUN] Would log error: {file_path}")
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
    progress_file = base_path / ".automat-progress.log"
    error_file = base_path / ".automat-failed.log"

    # Initialize logs: start fresh
    if not DRY_RUN:
        if progress_file.exists():
            try: progress_file.unlink()
            except Exception as e: display_error(f"Failed to remove existing progress log: {e}")
        if error_file.exists():
            try: error_file.unlink()
            except Exception as e: display_error(f"Failed to remove existing error log: {e}")
        try:
            progress_file.touch()
            error_file.touch()
        except Exception as e:
            display_error(f"Failed to create log files: {e}")

    # Counters
    proc = fail = skipped = 0
    update_progress_file(progress_file, total, proc, skipped, fail)

    for path in files:
        n = proc + skipped + fail + 1
        print(f"{n}/{total}: {path}", flush=True)
        if is_already_processed(path):
            print(f"{color_tag('[Skipped]', COLOR_YELLOW)} {path}", file=sys.stderr, flush=True)
            skipped += 1
            update_progress_file(progress_file, total, proc, skipped, fail)
            continue

        display_info(f"[{n}/{total}] Processing: {path}")
        if process_single_file(path, ["refine"], None, codec, fmt):
            proc += 1
        else:
            print(f"{color_tag('[Failed]', COLOR_RED)} {path}", file=sys.stderr, flush=True)
            fail += 1
            log_error_file(error_file, path)
        update_progress_file(progress_file, total, proc, skipped, fail)

    display_info(f"Done. Processed: {proc}, Skipped: {skipped}, Failed: {fail}")
    if not DRY_RUN:
        notify("Automat", f"Processed {proc} files, {fail} failed")
        # Auto-remove logs if no failures
        if fail == 0:
            for f in (progress_file, error_file):
                if f.exists():
                    try: f.unlink()
                    except Exception as e: display_error(f"Failed to remove log file {f}: {e}")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

class InteractiveMode:
    """Guided prompts for rare operations"""

    def run(self, files: List[Path], config: Config) -> Tuple[List[str], Config]:
        """
        Run interactive prompts and return selected operations and updated config.
        Returns: (operations_list, updated_config)
        """
        print("\nAutomat Interactive Mode")
        print("========================")
        print(f"Files to process: {len(files)}")
        print()
        print("What would you like to do?")
        print("1. Refine/compress (default)")
        print("2. Add audio track (AMV)")
        print("3. Loop audio to match video length")
        print("4. Extract audio to MP3")
        print("5. Create video from image + audio")
        print()

        choice = input("Choice [1]: ").strip() or "1"

        operations = []

        if choice == "1":
            # Refine operation - ask about preset
            print("\nQuality preset:")
            print("  meme    - Smallest size for sharing")
            print("  share   - Balanced for messaging apps (default)")
            print("  archive - Best quality for storage")
            preset_choice = input("Preset [share]: ").strip() or "share"
            if preset_choice in PRESETS:
                config.preset = PRESETS[preset_choice]
            operations.append("refine")

        elif choice == "2":
            # AMV operation
            audio = input("\nAudio file path: ").strip()
            if not audio:
                print("Error: Audio file required for AMV operation")
                return (["refine"], config)  # Fall back to refine
            config.audio_file = Path(audio)
            if not config.audio_file.exists():
                print(f"Error: Audio file not found: {audio}")
                return (["refine"], config)
            operations.append("amv")

        elif choice == "3":
            # Loop audio
            operations.append("loop_audio")

        elif choice == "4":
            # Audiofy
            operations.append("audiofy")

        elif choice == "5":
            # Image + audio to video
            audio = input("\nAudio file path: ").strip()
            if not audio:
                print("Error: Audio file required")
                return (["refine"], config)
            config.audio_file = Path(audio)
            if not config.audio_file.exists():
                print(f"Error: Audio file not found: {audio}")
                return (["refine"], config)
            operations.append("audiofy")

        else:
            print(f"Invalid choice: {choice}, defaulting to refine")
            operations.append("refine")

        # Common questions
        if operations:
            print()
            codec_choice = input(f"Output codec [h264]: ").strip() or "h264"
            if codec_choice in ["h264", "hevc", "av1"]:
                config.codec = codec_choice

            trash_choice = input("Move originals to trash? [y/N]: ").strip().lower()
            config.trash = trash_choice == "y"

        print(f"\nReady to process {len(files)} file(s)")
        print(f"Operations: {', '.join(operations)}")
        print(f"Codec: {config.codec}")
        print(f"Preset: {config.preset.name}")
        print(f"Trash originals: {'Yes' if config.trash else 'No'}")
        print()
        confirm = input("Proceed? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("Cancelled.")
            sys.exit(0)

        return (operations, config)

def main():
    """
    Main entry point for Automat.

    Progress and error logs:
      - Progress and error logs are only created and updated for long runs (over 3 minutes).
      - Progress log: .automat-progress.log in the first input's directory.
      - Error log: .automat-failed.log in the first input's directory.
      - These files are NOT created at start. They are only created/updated if the run exceeds 3 minutes.
      - At the end, if these files exist and there were no errors, they are deleted.
    """
    global ENABLE_LOGGING, TRASH_MODE, DEBUG_MODE, DRY_RUN, SUFFIX
    p = argparse.ArgumentParser(
        description=(
            "Automat: refine videos and images using CPU or hardware-accelerated (GPU) encoding.\n\n"
            "This tool optimizes media files by re-encoding them with efficient codecs.\n"
            "Videos are processed with ffmpeg using CPU encoding by default (better compression, slower).\n"
            "Use --use-gpu to enable hardware-accelerated (videotoolbox) encoding for faster results (less compression).\n"
            "Images are converted to HEIC format using sips for better compression.\n\n"
            "Progress and error logs (.automat-progress.log, .automat-failed.log) are only created for long runs (over 3 minutes) and are always located in the first input's directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Quick refine (most common use case):\n"
            "  automat.py video.mp4\n\n"
            "  # Smallest size for memes:\n"
            "  automat.py --preset meme funny.mp4\n\n"
            "  # Batch process folder and move originals to trash:\n"
            "  automat.py -t ~/Downloads/videos/\n\n"
            "  # Auto-detect best preset based on file size:\n"
            "  automat.py --auto large_file.mov\n\n"
            "  # Interactive mode for advanced operations:\n"
            "  automat.py -i video.mp4\n\n"
            "  # Create AMV with new audio track:\n"
            "  automat.py --amv -a new_track.mp3 video.avi\n\n"
            "  # Use GPU for faster encoding:\n"
            "  automat.py --use-gpu video.mp4\n\n"
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
            "automat.py -t \"$@\"\n"
            "```\n"
        )
    )

    # Mode selection
    p.add_argument("-i", "--interactive", action="store_true",
                   help="Interactive mode with guided prompts for operations")

    # Operation flags
    p.add_argument("--refine", action="store_true",
                   help="Refine/optimize video or image files (default operation)")
    p.add_argument("--amv", action="store_true",
                   help="Add or replace audio track in video (requires -a)")
    p.add_argument("--loop-audio", dest="loop_audio", action="store_true",
                   help="Loop audio to match video duration")
    p.add_argument("--audiofy", action="store_true",
                   help="Extract audio from video as MP3, or create video from image+audio")

    # Quality presets
    p.add_argument("--preset", choices=["meme", "share", "archive"], default="share",
                   help="Quality preset: meme (smallest), share (balanced, default), archive (best quality)")
    p.add_argument("--auto", action="store_true",
                   help="Auto-detect best preset based on file size")

    # Configuration options
    p.add_argument("-a", "--audio", metavar="FILE",
                   help="Audio file for AMV/audiofy operations")
    p.add_argument("-c", "--codec", choices=["h264", "hevc", "av1"], default="h264",
                   help="Video codec: h264 (most compatible, default), hevc (better compression), av1 (experimental)")
    p.add_argument("-f", "--format", default=DEFAULT_FORMAT, metavar="FORMAT",
                   help=f"Output format (mov, mp4, mkv, webm) [default: {DEFAULT_FORMAT}]")
    p.add_argument("-s", "--suffix", default=SUFFIX,
                   help=f"Custom suffix for output files [default: {SUFFIX}]")

    # Behavior flags
    p.add_argument("--use-gpu", action="store_true",
                   help="Use hardware acceleration (videotoolbox) for faster encoding")
    p.add_argument("-t", "--trash", action="store_true",
                   help="Move original files to trash after processing")
    p.add_argument("-d", "--debug", action="store_true",
                   help="Debug mode (extra logging)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Dry-run mode (show what would happen without processing)")

    # File arguments
    p.add_argument("files", nargs="+",
                   help="Files or directories to process")

    args = p.parse_args()

    # Validate argument dependencies
    if args.amv and not args.audio:
        p.error("--amv requires -a/--audio AUDIO_FILE")

    # Set global flags (for backward compatibility with existing functions)
    DEBUG_MODE = args.debug
    TRASH_MODE = args.trash
    DRY_RUN = args.dry_run
    SUFFIX = args.suffix

    # Build configuration object
    preset = PRESETS[args.preset]
    config = Config(
        preset=preset,
        codec=args.codec,
        format=args.format,
        use_gpu=args.use_gpu,
        trash=args.trash,
        dry_run=args.dry_run,
        interactive=args.interactive,
        suffix=args.suffix,
        audio_file=Path(args.audio) if args.audio else None,
        debug=args.debug,
    )

    # Check GPU availability
    if config.use_gpu:
        if not is_videotoolbox_available():
            display_info("GPU encoding requested but videotoolbox not available. Falling back to CPU.")
            config.use_gpu = False

    # Determine which operations to perform
    operations = []

    if config.interactive:
        # Interactive mode will determine operations
        pass  # Handled below after file discovery
    else:
        # CLI mode - collect operations from flags
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

    # First, recursively expand all paths into a flat list of media files
    if not config.interactive:
        display_info("Expanding paths and discovering media files...")
    all_files = expand_paths_recursively(args.files)

    if not all_files:
        display_error("No media files found to process")
        return 1

    total_files = len(all_files)

    # Handle interactive mode
    if config.interactive:
        interactive = InteractiveMode()
        operations, config = interactive.run([Path(f) for f in all_files], config)
        # Update global flags based on interactive config
        TRASH_MODE = config.trash
        DRY_RUN = config.dry_run
        SUFFIX = config.suffix
    elif args.auto and total_files == 1:
        # Auto-detect preset for single file
        media = MediaFile(Path(all_files[0]))
        recommended = media.recommend_preset()
        print(f"Auto-detected preset: {recommended} ({PRESETS[recommended].description})")
        config.preset = PRESETS[recommended]

    if not config.interactive:
        display_info(f"Found {total_files} media files to process")

    # Set up progress, error, and raw log tracking
    base_path = get_base_path(args.files)
    progress_file = base_path / ".automat-progress.log"
    error_file = base_path / ".automat-failed.log"
    raw_log_file = base_path / ".automat-raw.log"

    # Remove existing log files to start fresh (only if they exist)
    if not DRY_RUN:
        for f in (progress_file, error_file, raw_log_file):
            if f.exists():
                try:
                    f.unlink()
                except Exception as e:
                    display_error(f"Failed to remove existing log file {f}: {e}")
        # Create empty progress and error log files at start
        try:
            progress_file.touch()
            error_file.touch()
        except Exception as e:
            display_error(f"Failed to create log files: {e}")

    # Always log to RAW_LOG_FILE in the base_path
    setup_logging(str(raw_log_file))

    processed = 0
    failed = 0
    skipped = 0

    process_start = time.time()

    # Process each file
    for file_path in all_files:
        n = processed + skipped + failed + 1
        print(f"{n}/{total_files}: {file_path}", flush=True)
        if is_already_processed(file_path):
            print(f"{color_tag('[Skipped]', COLOR_YELLOW)} {file_path}", file=sys.stderr, flush=True)
            skipped += 1
            update_progress_file(progress_file, total_files, processed, skipped, failed)
            continue
        ok = process_single_file(file_path, operations, config.audio_file, config.codec, config.format, is_cpu=not config.use_gpu)
        if ok:
            processed += 1
        else:
            print(f"{color_tag('[Failed]', COLOR_RED)} {file_path}", file=sys.stderr, flush=True)
            failed += 1
            log_error_file(error_file, file_path)
        update_progress_file(progress_file, total_files, processed, skipped, failed)

    # Print summary to raw log (not stdout)
    display_info(f"Processing complete. Success: {processed}, Skipped: {skipped}, Failed: {failed}")

    # Show notification when complete
    if not DRY_RUN and total_files > 1:
        notify("Automat", f"Processed {processed} files, {failed} failed")

    # If there were no errors, delete all three log files
    if not DRY_RUN and failed == 0:
        for f in (progress_file, error_file, raw_log_file):
            if f.exists():
                try:
                    f.unlink()
                except Exception as e:
                    display_error(f"Failed to remove log file {f}: {e}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
