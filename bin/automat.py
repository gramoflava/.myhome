#!/usr/bin/env python3
"""
automat.py - Self-contained media processing tool for macOS

A single-file script for optimizing videos and images with ffmpeg/sips.
No external Python dependencies required (only stdlib).

ARCHITECTURE:
  - Configuration & Presets: Quality presets (meme/share/archive) + Config dataclass
  - Media File Abstraction: MediaFile class with cached metadata + VideoInfo
  - Processed File Tracking: embedded metadata tag (automat JSON signature)
  - Codec Strategies: Pluggable codec strategies (H264/HEVC CPU/GPU, AV1)
  - Operations: Base Operation class for refine/amv/audiofy/loop_audio
  - Interactive Mode: Guided prompts for rare operations
  - CLI Interface: Simplified argument parsing with preset support

USAGE:
  Most common: automat.py video.mp4                        # Quick refine with defaults
  Memes:       automat.py --preset meme funny.mp4          # Smallest size
  Batch:       automat.py -t ~/Downloads/videos/           # Process folder, trash backups
  Interactive: automat.py -i video.mp4                     # Guided prompts
  Auto:        automat.py --auto large_file.mov            # Auto-detect best preset
  AMV:         automat.py --amv video.mov -a audio.mp3    # Add audio (creates video-amv.mov)
  Extract:     automat.py --audiofy video.mov              # Extract audio (creates video-audio.mp3)

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
import secrets
import signal
import atexit
import glob
import mimetypes
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
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
    format: str = "mp4"  # Default format for video output
    use_gpu: bool = False
    trash_backups: bool = False
    dry_run: bool = False
    interactive: bool = False
    audio_file: Optional[Path] = None
    log: bool = False

# Global constants
DEFAULT_CODEC = "h264"
DEFAULT_FORMAT = "mp4"
ORIGINAL_PREFIX = "~"
VIDEO_METADATA_KEY = "comment"  # Standard metadata field for all video formats
IMAGE_METADATA_KEY = "description"
IMAGE_METADATA_PREFIX = "automat:"
METADATA_VERSION = 1

# Global runtime state (set by main())
_GLOBAL_CONFIG: Optional['Config'] = None
_SESSION_CODE: Optional[str] = None  # Random code for this run (generated once)

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
        self._mime_type: str = ""
        self._video_info: Optional[VideoInfo] = None

    @property
    def mime_type(self) -> str:
        """Cached MIME type detection"""
        if not self._mime_type:
            if not self.path.is_file():
                self._mime_type = ""
            else:
                if _GLOBAL_CONFIG and _GLOBAL_CONFIG.dry_run:
                    guessed, _ = mimetypes.guess_type(str(self.path))
                    self._mime_type = guessed or ""
                else:
                    res = run_command(["file", "--mime-type", "-b", str(self.path)])
                    self._mime_type = res.stdout.strip()
                    if not self._mime_type:
                        guessed, _ = mimetypes.guess_type(str(self.path))
                        self._mime_type = guessed or ""
        return self._mime_type

    def is_video(self) -> bool:
        """Check if file is a video"""
        return self.mime_type.startswith("video/")

    def is_image(self) -> bool:
        """Check if file is an image"""
        return self.mime_type.startswith("image/")

    def is_already_processed(self, signature: dict) -> bool:
        """Check if file was already processed by metadata signature"""
        return is_already_processed(self.path, signature)

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

def setup_logging(log_path=None, debug=False):
    logger.handlers = []            # Remove any existing handlers
    logger.propagate = False        # Don't propagate messages to root logger
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
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

def display_warning(message):
    logger.warning(message)

def display_info(message):
    logger.info(message)

def display_debug(message):
    if _GLOBAL_CONFIG and _GLOBAL_CONFIG.log:
        logger.debug(message)

def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    if _GLOBAL_CONFIG and _GLOBAL_CONFIG.dry_run:
        display_info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)

PENDING_TEMP_OUTPUTS: set[Path] = set()

def _normalize_ext(ext: str) -> str:
    return ext[1:] if ext.startswith(".") else ext

def _base_name(path: Path) -> str:
    name = path.name
    if name.startswith(ORIGINAL_PREFIX):
        name = name[len(ORIGINAL_PREFIX):]
    return Path(name).stem

def make_temp_output_path(src: Path, ext: str) -> Path:
    """
    Generate a unique temporary output path using session code.
    If session code is not set, falls back to random token.
    Uses exclusive file creation to prevent race conditions.
    """
    ext = _normalize_ext(ext)

    # Use session code if available, otherwise generate random token
    if _SESSION_CODE:
        candidate = src.parent / f"{_base_name(src)}-automat-{_SESSION_CODE}.{ext}"
        # Check if this path is already taken (shouldn't happen in normal flow)
        if not candidate.exists():
            return candidate
        # Fallback: append additional random suffix if collision
        token = secrets.token_hex(2)
        candidate = src.parent / f"{_base_name(src)}-automat-{_SESSION_CODE}-{token}.{ext}"
        return candidate

    # Fallback for when session code not set (shouldn't happen)
    max_attempts = 100
    for attempt in range(max_attempts):
        token = secrets.token_hex(4)
        candidate = src.parent / f"{_base_name(src)}-automat-{token}.{ext}"
        try:
            # Try to create an empty marker file with O_EXCL to ensure atomicity
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            # Immediately remove it - we just wanted to reserve the name
            candidate.unlink()
            return candidate
        except FileExistsError:
            # Collision - try again with new token
            continue
        except Exception as e:
            # Other error - just check existence as fallback
            display_debug(f"Warning: atomic create failed ({e}), falling back to exists check")
            if not candidate.exists():
                return candidate
    # Should never happen with 100 attempts and random hex tokens
    raise RuntimeError(f"Failed to generate unique temp path after {max_attempts} attempts")

def final_output_path(src: Path, ext: str, suffix: str = "") -> Path:
    """
    Generate final output path for processed file.

    Args:
        src: Source file path
        ext: Output extension (e.g., 'mov', 'mp3')
        suffix: Optional suffix to add before extension (e.g., '-amv', '-loop')

    Returns:
        Path like: src/basename-suffix.ext or src/basename.ext if no suffix
    """
    ext = _normalize_ext(ext)
    base = _base_name(src)
    if suffix:
        return src.parent / f"{base}{suffix}.{ext}"
    return src.parent / f"{base}.{ext}"

def prefixed_original_path(src: Path) -> Path:
    if src.name.startswith(ORIGINAL_PREFIX):
        return src
    return src.parent / f"{ORIGINAL_PREFIX}{src.name}"

def safe_unlink(path: Path, label: str = "temp file"):
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception as e:
        display_error(f"Failed to remove {label}: {path} ({e})")

def register_temp_output(path: Path):
    PENDING_TEMP_OUTPUTS.add(path)

def unregister_temp_output(path: Path):
    PENDING_TEMP_OUTPUTS.discard(path)

def cleanup_pending_temps():
    for path in list(PENDING_TEMP_OUTPUTS):
        safe_unlink(path, label="temp output")
        PENDING_TEMP_OUTPUTS.discard(path)

def install_signal_handlers():
    atexit.register(cleanup_pending_temps)
    def _handle_exit(signum, frame):
        cleanup_pending_temps()
        sys.exit(1)
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

def describe_file(path: Path) -> dict:
    info = {"path": str(path)}
    try:
        stat = path.stat()
        info["size"] = stat.st_size
        info["mtime"] = int(stat.st_mtime)
    except FileNotFoundError:
        info["size"] = None
        info["mtime"] = None
    return info

def build_run_signature(config: Config, operation: str, encoding_params: Optional[dict] = None) -> dict:
    """
    Build signature for this processing run.

    Args:
        config: Runtime configuration
        operation: The operation being performed (refine, amv, etc)
        encoding_params: Optional dict with actual encoding parameters:
            - crf: CRF value for CPU encoding
            - bitrate: Bitrate for GPU encoding
            - resolution: Output resolution (width, height)
            - audio_bitrate: Audio bitrate string
            - audio_codec: Audio codec string
    """
    signature = {
        "version": METADATA_VERSION,
        "operation": operation,
        "codec": config.codec,
        "format": _normalize_ext(config.format),
        "use_gpu": config.use_gpu,
        "preset": config.preset.name,
    }

    # Add actual encoding parameters if provided (for refine operation)
    if encoding_params:
        if encoding_params.get("crf") is not None:
            signature["crf"] = encoding_params["crf"]
        if encoding_params.get("bitrate") is not None:
            signature["bitrate"] = encoding_params["bitrate"]
        if encoding_params.get("resolution"):
            signature["resolution"] = encoding_params["resolution"]
        if encoding_params.get("audio_bitrate"):
            signature["audio_bitrate"] = encoding_params["audio_bitrate"]
        if encoding_params.get("audio_codec"):
            signature["audio_codec"] = encoding_params["audio_codec"]

    # Only include audio file hash for amv/audiofy operations (not needed for refine)
    if operation in {"amv", "audiofy"} and config.audio_file:
        signature["audio"] = describe_file(config.audio_file)

    return signature

def metadata_payload(metadata: dict) -> str:
    return json.dumps(metadata, separators=(",", ":"), sort_keys=True)

def _run_readonly(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)

def validate_signature(signature: Any) -> bool:
    """Validate that signature has all required fields"""
    if not isinstance(signature, dict):
        return False
    required_fields = {"version", "operation", "codec", "format", "use_gpu", "preset"}
    return required_fields.issubset(signature.keys())

class MetadataBackend:
    """Unified metadata read/write backend for embedded metadata in media files"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or _GLOBAL_CONFIG

    def read(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read metadata from file, validating structure"""
        if is_video_file(path) or is_audio_file(path):
            meta = self._read_video_metadata(path)
        elif is_image_file(path):
            meta = self._read_image_metadata(path)
        else:
            return None

        # Validate metadata structure
        if meta and isinstance(meta, dict):
            # Check that signature is valid
            if "signature" in meta and validate_signature(meta["signature"]):
                return meta
        return None

    def write(self, path: Path, metadata: Dict[str, Any]) -> bool:
        """Write metadata to file"""
        if self.config and self.config.dry_run:
            display_info(f"[DRY RUN] Would write metadata to: {path}")
            return True

        # Only images need explicit metadata write (videos get it via ffmpeg)
        if not is_image_file(path):
            display_error(f"Metadata write unsupported for non-image: {path}")
            return False

        try:
            payload = metadata_payload(metadata)
        except (TypeError, ValueError) as e:
            display_error(f"Failed to serialize metadata for {path}: {e}")
            return False

        value = f"{IMAGE_METADATA_PREFIX}{payload}"
        cmd = ["sips", "-s", IMAGE_METADATA_KEY, value, str(path)]
        res = _run_readonly(cmd)
        if res.returncode != 0:
            display_error(f"Failed to write metadata to {path}: {res.stderr.strip()}")
            return False
        return True

    def _read_video_metadata(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read metadata from video/audio file using standard 'comment' field"""
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_entries", "format_tags", str(path)]
        res = _run_readonly(cmd)
        if res.returncode != 0 or not res.stdout.strip():
            return None
        try:
            info = json.loads(res.stdout)
        except json.JSONDecodeError:
            return None
        tags = info.get("format", {}).get("tags", {}) or {}
        if not isinstance(tags, dict):
            return None

        # Case-insensitive search for the comment field
        for key, value in tags.items():
            if isinstance(key, str) and key.lower() == VIDEO_METADATA_KEY.lower():
                if not value:
                    return None
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return None
        return None

    def _read_image_metadata(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read metadata from image file with case-insensitive key search"""
        cmd = ["sips", "-g", IMAGE_METADATA_KEY, "-1", str(path)]
        res = _run_readonly(cmd)
        if res.returncode != 0 or not res.stdout.strip():
            return None
        value = None
        for line in res.stdout.splitlines():
            line = line.strip()
            # Case-insensitive search for the key
            if ":" in line:
                line_key, line_value = line.split(":", 1)
                if line_key.strip().lower() == IMAGE_METADATA_KEY.lower():
                    value = line_value.strip()
                    break
        if not value or value in {"(null)", "null"}:
            return None
        if not value.startswith(IMAGE_METADATA_PREFIX):
            return None
        raw = value[len(IMAGE_METADATA_PREFIX):]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

def insert_ffmpeg_metadata(cmd: list[str], metadata: dict) -> list[str]:
    """
    Insert metadata into ffmpeg command.
    Metadata must be placed before -y flag but after all other encoding options.
    Uses standard 'comment' field for all formats.
    """
    payload = metadata_payload(metadata)

    # Find the position of -y flag
    try:
        y_index = cmd.index("-y")
        # Insert metadata before -y
        return cmd[:y_index] + ["-metadata", f"{VIDEO_METADATA_KEY}={payload}"] + cmd[y_index:]
    except ValueError:
        # No -y flag found, append before output file (last argument)
        return cmd[:-1] + ["-metadata", f"{VIDEO_METADATA_KEY}={payload}"] + [cmd[-1]]

# Legacy wrapper functions (for backward compatibility)
def read_video_metadata(path: Path) -> Optional[dict]:
    """Legacy wrapper - use MetadataBackend.read() instead"""
    backend = MetadataBackend()
    return backend._read_video_metadata(path)

def read_image_metadata(path: Path) -> Optional[dict]:
    """Legacy wrapper - use MetadataBackend.read() instead"""
    backend = MetadataBackend()
    return backend._read_image_metadata(path)

def read_processing_metadata(path: Path) -> Optional[dict]:
    """Legacy wrapper - use MetadataBackend.read() instead"""
    backend = MetadataBackend()
    return backend.read(path)

def write_processing_metadata(path: Path, metadata: dict) -> bool:
    """Legacy wrapper - use MetadataBackend.write() instead"""
    backend = MetadataBackend()
    return backend.write(path, metadata)

def is_already_processed(path: Path, signature: dict) -> bool:
    meta = read_processing_metadata(path)
    if not meta or not isinstance(meta, dict):
        return False
    return meta.get("signature") == signature

def check_file_status(path: Path, config: Config, operation: str) -> dict:
    """
    Check processing status of a file.
    Returns dict with:
      - processed: bool - whether file has automat metadata
      - metadata: dict | None - full metadata if present
      - will_reprocess: bool - whether file would be processed again with current config
      - reason: str - explanation

    For refine operation, calculates what encoding parameters WOULD be used
    and compares them to stored signature for accurate reprocess detection.
    """
    backend = MetadataBackend(config)
    meta = backend.read(path)

    if not meta:
        return {
            "processed": False,
            "metadata": None,
            "will_reprocess": True,
            "reason": "No automat metadata found - file will be processed"
        }

    signature = meta.get("signature", {})

    # For refine operation, calculate what encoding params would be used
    if operation == "refine" and (is_video_file(path) or is_audio_file(path)):
        try:
            media = MediaFile(path)
            video_info = media.get_video_info()
            w, h = video_info.width, video_info.height
            br = video_info.bitrate

            is_hd = max(w, h) > 1080
            low_bitrate_thresh = 2_500_000 if is_hd else 1_200_000
            gpu_bitrate_thresh = 4_000_000 if is_hd else 2_500_000
            min_gpu_bitrate = 600_000 if is_hd else 450_000

            is_cpu = not config.use_gpu
            crf_value = None
            dynamic_bitrate = None

            if is_cpu:
                if config.codec == "h264":
                    crf_value = 28 if (br > 0 and br < low_bitrate_thresh) else 24
                elif config.codec == "hevc":
                    crf_value = 32 if (br > 0 and br < low_bitrate_thresh) else 28
            else:
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

            encoding_params = {
                "crf": crf_value,
                "bitrate": dynamic_bitrate,
                "resolution": (w, h),
                "audio_bitrate": "96k",
                "audio_codec": "aac",
            }
            current_signature = build_run_signature(config, operation, encoding_params)
        except Exception:
            # If we can't get video info, fall back to basic signature
            current_signature = build_run_signature(config, operation)
    else:
        current_signature = build_run_signature(config, operation)

    if signature == current_signature:
        return {
            "processed": True,
            "metadata": meta,
            "will_reprocess": False,
            "reason": "Already processed with identical settings - will be skipped"
        }

    # Build detailed reason for reprocessing with all relevant fields
    diffs = []
    all_keys = set(signature.keys()) | set(current_signature.keys())
    # Order keys logically
    key_order = ["operation", "codec", "format", "use_gpu", "preset", "crf", "bitrate", "resolution", "audio_bitrate", "audio_codec"]
    for key in key_order:
        if key in all_keys and key != "version":
            old_val = signature.get(key)
            new_val = current_signature.get(key)
            if old_val != new_val:
                # Format values nicely
                if key == "bitrate" and new_val is not None:
                    new_val_str = f"{new_val // 1000}k" if isinstance(new_val, int) else str(new_val)
                    old_val_str = f"{old_val // 1000}k" if isinstance(old_val, int) and old_val is not None else str(old_val)
                    diffs.append(f"{key}: {old_val_str} → {new_val_str}")
                elif key == "resolution":
                    old_val_str = f"{old_val[0]}x{old_val[1]}" if old_val else "None"
                    new_val_str = f"{new_val[0]}x{new_val[1]}" if new_val else "None"
                    diffs.append(f"{key}: {old_val_str} → {new_val_str}")
                else:
                    diffs.append(f"{key}: {old_val} → {new_val}")

    if not diffs:
        reason = "Settings appear unchanged but signature mismatch detected"
    else:
        reason = "Settings changed - file will be reprocessed:\n  " + "\n  ".join(diffs)

    return {
        "processed": True,
        "metadata": meta,
        "will_reprocess": True,
        "reason": reason
    }

def ensure_metadata_present(path: Path, signature: Optional[dict]) -> bool:
    """
    Verify that metadata is present and matches signature.
    Graceful degradation: if metadata can't be read, log warning but don't fail.

    Note: Files without readable metadata will NOT be skipped on subsequent runs,
    as the system won't be able to detect they've already been processed.
    """
    if not signature:
        display_error(f"Missing signature for metadata verification: {path}")
        return False

    backend = MetadataBackend()
    meta = backend.read(path)

    if not meta:
        # Graceful degradation: metadata might not be supported for this format
        # Upgrade to warning level so user is aware of the limitation
        display_warning(
            f"Could not read metadata from {path}. "
            f"File will be reprocessed on next run (doneness tracking unavailable)."
        )
        return True

    if meta.get("signature") != signature:
        display_error(f"Processed metadata mismatched (expected signature not found): {path}")
        return False

    return True

def finalize_processed_output(
    src: Path,
    temp_out: Path,
    final_out: Path,
    config: Config,
    metadata: dict,
    rename_original: bool = False,
) -> bool:
    """
    Finalize processed output with atomic operations and full rollback on failure.

    Behavior:
      - If rename_original=True (REFINE):
        1. Verify metadata in temp_out
        2. Rename src -> ~src (prefixed backup)
        3. Move temp_out -> final_out (replaces src)
        4. Optionally trash ~src if --trash-backups

      - If rename_original=False (AMV/AUDIOFY/LOOP/etc):
        1. Verify metadata in temp_out
        2. Move temp_out -> final_out (new file with suffix)
        3. Keep src completely untouched

    Args:
        rename_original: If True, rename source to ~source (for refine operation).
                        If False (default), keep original untouched (for amv/portrait/audiofy operations).
    """
    prefixed_src = prefixed_original_path(src) if rename_original else None

    if config.dry_run:
        if rename_original:
            display_info(f"[DRY RUN] Would finalize: {src} -> {prefixed_src}, {temp_out} -> {final_out}")
        else:
            display_info(f"[DRY RUN] Would finalize: {temp_out} -> {final_out} (keeping {src} untouched)")
        unregister_temp_output(temp_out)
        return True

    if not ensure_metadata_present(temp_out, metadata.get("signature")):
        cleanup_temp_output(temp_out)
        return False

    # Track state for rollback
    src_was_renamed = False
    final_out_existed = final_out.exists()
    final_out_backup = None

    try:
        # Step 1: Rename source to prefixed FIRST (only if rename_original=True)
        # This prevents issues when final_out == src (same base name)
        if rename_original and prefixed_src and prefixed_src != src:
            src.replace(prefixed_src)
            src_was_renamed = True

        # Step 2: Backup existing final_out if it exists
        # Only needed if we're replacing an existing file (not the original source)
        if final_out_existed and final_out.exists():
            # Don't backup if final_out was the original source that we just renamed
            if not (rename_original and final_out.resolve() == src.resolve()):
                final_out_backup = final_out.parent / f".automat-backup-{uuid.uuid4().hex}{final_out.suffix}"
                final_out.replace(final_out_backup)

        # Step 3: Move temp output to final location
        try:
            temp_out.replace(final_out)
        except OSError as move_error:
            # If replace fails, try shutil.move as fallback
            display_debug(f"Path.replace failed ({move_error}), trying shutil.move")
            try:
                shutil.move(str(temp_out), str(final_out))
            except OSError as shutil_error:
                if shutil_error.errno == 1:  # Operation not permitted
                    raise OSError(
                        f"Permission denied when moving file. "
                        f"If using Automator, grant Full Disk Access in System Settings → Privacy & Security. "
                        f"Original error: {shutil_error}"
                    ) from shutil_error
                else:
                    raise

        # Success - cleanup backup if it exists
        if final_out_backup and final_out_backup.exists():
            safe_unlink(final_out_backup, label="backup file")

    except Exception as e:
        display_error(f"Finalize failed for {src}: {e}")

        # Rollback: restore original state
        # 1. Restore final_out from backup
        if final_out_backup and final_out_backup.exists():
            try:
                final_out_backup.replace(final_out)
            except Exception as restore_error:
                logger.warning(f"Failed to restore backup: {final_out_backup} -> {final_out} ({restore_error})")

        # 2. Restore src from prefixed_src (only if it was renamed)
        if src_was_renamed and prefixed_src and prefixed_src.exists() and not src.exists():
            try:
                prefixed_src.replace(src)
            except Exception as restore_error:
                logger.warning(f"Failed to restore original: {prefixed_src} -> {src} ({restore_error})")

        # 3. Cleanup temp output
        safe_unlink(temp_out, label="temp output")

        return False
    finally:
        unregister_temp_output(temp_out)

    # Move backup to trash if requested (only for refine operation)
    if config.trash_backups and rename_original and prefixed_src:
        # Trash the prefixed backup (~src)
        move_to_trash(prefixed_src)
    return True

def cleanup_temp_output(temp_out: Path):
    unregister_temp_output(temp_out)
    safe_unlink(temp_out, label="temp output")

def mark_original_processed(src: Path, config: Config, metadata: dict) -> bool:
    prefixed_src = prefixed_original_path(src)
    if config.dry_run:
        display_info(f"[DRY RUN] Would mark processed: {src} -> {prefixed_src}")
        return True
    try:
        if prefixed_src != src:
            src.replace(prefixed_src)
    except Exception as e:
        display_error(f"Failed to rename original {src}: {e}")
        return False
    if not write_processing_metadata(prefixed_src, metadata):
        try:
            if prefixed_src != src and not src.exists():
                prefixed_src.replace(src)
        except Exception as restore_error:
            logger.warning(f"Failed to restore original: {prefixed_src} -> {src} ({restore_error})")
        return False
    if config.trash_backups:
        move_to_trash(prefixed_src)
    return True

def is_video_file(path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    if _GLOBAL_CONFIG and _GLOBAL_CONFIG.dry_run:
        guessed, _ = mimetypes.guess_type(str(path))
        return (guessed or "").startswith("video/")
    res = run_command(["file", "--mime-type", "-b", str(path)])
    mime = res.stdout.strip()
    if not mime:
        guessed, _ = mimetypes.guess_type(str(path))
        mime = guessed or ""
    return mime.startswith("video/")

def is_audio_file(path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    if _GLOBAL_CONFIG and _GLOBAL_CONFIG.dry_run:
        guessed, _ = mimetypes.guess_type(str(path))
        return (guessed or "").startswith("audio/")
    res = run_command(["file", "--mime-type", "-b", str(path)])
    mime = res.stdout.strip()
    if not mime:
        guessed, _ = mimetypes.guess_type(str(path))
        mime = guessed or ""
    return mime.startswith("audio/")

def is_image_file(path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    if _GLOBAL_CONFIG and _GLOBAL_CONFIG.dry_run:
        guessed, _ = mimetypes.guess_type(str(path))
        return (guessed or "").startswith("image/")
    res = run_command(["file", "--mime-type", "-b", str(path)])
    mime = res.stdout.strip()
    if not mime:
        guessed, _ = mimetypes.guess_type(str(path))
        mime = guessed or ""
    return mime.startswith("image/")

def move_to_trash(path):
    if _GLOBAL_CONFIG and _GLOBAL_CONFIG.dry_run:
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
    if _GLOBAL_CONFIG and _GLOBAL_CONFIG.dry_run:
        filesize = source.stat().st_size if source.exists() else 0
        return 0, 0, 0.0, 0, filesize
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(source)]
    res = run_command(cmd)
    if res.returncode != 0 or not res.stdout.strip():
        display_error(f"ffprobe error on {source}")
        filesize = source.stat().st_size if source.exists() else 0
        return 0,0,0.0,0,filesize
    try:
        info = json.loads(res.stdout)
    except json.JSONDecodeError:
        display_error(f"ffprobe returned invalid JSON for {source}")
        filesize = source.stat().st_size if source.exists() else 0
        return 0,0,0.0,0,filesize
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

def build_ffmpeg_command(src, codec, fmt, out, is_cpu=None, crf_value=None, dynamic_bitrate=None):
    src = Path(src)
    out = Path(out)

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
        if config.trash_backups and not config.dry_run:
            move_to_trash(media.path)

def process_video_refine(src, config: Config, signature: dict):
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

    is_cpu = not config.use_gpu

    if is_cpu:
        # Aggressive CRF selection for CPU encoder
        if config.codec == "h264":
            if br > 0 and br < low_bitrate_thresh:
                crf_value = 28
            else:
                crf_value = 24
        elif config.codec == "hevc":
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
    temp_out = make_temp_output_path(src, config.format)
    final_out = final_output_path(src, config.format)
    register_temp_output(temp_out)

    # Build enriched signature with actual encoding parameters
    encoding_params = {
        "crf": crf_value,
        "bitrate": dynamic_bitrate,
        "resolution": (w, h),
        "audio_bitrate": "96k",
        "audio_codec": "aac",
    }
    enriched_signature = build_run_signature(config, "refine", encoding_params)

    details = {
        "input": {
            "width": w,
            "height": h,
            "duration": dur,
            "bitrate": br,
            "filesize": sz,
        },
        "encoding": {
            "codec": config.codec,
            "format": config.format,
            "mode": "cpu" if is_cpu else "gpu",
            "crf": crf_value,
            "bitrate": dynamic_bitrate,
            "audio_bitrate": "96k",
        },
    }
    metadata = {"signature": enriched_signature, "details": details}
    cmd, _ = build_ffmpeg_command(
        src, config.codec, config.format, temp_out,
        is_cpu=is_cpu, crf_value=crf_value, dynamic_bitrate=dynamic_bitrate
    )
    cmd = insert_ffmpeg_metadata(cmd, metadata)
    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not config.dry_run:
        display_error("ffmpeg failed")
        cleanup_temp_output(temp_out)
        return False
    if not config.dry_run and (not temp_out.is_file() or temp_out.stat().st_size == 0):
        display_error(f"Output missing: {temp_out}")
        cleanup_temp_output(temp_out)
        return False
    orig_sz = src.stat().st_size
    new_sz = temp_out.stat().st_size if not config.dry_run else int(orig_sz * 0.7)  # Estimate for dry run
    red = (1 - new_sz / orig_sz) * 100 if orig_sz > 0 else 0
    display_info(f"{orig_sz/1e6:.2f}→{new_sz/1e6:.2f} MB ({red:.1f}% reduction)")
    return finalize_processed_output(src, temp_out, final_out, config, metadata, rename_original=True)

def process_video_amv(src, audio_track, config: Config, signature: dict):
    """Add or replace audio track in video (AMV operation)"""
    src = Path(src)
    audio_track = Path(audio_track)
    temp_out = make_temp_output_path(src, config.format)
    final_out = final_output_path(src, config.format, suffix="-amv")
    register_temp_output(temp_out)

    if not audio_track.exists():
        display_error(f"Audio track not found: {audio_track}")
        cleanup_temp_output(temp_out)
        return False

    details = {
        "audio_source": describe_file(audio_track),
        "encoding": {
            "video": "copy",
            "audio_codec": "aac",
            "audio_bitrate": "128k",
            "format": config.format,
        },
    }
    metadata = {"signature": signature, "details": details}

    # Build FFmpeg command for AMV operation
    cmd = ["ffmpeg"]
    if config.use_gpu:
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
        str(temp_out)
    ]
    cmd = insert_ffmpeg_metadata(cmd, metadata)

    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not config.dry_run:
        display_error("ffmpeg AMV failed")
        cleanup_temp_output(temp_out)
        return False
    if not config.dry_run and (not temp_out.is_file() or temp_out.stat().st_size == 0):
        display_error(f"Output missing: {temp_out}")
        cleanup_temp_output(temp_out)
        return False

    display_info(f"AMV created: {temp_out}")
    return finalize_processed_output(src, temp_out, final_out, config, metadata)

def process_video_loop_audio(src, config: Config, signature: dict):
    """Loop audio to match video duration"""
    src = Path(src)
    temp_out = make_temp_output_path(src, config.format)
    final_out = final_output_path(src, config.format, suffix="-loop")
    register_temp_output(temp_out)

    # Get video and audio duration info
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(src)]
    res = run_command(cmd)
    if res.returncode != 0:
        display_error(f"ffprobe error on {src}")
        cleanup_temp_output(temp_out)
        return False

    info = json.loads(res.stdout)
    video_stream = next((s for s in info.get("streams",[]) if s.get("codec_type")=="video"), {})
    audio_stream = next((s for s in info.get("streams",[]) if s.get("codec_type")=="audio"), {})

    if not video_stream or not audio_stream:
        display_error(f"Missing video or audio stream in {src}")
        cleanup_temp_output(temp_out)
        return False

    video_duration = float(video_stream.get("duration", 0) or info.get("format",{}).get("duration", 0))

    details = {
        "encoding": {
            "video": "copy",
            "audio_codec": "aac",
            "audio_bitrate": "128k",
            "duration": video_duration,
            "format": config.format,
        },
    }
    metadata = {"signature": signature, "details": details}

    # Build FFmpeg command to loop audio
    cmd = ["ffmpeg"]
    if config.use_gpu:
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
        str(temp_out)
    ]
    cmd = insert_ffmpeg_metadata(cmd, metadata)

    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not config.dry_run:
        display_error("ffmpeg loop_audio failed")
        cleanup_temp_output(temp_out)
        return False
    if not config.dry_run and (not temp_out.is_file() or temp_out.stat().st_size == 0):
        display_error(f"Output missing: {temp_out}")
        cleanup_temp_output(temp_out)
        return False

    display_info(f"Audio looped: {temp_out}")
    return finalize_processed_output(src, temp_out, final_out, config, metadata)

def process_video_audiofy(src, config: Config, signature: dict):
    """Extract audio from video and save as audio file"""
    src = Path(src)
    temp_out = make_temp_output_path(src, "mp3")
    final_out = final_output_path(src, "mp3", suffix="-audio")
    register_temp_output(temp_out)

    details = {
        "encoding": {
            "audio_codec": "libmp3lame",
            "audio_bitrate": "192k",
            "format": "mp3",
        },
    }
    metadata = {"signature": signature, "details": details}

    cmd = ["ffmpeg", "-i", str(src), "-vn", "-c:a", "libmp3lame", "-b:a", "192k", "-y", str(temp_out)]
    cmd = insert_ffmpeg_metadata(cmd, metadata)

    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not config.dry_run:
        display_error("ffmpeg audiofy failed")
        cleanup_temp_output(temp_out)
        return False
    if not config.dry_run and (not temp_out.is_file() or temp_out.stat().st_size == 0):
        display_error(f"Output missing: {temp_out}")
        cleanup_temp_output(temp_out)
        return False

    display_info(f"Audio extracted: {temp_out}")
    return finalize_processed_output(src, temp_out, final_out, config, metadata)

def process_image(src, config: Config, signature: dict):
    src = Path(src)
    temp_out = make_temp_output_path(src, "heic")
    final_out = final_output_path(src, "heic")
    register_temp_output(temp_out)
    details = {
        "encoding": {
            "format": "heic",
        },
    }
    metadata = {"signature": signature, "details": details}
    cmd = ["sips","-s","format","heic",str(src),"--out",str(temp_out)]
    logger.info("Running: " + " ".join(cmd))
    res = run_command(cmd)
    if res.returncode != 0 and not config.dry_run:
        display_error("sips failed")
        cleanup_temp_output(temp_out)
        return False
    if not config.dry_run and (not temp_out.is_file() or temp_out.stat().st_size == 0):
        display_error(f"Output missing: {temp_out}")
        cleanup_temp_output(temp_out)
        return False
    if not write_processing_metadata(temp_out, metadata):
        cleanup_temp_output(temp_out)
        return False
    orig_sz = src.stat().st_size
    new_sz = temp_out.stat().st_size if not config.dry_run else int(orig_sz * 0.5)  # Estimate for dry run
    red = (1-new_sz/orig_sz)*100 if orig_sz>0 else 0
    display_info(f"{orig_sz/1e3:.2f}→{new_sz/1e3:.2f} KB ({red:.1f}% reduction)")
    return finalize_processed_output(src, temp_out, final_out, config, metadata, rename_original=True)

def process_portrait(src, config: Config, signature: dict):
    """Generate Baldur's Gate portrait sizes as 24-bit BMPs."""
    src = Path(src)
    portraits_dir = src.parent / "Portraits"
    if config.dry_run:
        display_info(f"[DRY RUN] Would create folder: {portraits_dir}")
    else:
        try:
            portraits_dir.mkdir(exist_ok=True)
        except Exception as e:
            display_error(f"Failed to create Portraits folder: {e}")
            return False

    sizes = [
        ("L", 420, 660),
        ("M", 338, 532),
        ("S", 108, 168),
    ]

    success = True
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for suffix, width, height in sizes:
            temp_resample = tmpdir_path / f"{src.stem}_{suffix}_resample.png"
            temp_cropped = tmpdir_path / f"{src.stem}_{suffix}_crop.png"
            out = portraits_dir / f"{src.stem}{suffix}.bmp"

            cmd_resample = ["sips", "--resampleHeight", str(height), str(src), "--out", str(temp_resample)]
            logger.info("Running: " + " ".join(cmd_resample))
            res = run_command(cmd_resample)
            if res.returncode != 0 and not config.dry_run:
                display_error("sips resample failed")
                success = False
                break

            cmd_crop_png = [
                "sips",
                "--cropToHeightWidth", str(height), str(width),
                "-s", "format", "png",
                str(temp_resample),
                "--out", str(temp_cropped),
            ]
            logger.info("Running: " + " ".join(cmd_crop_png))
            res = run_command(cmd_crop_png)
            if res.returncode != 0 and not config.dry_run:
                display_error("sips crop failed")
                success = False
                break

            cmd_bmp = [
                "ffmpeg",
                "-y",
                "-i", str(temp_cropped),
                "-vf", "format=bgr24",
                str(out),
            ]
            logger.info("Running: " + " ".join(cmd_bmp))
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path and not config.dry_run:
                display_error("ffmpeg not found (required for 24-bit BMP output)")
                success = False
                break
            if ffmpeg_path:
                cmd_bmp[0] = ffmpeg_path
            res = run_command(cmd_bmp)
            if res.returncode != 0 and not config.dry_run:
                display_error("ffmpeg bmp conversion failed")
                success = False
                break
            if not config.dry_run and (not out.is_file() or out.stat().st_size == 0):
                display_error(f"Output missing: {out}")
                success = False
                break

            display_info(f"Portrait created: {out}")

    if success:
        # For portrait operation, keep the original image completely untouched
        # No metadata tracking, no renaming, no backup - just create the BMP outputs
        if config.dry_run:
            display_info(f"[DRY RUN] Portrait operation complete, original {src} untouched")
        else:
            display_info(f"Portrait operation complete, original {src} kept untouched")
        return True
    return False

def process_single_file(file_path, operation, config: Config, signature: dict):
    """Process a single file with the specified operation"""
    file_path = Path(file_path)

    # Convert to absolute path if not already
    if not file_path.is_absolute():
        file_path = Path(os.getcwd()) / file_path

    if not file_path.exists():
        display_error(f"File not found: {file_path}")
        return False

    success = True
    # Process based on file type and operations
    if is_video_file(file_path):
        display_info(f"Processing video: {file_path}")
        if operation == "refine":
            if not process_video_refine(file_path, config, signature):
                success = False
        elif operation == "amv":
            if not config.audio_file:
                display_error("AMV operation requires audio track (-a parameter)")
                success = False
            elif not process_video_amv(file_path, config.audio_file, config, signature):
                success = False
        elif operation == "loop_audio":
            if not process_video_loop_audio(file_path, config, signature):
                success = False
        elif operation == "audiofy":
            if not process_video_audiofy(file_path, config, signature):
                success = False
        elif operation == "portrait":
            display_info(f"Skipping portrait for video: {file_path}")
    elif is_image_file(file_path):
        display_info(f"Processing image: {file_path}")
        if operation == "refine":
            if not process_image(file_path, config, signature):
                success = False
        elif operation == "portrait":
            if not process_portrait(file_path, config, signature):
                success = False
        elif operation == "audiofy" and config.audio_file:
            if not process_image_audiofy(file_path, config.audio_file, config, signature):
                success = False
        else:
            if operation not in ("refine", "audiofy", "portrait"):
                display_info(f"Skipping image (no supported operations): {file_path}")
    else:
        display_error(f"Unsupported file type: {file_path}")
        success = False
    return success
def process_image_audiofy(src, audio_track, config: Config, signature: dict):
    """
    Given an image and an audio track, create a video of the image with the audio.
    If audio_track is a video, extract its audio stream first.
    """
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
            if res.returncode != 0 and not config.dry_run:
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
        temp_out = make_temp_output_path(src, config.format)
        final_out = final_output_path(src, config.format, suffix="-amv")
        register_temp_output(temp_out)
        details = {
            "audio_source": describe_file(audio_track),
            "encoding": {
                "video_codec": config.codec,
                "audio_codec": "aac",
                "audio_bitrate": "192k",
                "duration": duration,
                "format": config.format,
            },
        }
        metadata = {"signature": signature, "details": details}

        # Build ffmpeg command to create video from image + audio
        # Use -loop 1 to repeat image, -framerate 30, -t for duration, -shortest for audio, -r 30 before output
        cmd = [
            "ffmpeg",
            "-loop", "1",
            "-framerate", "30",
            "-i", str(src),
            "-i", str(audio_input),
            "-c:v", "libx264" if config.codec == "h264" else ("libx265" if config.codec == "hevc" else "libaom-av1"),
            "-t", str(duration),
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            "-r", "30",
            "-y",
            str(temp_out)
        ]
        cmd = insert_ffmpeg_metadata(cmd, metadata)
        logger.info("Running: " + " ".join(str(x) for x in cmd))
        res = run_command(cmd)
        if res.returncode != 0 and not config.dry_run:
            display_error("ffmpeg failed to create video from image and audio")
            cleanup_temp_output(temp_out)
            return False
        if not config.dry_run and (not temp_out.is_file() or temp_out.stat().st_size == 0):
            display_error(f"Output missing: {temp_out}")
            cleanup_temp_output(temp_out)
            return False
        display_info(f"Created video from image and audio: {temp_out}")
        return finalize_processed_output(src, temp_out, final_out, config, metadata)
    finally:
        # Clean up temp audio file if it was created
        if tmp_audio is not None and Path(tmp_audio).exists():
            try:
                Path(tmp_audio).unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp audio file: {tmp_audio} ({e})")

def expand_paths_recursively(paths):
    """Recursively expand all given paths into a flat list of media files"""
    expanded_inputs = []
    for path_str in paths:
        path_str = os.path.expanduser(path_str)
        if any(ch in path_str for ch in "*?[]"):
            matches = glob.glob(path_str)
            if matches:
                expanded_inputs.extend(matches)
            else:
                display_error(f"No matches for pattern: {path_str}")
        else:
            expanded_inputs.append(path_str)

    expanded_files = []

    for path_str in expanded_inputs:
        path = Path(path_str)

        # Convert to absolute path
        if not path.is_absolute():
            path = Path(os.getcwd()) / path

        if path.is_file():
            # Single file - add if it's a media file (include already-processed too)
            if path.name.startswith(ORIGINAL_PREFIX):
                continue
            if is_video_file(path) or is_image_file(path):
                expanded_files.append(path)
        elif path.is_dir():
            # Directory - recursively find all media files (include already-processed too)
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    if file_path.name.startswith(ORIGINAL_PREFIX):
                        continue
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
    """Get the base path for progress log files from the first path argument.
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

# Old progress tracking functions removed - now using ProgressTracker class

def refine_recursively(directory, codec, fmt):
    """Recursively process all files in directory (legacy function for compatibility)"""
    display_info(f"Recursively refining: {directory}")
    files = expand_paths_recursively([directory])
    total = len(files)
    display_info(f"Found {total} files")
    # Use global config if available, otherwise create default
    config = _GLOBAL_CONFIG or Config(
        preset=PRESETS["share"],
        codec=codec,
        format=fmt,
        use_gpu=False,
        trash=False,
        dry_run=False,
        interactive=False,
        audio_file=None,
        log=False,
    )
    signature = build_run_signature(config, "refine")

    # Set up progress tracking (legacy function - uses old format)
    base_path = get_base_path([directory])
    progress_file = base_path / ".automat-progress.log"

    # Initialize logs: start fresh
    if not config.dry_run:
        if progress_file.exists():
            try: progress_file.unlink()
            except Exception as e: display_error(f"Failed to remove existing progress log: {e}")
        try:
            progress_file.touch()
        except Exception as e:
            display_error(f"Failed to create log file: {e}")

    # Counters
    proc = fail = skipped = 0

    for path in files:
        n = proc + skipped + fail + 1
        print(f"{n}/{total}: {path}", flush=True)
        if is_already_processed(path, signature):
            print(f"{color_tag('[Skipped]', COLOR_YELLOW)} {path}", file=sys.stderr, flush=True)
            skipped += 1
            continue

        display_info(f"[{n}/{total}] Processing: {path}")
        if process_single_file(path, "refine", config, signature):
            proc += 1
        else:
            print(f"{color_tag('[Failed]', COLOR_RED)} {path}", file=sys.stderr, flush=True)
            fail += 1
            # Failed files are no longer logged to separate file

    display_info(f"Done. Processed: {proc}, Skipped: {skipped}, Failed: {fail}")
    if not config.dry_run:
        notify("Automat", f"Processed {proc} files, {fail} failed")
        # Auto-remove progress log if no failures
        if fail == 0 and progress_file.exists():
            try:
                progress_file.unlink()
            except Exception as e:
                display_error(f"Failed to remove log file {progress_file}: {e}")

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
        print("6. Create Baldur's Gate portraits from images")
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
        elif choice == "6":
            operations.append("portrait")

        else:
            print(f"Invalid choice: {choice}, defaulting to refine")
            operations.append("refine")

        # Common questions
        if operations:
            print()
            codec_choice = input(f"Output codec [h264]: ").strip() or "h264"
            if codec_choice in ["h264", "hevc", "av1"]:
                config.codec = codec_choice

            trash_choice = input("Move backup files (~originals) to trash? [y/N]: ").strip().lower()
            config.trash_backups = trash_choice == "y"

        print(f"\nReady to process {len(files)} file(s)")
        print(f"Operations: {', '.join(operations)}")
        print(f"Codec: {config.codec}")
        print(f"Preset: {config.preset.name}")
        print(f"Trash backups: {'Yes' if config.trash_backups else 'No'}")
        print()
        confirm = input("Proceed? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("Cancelled.")
            sys.exit(0)

        return (operations, config)

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

@dataclass
class FileProgress:
    """Track progress for a single file"""
    path: Path
    status: str  # "Pending", "Skipped", "Recoding", "Done", "Failed"
    time_taken: float = 0.0  # seconds
    original_size: int = 0
    new_size: int = 0
    original_params: str = ""
    new_params: str = ""

class ProgressTracker:
    """
    Manages progress tracking in a markdown table file.
    Filename format: automat-XXXXX-progress-total-done-failed.md
    """
    def __init__(self, base_path: Path, session_code: str):
        self.base_path = base_path
        self.session_code = session_code
        self.files: Dict[Path, FileProgress] = {}
        self.current_file: Optional[Path] = None

    def get_progress_filename(self) -> Path:
        """Generate filename based on current counters"""
        total = len(self.files)
        done = sum(1 for f in self.files.values() if f.status in ("Done", "Skipped"))
        failed = sum(1 for f in self.files.values() if f.status == "Failed")
        return self.base_path / f"automat-{self.session_code}-progress-{total}-{done}-{failed}.md"

    def initialize(self, file_paths: List[Path]):
        """Initialize with all files as Pending"""
        for path in file_paths:
            # Get file info for original params
            info = describe_file(path)
            original_params = self._format_file_params(path, info)

            self.files[path] = FileProgress(
                path=path,
                status="Pending",
                original_size=info.get("size") or 0,
                original_params=original_params,
            )

        self._write_table()

    def _format_file_params(self, path: Path, info: dict) -> str:
        """Format file parameters based on file type"""
        if is_video_file(path) or is_audio_file(path):
            # Video params: resolution, fps, codec, bitrate
            width = info.get("width", 0)
            height = info.get("height", 0)
            bitrate = info.get("bitrate", 0)
            duration = info.get("duration", 0)

            params = []
            if width and height:
                params.append(f"{width}×{height}")
            # FPS would need to be extracted from ffprobe - skip for now
            if bitrate:
                params.append(f"{int(bitrate/1000)}kbps")
            if duration:
                mins = int(duration / 60)
                params.append(f"{mins}m")

            return " ".join(params) if params else "-"

        elif is_image_file(path):
            # Image params: format, dimensions
            # Would need sips or similar to get dimensions - simplified for now
            return path.suffix[1:].upper()

        return "-"

    def mark_skipped(self, path: Path):
        """Mark file as skipped"""
        if path in self.files:
            old_path = self.get_progress_filename()  # Save before changing status
            self.files[path].status = "Skipped"
            self.files[path].time_taken = 0
            self._update_and_rename(old_path)

    def start_processing(self, path: Path):
        """Mark file as currently being processed"""
        if path in self.files:
            self.files[path].status = "Recoding"
            self.current_file = path
            self._write_table()  # Update without rename

    def mark_done(self, path: Path, time_taken: float, new_path: Path, new_params: str = ""):
        """Mark file as successfully completed"""
        if path in self.files:
            old_path = self.get_progress_filename()  # Save before changing status
            info = describe_file(new_path)
            self.files[path].status = "Done"
            self.files[path].time_taken = time_taken
            self.files[path].new_size = info.get("size") or 0
            self.files[path].new_params = new_params or self._format_file_params(new_path, info)
            self.current_file = None
            self._update_and_rename(old_path)

    def mark_failed(self, path: Path, time_taken: float):
        """Mark file as failed"""
        if path in self.files:
            old_path = self.get_progress_filename()  # Save before changing status
            self.files[path].status = "Failed"
            self.files[path].time_taken = time_taken
            self.current_file = None
            self._update_and_rename(old_path)

    def _update_and_rename(self, old_path: Path):
        """Update table and rename file if counters changed"""
        # Calculate new path based on updated counters
        new_path = self.get_progress_filename()

        # If filename changed, delete old file first
        if old_path != new_path and old_path.exists():
            try:
                old_path.unlink()
            except Exception as e:
                display_error(f"Failed to remove old progress file: {e}")

        # Write table to the (possibly new) path
        self._write_table()

    def _write_table(self):
        """Write current state to markdown table"""
        path = self.get_progress_filename()

        try:
            with open(path, 'w') as f:
                # Write header
                f.write("| Status | Time (s) | Filename | Original Size | New Size | Original Params | New Params |\n")
                f.write("|--------|----------|----------|---------------|----------|-----------------|------------|\n")

                # Write rows
                for file_progress in self.files.values():
                    status = file_progress.status
                    time_str = f"{file_progress.time_taken:.1f}" if file_progress.time_taken > 0 else "-"
                    filename = file_progress.path.name
                    orig_size = self._format_size(file_progress.original_size)
                    new_size = self._format_size(file_progress.new_size) if file_progress.new_size > 0 else "-"
                    orig_params = file_progress.original_params or "-"
                    new_params = file_progress.new_params or "-"

                    f.write(f"| {status} | {time_str} | {filename} | {orig_size} | {new_size} | {orig_params} | {new_params} |\n")

        except Exception as e:
            display_error(f"Failed to write progress file: {e}")

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        if size_bytes is None or size_bytes == 0:
            return "-"

        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


def main():
    """
    Main entry point for Automat.

    Progress logging:
      - Progress is tracked in a markdown table file (only for 'refine' operation).
      - Filename format: automat-XXXXX-progress-total-done-failed.md
      - Created at the start with all files as "Pending" status.
      - Updated on every file completion, and renamed when counters change.
      - Deleted automatically if no errors occurred.
    """
    global _GLOBAL_CONFIG
    p = argparse.ArgumentParser(
        description=(
            "Automat: refine videos and images using CPU or hardware-accelerated (GPU) encoding.\n\n"
            "This tool optimizes media files by re-encoding them with efficient codecs.\n"
            "Videos are processed with ffmpeg using CPU encoding by default (better compression, slower).\n"
            "Use --use-gpu to enable hardware-accelerated (videotoolbox) encoding for faster results (less compression).\n"
            "Images are converted to HEIC format using sips for better compression.\n\n"
            "Progress tracking (refine operation only): automat-XXXXX-progress-total-done-failed.md"
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
            "  # Check processing status:\n"
            "  automat.py --status video.mov\n"
            "  automat.py --status --preset meme video.mov  # Check if will reprocess with meme preset\n\n"
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
    p.add_argument("--portrait", action="store_true",
                   help="Generate Baldur's Gate portrait sizes from images")
    p.add_argument("--status", action="store_true",
                   help="Check processing status: show metadata and whether file will be reprocessed")

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
    p.add_argument("-s", "--suffix", default="-re",
                   help="Legacy suffix option (ignored) [default: -re]")

    # Behavior flags
    p.add_argument("--use-gpu", action="store_true",
                   help="Use hardware acceleration (videotoolbox) for faster encoding")
    p.add_argument("-t", "--trash-backups", action="store_true",
                   help="Move backup files (~originals) to trash after processing (refine only)")
    p.add_argument("--log", action="store_true",
                   help="Enable detailed logging to file")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Dry-run mode (show what would happen without processing)")

    # File arguments
    p.add_argument("files", nargs="+",
                   help="Files or directories to process")

    args = p.parse_args()

    # Validate argument dependencies
    if args.amv and not args.audio:
        p.error("--amv requires -a/--audio AUDIO_FILE")

    # Build configuration object
    preset = PRESETS[args.preset]
    config = Config(
        preset=preset,
        codec=args.codec,
        format=DEFAULT_FORMAT,  # Always use mp4 for videos
        use_gpu=args.use_gpu,
        trash_backups=args.trash_backups,
        dry_run=args.dry_run,
        interactive=args.interactive,
        audio_file=Path(args.audio) if args.audio else None,
        log=args.log,
    )

    # Set global config for functions that need it
    _GLOBAL_CONFIG = config

    install_signal_handlers()

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
        if args.portrait:
            operations.append("portrait")
        if args.status:
            operations.append("status")

        # Default to refine if no operations specified (unless status)
        if not operations:
            operations.append("refine")
        if len(operations) > 1:
            p.error("Only one operation can be specified at a time.")

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
        if len(operations) != 1:
            display_error("Only one operation can be specified at a time.")
            return 1
        # Update global config based on interactive config
        _GLOBAL_CONFIG = config
    elif args.auto and total_files == 1:
        # Auto-detect preset for single file
        media = MediaFile(Path(all_files[0]))
        recommended = media.recommend_preset()
        print(f"Auto-detected preset: {recommended} ({PRESETS[recommended].description})")
        config.preset = PRESETS[recommended]

    operation = operations[0]

    # Handle status operation separately (read-only)
    if operation == "status":
        # For status, we need to know what operation would be used
        # Use refine as default operation for comparison
        check_operation = "refine"
        for file_path in all_files:
            path = Path(file_path)
            status = check_file_status(path, config, check_operation)

            print(f"\n{'=' * 70}")
            print(f"File: {path}")
            print(f"{'=' * 70}")

            if status["processed"]:
                sig = status["metadata"]["signature"]
                print(f"✓ Processed: YES")
                print(f"  Operation:  {sig.get('operation', 'unknown')}")
                print(f"  Codec:      {sig.get('codec', 'unknown')}")
                print(f"  Format:     {sig.get('format', 'unknown')}")
                print(f"  Preset:     {sig.get('preset', 'unknown')}")
                print(f"  GPU:        {sig.get('use_gpu', False)}")

                # Show actual encoding parameters from signature
                if sig.get('crf') is not None:
                    print(f"  CRF:        {sig['crf']}")
                if sig.get('bitrate') is not None:
                    print(f"  Bitrate:    {sig['bitrate'] // 1000}k")
                if sig.get('resolution'):
                    w, h = sig['resolution']
                    print(f"  Resolution: {w}x{h}")
                if sig.get('audio_bitrate'):
                    print(f"  Audio BR:   {sig['audio_bitrate']}")
                if sig.get('audio_codec'):
                    print(f"  Audio Codec: {sig['audio_codec']}")

                # Show file size comparison
                if "details" in status["metadata"]:
                    details = status["metadata"]["details"]
                    if "input" in details:
                        input_info = details["input"]
                        orig_size = input_info.get("filesize", 0)
                        curr_size = path.stat().st_size
                        if orig_size > 0:
                            ratio = curr_size / orig_size
                            reduction = (1 - ratio) * 100
                            print(f"\n  File size:")
                            print(f"    Original:  {orig_size // 1_000_000} MB")
                            print(f"    Processed: {curr_size // 1_000_000} MB")
                            print(f"    Reduction: {reduction:.1f}%")

                    # Show encoding details if available
                    if "encoding" in details:
                        enc = details["encoding"]
                        print(f"\n  Encoding details:")
                        for k, v in enc.items():
                            print(f"    {k}: {v}")

                print(f"\nWith current settings ({config.codec}, {config.preset.name}, GPU={config.use_gpu}):")
                if status["will_reprocess"]:
                    print(f"⚠ WILL REPROCESS")
                    print(f"  {status['reason']}")
                else:
                    print(f"✓ WILL SKIP (already processed with identical settings)")
            else:
                print(f"✗ Processed: NO")
                print(f"  {status['reason']}")

        return 0

    signature = build_run_signature(config, operation)

    if not config.interactive:
        display_info(f"Found {total_files} media files to process")

    # Generate session code for this run
    global _SESSION_CODE
    _SESSION_CODE = secrets.token_hex(3)  # 6-character hex code

    # Set up logging
    base_path = get_base_path(args.files)

    # Setup raw logging only if --log is enabled
    if config.log:
        raw_log_file = base_path / f"automat-{_SESSION_CODE}-raw.log"
        setup_logging(str(raw_log_file), debug=True)
    else:
        setup_logging(None, debug=False)

    # Initialize progress tracker (only for refine operation)
    tracker = None
    if operation == "refine":
        tracker = ProgressTracker(base_path, _SESSION_CODE)
        tracker.initialize([Path(f) for f in all_files])

    processed = 0
    failed = 0
    skipped = 0

    process_start = time.time()

    # Process each file
    for file_path in all_files:
        file_start = time.time()
        n = processed + skipped + failed + 1
        print(f"{n}/{total_files}: {file_path}", flush=True)

        path = Path(file_path)

        if is_already_processed(file_path, signature):
            print(f"{color_tag('[Skipped]', COLOR_YELLOW)} {file_path}", file=sys.stderr, flush=True)
            skipped += 1
            if tracker:
                tracker.mark_skipped(path)
            continue

        if tracker:
            tracker.start_processing(path)

        ok = process_single_file(file_path, operation, config, signature)
        elapsed = time.time() - file_start

        if ok:
            processed += 1
            if tracker:
                # Get new file path (might have different extension)
                new_path = final_output_path(path, config.format)
                tracker.mark_done(path, elapsed, new_path)
        else:
            print(f"{color_tag('[Failed]', COLOR_RED)} {file_path}", file=sys.stderr, flush=True)
            failed += 1
            if tracker:
                tracker.mark_failed(path, elapsed)

    # Print summary to raw log (not stdout)
    display_info(f"Processing complete. Success: {processed}, Skipped: {skipped}, Failed: {failed}")

    # Show notification when complete
    if not config.dry_run and total_files > 1:
        notify("Automat", f"Processed {processed} files, {failed} failed")

    # Progress and log files are kept for user review - no automatic cleanup

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
