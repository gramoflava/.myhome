#!/usr/bin/env python3
"""
png2icns: Convert a PNG (or other raster) to a macOS .icns file.

Updates:
- Uses modern Pillow API (Image.Resampling.LANCZOS).
- Graceful checks for missing dependencies and tools.
- Warns about too-small sources and suggests installs.
- Keeps behavior simple and deterministic.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# --- Dependency guards -------------------------------------------------------
MISSING: list[str] = []
try:
    from PIL import Image  # pillow provides the PIL namespace
except Exception:  # ImportError or other import-time errors
    Image = None  # type: ignore
    MISSING.append("pillow")

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("png2icns")

# --- Helpers -----------------------------------------------------------------

def require_deps() -> None:
    if MISSING:
        pkg_list = ", ".join(MISSING)
        log.error("Missing Python packages: %s", pkg_list)
        log.error("Install with: pip3 install %s", pkg_list)
        sys.exit(1)


def require_tool(cmd: str) -> None:
    if shutil.which(cmd) is None:
        log.error("Required tool '%s' not found in PATH.", cmd)
        if cmd == "iconutil":
            log.error("'iconutil' is available on macOS. If you are on macOS, run: xcode-select --install")
        sys.exit(1)


def resample_filter():
    """Return a Pillow LANCZOS-compatible resampling filter across versions."""
    # Pillow >= 9.1: Image.Resampling.LANCZOS. Older: Image.LANCZOS.
    if Image is None:  # pragma: no cover (guarded by require_deps)
        return None
    resampling = getattr(Image, "Resampling", None)
    return getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))


def load_image(path: Path) -> Image.Image:
    try:
        img = Image.open(path).convert("RGBA")
    except Exception as e:
        log.error("Unable to open '%s': %s", path, e)
        sys.exit(1)
    return img


def to_square(img: "Image.Image") -> "Image.Image":
    """Pad to a square with transparent background without distortion."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


# Apple iconset sizes (base size, scale)
ICON_SPECS = [
    (16, 1), (16, 2),
    (32, 1), (32, 2),
    (128, 1), (128, 2),
    (256, 1), (256, 2),
    (512, 1), (512, 2),  # 512@2x = 1024px
]


def build_iconset(img: "Image.Image", iconset_dir: Path) -> None:
    iconset_dir.mkdir(exist_ok=True)
    filt = resample_filter()
    max_needed = max(base * scale for base, scale in ICON_SPECS)

    # Warn if the source is smaller than the largest required size
    if max(img.size) < max_needed:
        log.warning("Source image is smaller than %d px. Result will be upscaled.", max_needed)

    for base, scale in ICON_SPECS:
        pixels = base * scale
        suffix = "@2x" if scale == 2 else ""
        target_name = iconset_dir / f"icon_{base}x{base}{suffix}.png"

        # Resize with high-quality filter
        resized = img.resize((pixels, pixels), filt)
        try:
            resized.save(target_name)
            log.info("Generated %s", target_name)
        except Exception as e:
            log.error("Failed to save %s: %s", target_name, e)
            sys.exit(1)


def run_iconutil(iconset_dir: Path, icns_file: Path) -> None:
    result = subprocess.run(
        ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        log.error("iconutil failed (%d): %s", result.returncode, stderr or "no error output")
        sys.exit(result.returncode)
    log.info("ICNS file generated: %s", icns_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a PNG image into an .icns icon")
    p.add_argument("input_image", help="Path to the input PNG (or any raster) file")
    p.add_argument("-o", "--output", help="Output .icns path (defaults next to input)")
    return p.parse_args()


def main() -> None:
    # Dependencies and tools
    require_deps()
    require_tool("iconutil")

    args = parse_args()
    input_path = Path(args.input_image)
    if not input_path.is_file():
        log.error("File '%s' not found.", input_path)
        sys.exit(1)

    # Prepare paths
    input_dir = input_path.parent
    base_name = input_path.stem
    iconset_dir = input_dir / f"{base_name}.iconset"
    icns_file = Path(args.output) if args.output else input_dir / f"{base_name}.icns"

    # Load and normalize image
    img = load_image(input_path)
    img = to_square(img)

    # Build iconset and convert
    try:
        build_iconset(img, iconset_dir)
        run_iconutil(iconset_dir, icns_file)
    finally:
        # Clean up regardless of success/failure of iconutil
        try:
            shutil.rmtree(iconset_dir)
            log.info("Cleaned up %s", iconset_dir)
        except Exception as e:
            log.warning("Failed to remove %s: %s", iconset_dir, e)


if __name__ == "__main__":
    main()