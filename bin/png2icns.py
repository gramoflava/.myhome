#!/usr/bin/env python3
import argparse
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Convert a PNG image into an .icns icon")
    parser.add_argument("input_image", help="Path to the input PNG file")
    args = parser.parse_args()

    input_path = Path(args.input_image)
    if not input_path.is_file():
        logging.error("File '%s' not found.", input_path)
        return

    try:
        img = Image.open(input_path)
        width, height = img.size
    except Exception as e:
        logging.error("Unable to open '%s': %s", input_path, e)
        return
    max_dim = max(width, height)

    input_dir = input_path.parent
    base_name = input_path.stem
    iconset_dir = input_dir / f"{base_name}.iconset"
    icns_file = input_dir / f"{base_name}.icns"

    iconset_dir.mkdir(exist_ok=True)

    sizes = [16, 32, 64, 128, 256, 512]
    for size in sizes:
        for scale in (1, 2):
            pixels = size * scale
            if pixels <= max_dim:
                suffix = "@2x" if scale == 2 else ""
                output = iconset_dir / f"icon_{size}x{size}{suffix}.png"
                img.resize((pixels, pixels), Image.LANCZOS).save(output)
                logging.info("Generated %s", output)

    result = subprocess.run(
        ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_file)],
        capture_output=True,
    )
    if result.returncode != 0:
        logging.error("iconutil failed (%d): %s", result.returncode, result.stderr.decode().strip())
        return
    logging.info("ICNS file generated: %s", icns_file)

    try:
        shutil.rmtree(iconset_dir)
        logging.info("Cleaned up %s", iconset_dir)
    except Exception as e:
        logging.warning("Failed to remove %s: %s", iconset_dir, e)

    return

if __name__ == "__main__":
    main()