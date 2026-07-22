# Utility Scripts

This directory contains helper utilities for common repetitive tasks. These scripts are designed with the following principles:

- **Portability**: Maximum compatibility across Linux and macOS with minimal dependencies
- **No external packages**: Scripts avoid external library dependencies when possible
- **Low maintenance**: Designed to work reliably over time without frequent updates
- **Self-contained**: Each utility is standalone and can be used independently

## Utilities

- **automat.py** - Media processing tool for optimizing videos and images with ffmpeg/sips
- **png2icns.py** - Converts PNG images to macOS .icns icon format
- **bigsync** - Robust, resumable directory sync over SSH with smart defaults
- **bookcompile** - Compiles selected Markdown files into a full Markdown manuscript, EPUB, and Word DOCX
- **stickerfy** - Turns selected images into Telegram-ready stickers using local Apple Vision subject extraction

## Requirements

- Python 3 (for Python scripts)
- zsh (for shell scripts)
- Standard Unix tools (available on both Linux and macOS)

Individual utilities may have specific requirements documented in their help output.

### bookcompile

Select one or more Markdown files in Finder and pass them as arguments. Numbered
chapters are sorted numerically; other filenames are sorted alphabetically:

```sh
bookcompile --author "Author Name" "/path/to/Book/01 - Start.md" "/path/to/Book/02 - End.md"
```

Passing a folder instead selects all Markdown files directly inside it. Either
way, it writes `Book Name.md`, `Book Name.epub`, and `Book Name.docx` under the
selected files' `Exports/` folder. Subfolders are ignored. When only one file is
selected, its filename becomes the output name. Use `--format md`, `--format epub`, or `--format docx`
to request only selected formats; repeat the option to request more than one.

For a Finder Quick Action in Automator:

1. Choose “Workflow receives current: files or folders” in Finder.
2. Add “Run Shell Script”, choose `/bin/zsh`, and pass input “as arguments”.
3. Use:

```sh
source "$HOME/.zprofile"
bookcompile --format epub "$@"
```

Add `--author "Author Name"` before `"$@"` in that last line when you want the
name embedded in EPUB and Word metadata.

The command is non-interactive and shows a macOS notification when conversion
finishes. All selected chapters must belong to the same folder.

EPUB and DOCX conversion uses Pandoc, installed by `~/.myhome/manage.sh --init`.
The combined Markdown export needs only Python 3.

### stickerfy

Select one or more images in Finder and pass them as arguments:

```sh
stickerfy photo.jpg drawing.png
```

Transparent images keep their alpha channel. Opaque images use Apple's on-device
subject extraction, then receive a white border and soft black shadow. The command
writes 512 × 512 transparent PNG files to a `Stickers` folder beside each input.
If a detailed PNG would exceed Telegram's 512 KB limit, it automatically falls
back to WebP using cwebp. It requires macOS 14 or newer for automatic background
removal, and processing never sends an image over the network.

For a Finder Quick Action in Automator:

1. Choose “Workflow receives current: image files” in Finder.
2. Add “Run Shell Script”, choose `/bin/zsh`, and pass input “as arguments”.
3. Use:

```sh
source "$HOME/.zprofile"
stickerfy "$@"
```

The command is non-interactive and shows a macOS notification when processing
finishes. Run `stickerfy --help` for border, shadow, margin, and output options.
