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
