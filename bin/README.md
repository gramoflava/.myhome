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
- **bookcompile** - Compiles numbered Markdown chapters into a full Markdown manuscript, EPUB, and Word DOCX

## Requirements

- Python 3 (for Python scripts)
- zsh (for shell scripts)
- Standard Unix tools (available on both Linux and macOS)

Individual utilities may have specific requirements documented in their help output.

### bookcompile

Select the chapter files in Finder and pass them as arguments (their incoming
order does not matter; the utility sorts them by chapter number):

```sh
bookcompile --author "Author Name" "/path/to/Book/01 - Start.md" "/path/to/Book/02 - End.md"
```

Passing the book folder instead selects all of its numbered chapters. Either
way, it writes `Book Name.md`, `Book Name.epub`, and `Book Name.docx` under the
chapters' `Exports/` folder. Files such as `Notebook.md` and anything in
subfolders are ignored. Use `--format md`, `--format epub`, or `--format docx`
to request only selected formats; repeat the option to request more than one.

For a Finder Quick Action in Automator:

1. Choose “Workflow receives current: files or folders” in Finder.
2. Add “Run Shell Script”, choose `/bin/zsh`, and pass input “as arguments”.
3. Use:

```sh
source "$HOME/.zprofile"
bookcompile "$@"
```

Add `--author "Author Name"` before `"$@"` in that last line when you want the
name embedded in EPUB and Word metadata.

The command is non-interactive and shows a macOS notification when conversion
finishes. All selected chapters must belong to the same folder.

EPUB and DOCX conversion uses Pandoc, installed by `~/.myhome/init`.
The combined Markdown export needs only Python 3.
