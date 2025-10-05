# .myhome

Personal dotfiles, scripts, and preferences for macOS and Linux — easy setup and reliable cross-platform configuration management.

## Features

- **Cross-platform compatibility**: Works on macOS and Linux
- **Source-based config**: Uses `source` instead of symlinks for shell configs to prevent external tools from modifying your repository
- **Interactive management**: Menu-driven interface for configuration maintenance
- **Automatic detection**: Identifies and fixes configuration issues
- **Keyboard layouts**: Easy installation of custom keyboard layouts (macOS)

## Quickstart

Install Homebrew, clone repository, and launch auto-configuration all in one copy-paste:

- **Read-only**

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    git clone https://github.com/gramoflava/.myhome.git ~/.myhome
    chmod +x ~/.myhome/manage.sh && ~/.myhome/manage.sh --init
    ```

- **Write access**

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:gramoflava/.myhome.git ~/.myhome
    chmod +x ~/.myhome/manage.sh && ~/.myhome/manage.sh --init
    ```

## Configuration Management

### Interactive Menu

Run without arguments to access the interactive configuration manager:

```bash
~/.myhome/manage.sh
```

**Menu options:**
1. **Check configuration status** - Verify all dotfiles are configured correctly
2. **Fix symlinks** - Convert old symlinks to source-based configuration
3. **Install keyboard layout** - Install custom keyboard layout (macOS only)
4. **Uninstall keyboard layout** - Remove custom keyboard layout (macOS only)
5. **Edit HIToolbox plist** - Modify keyboard layout preferences (macOS only)
6. **Reinstall/update dotfiles** - Re-run full initialization

### Command Line

```bash
~/.myhome/manage.sh --init    # Initial setup (non-interactive)
~/.myhome/manage.sh --help    # Show usage information
```

## What Gets Configured

### Shell Configuration
- `~/.zprofile` - Login shell configuration (sources `cfg/zprofile`)
- `~/.zshrc` - Interactive shell configuration (sources `cfg/zshrc`)
- `~/.vimrc` - Vim configuration (symlinked to `cfg/vimrc`)

**Note:** Shell configs use `source` instead of symlinks to prevent external tools from modifying the repository when they append their own configurations.

### Installed Packages (via Homebrew)
- `bat` - Better cat with syntax highlighting
- `fd` - Modern find alternative
- `ripgrep` - Fast grep alternative
- `tree` - Directory structure viewer
- `jq` - JSON processor
- `tig` - Text-mode interface for git
- `zsh-completions` - Additional shell completions
- `ffmpeg` - Media processor

### Custom Features
- **rsyncx** - Robust, resumable directory sync over SSH with smart defaults
- **Path expansion** - Type `/o/a/c<Tab>` to expand to `/opt/ai-stack/compose`
- **Smart history** - 5000 entries, ignores common commands, shares across sessions
- **Custom keyboard layout** - ERDB layout for macOS (optional)

## Migration from Symlinks

If you previously used symlinks for `.zshrc` or `.zprofile`, the configuration manager will detect and offer to fix them:

```bash
~/.myhome/manage.sh  # Choose option 1 to check, then option 2 to fix
```

This prevents external tools (like `oh-my-zsh`, language installers, etc.) from modifying your repository when they append configurations.

## Useful Commands

### (Re)generate a new SSH key and copy it to the clipboard for adding to GitHub

> **Note:** This command will overwrite existing `~/.ssh/id_ed25519*` keys. Back up any existing keys before proceeding or adjust the filenames to avoid data loss.

```bash
rm -f ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub && ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q && cat ~/.ssh/id_ed25519.pub | pbcopy
```

## Structure

```
.myhome/
├── cfg/
│   ├── zprofile       # Login shell configuration
│   ├── zshrc          # Interactive shell configuration
│   ├── vimrc          # Vim configuration
│   └── kbd/           # Custom keyboard layouts
├── manage.sh          # Configuration manager
└── README.md          # This file
```

## Platform-Specific Notes

### macOS
- Supports custom keyboard layout installation
- Uses Homebrew from `/opt/homebrew` (Apple Silicon) or `/usr/local` (Intel)
- Includes `ls -G` for colored output

### Linux
- Uses system package manager alongside Homebrew
- Includes `ls --color=auto` for colored output
- Keyboard layout features are disabled (macOS only)

## Troubleshooting

**Q: My shell config isn't loading**
A: Run `~/.myhome/manage.sh` and choose option 1 to check status, then option 6 to reinstall if needed.

**Q: External tool modified my repository**
A: This shouldn't happen with source-based configs. Run option 1 to verify configs are correct.

**Q: Completions are slow**
A: The config uses optimized completers. If still slow, check for large completion caches: `rm ~/.zcompdump*`

**Q: How do I update to latest configs?**
A: `cd ~/.myhome && git pull && ./manage.sh` (option 6 to reinstall)
