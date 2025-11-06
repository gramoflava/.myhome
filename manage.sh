#!/bin/zsh
setopt extended_glob

# Determine the directory of this script so it can be run from anywhere
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
KBD_LAYOUT_NAME="ERDB.bundle"

usage() {
    cat << EOF
Usage: $0 [--init|--help]
    --init    Initialize dotfiles, install Homebrew and core packages
    --help    Display this usage message

Without arguments, runs interactive menu for configuration management.
EOF
}

# Check configuration status and detect issues
check_status() {
    local issues=()
    local status_ok=()

    echo "=== Configuration Status ==="
    echo

    # Check .zprofile
    if [ -L "$HOME/.zprofile" ]; then
        issues+=("~/.zprofile is a symlink (should use 'source' instead)")
    elif [ -f "$HOME/.zprofile" ] && grep -qF "source \"$BASEDIR/cfg/zprofile\"" "$HOME/.zprofile" 2>/dev/null; then
        status_ok+=("✓ ~/.zprofile sources config correctly")
    elif [ -f "$HOME/.zprofile" ]; then
        issues+=("~/.zprofile exists but doesn't source config")
    else
        issues+=("~/.zprofile not configured")
    fi

    # Check .zshrc
    if [ -L "$HOME/.zshrc" ]; then
        issues+=("~/.zshrc is a symlink (should use 'source' instead)")
    elif [ -f "$HOME/.zshrc" ] && grep -qF "source \"$BASEDIR/cfg/zshrc\"" "$HOME/.zshrc" 2>/dev/null; then
        status_ok+=("✓ ~/.zshrc sources config correctly")
    elif [ -f "$HOME/.zshrc" ]; then
        issues+=("~/.zshrc exists but doesn't source config")
    else
        issues+=("~/.zshrc not configured")
    fi

    # Check .vimrc (symlink is OK)
    if [ -L "$HOME/.vimrc" ] && [ "$(readlink "$HOME/.vimrc")" = "$BASEDIR/cfg/vimrc" ]; then
        status_ok+=("✓ ~/.vimrc symlinked correctly")
    elif [ -f "$HOME/.vimrc" ]; then
        issues+=("~/.vimrc exists but not linked to config")
    else
        issues+=("~/.vimrc not configured")
    fi

    # Check .tmux.conf (symlink is OK)
    if [ -L "$HOME/.tmux.conf" ] && [ "$(readlink "$HOME/.tmux.conf")" = "$BASEDIR/cfg/tmux.conf" ]; then
        status_ok+=("✓ ~/.tmux.conf symlinked correctly")
    elif [ -f "$HOME/.tmux.conf" ]; then
        issues+=("~/.tmux.conf exists but not linked to config")
    else
        issues+=("~/.tmux.conf not configured")
    fi

    # Check keyboard layout (macOS only)
    if [[ "$OSTYPE" == darwin* ]]; then
        if [ -d "/Library/Keyboard Layouts/$KBD_LAYOUT_NAME" ]; then
            status_ok+=("✓ Keyboard layout $KBD_LAYOUT_NAME installed")
        else
            issues+=("Keyboard layout $KBD_LAYOUT_NAME not installed")
        fi
    fi

    # Check Homebrew
    if command -v brew >/dev/null 2>&1; then
        status_ok+=("✓ Homebrew installed")
    else
        issues+=("Homebrew not installed")
    fi

    # Print OK items
    for item in "${status_ok[@]}"; do
        echo "$item"
    done

    # Print issues
    if [ ${#issues[@]} -gt 0 ]; then
        echo
        echo "Issues found:"
        for issue in "${issues[@]}"; do
            echo "  ⚠ $issue"
        done
        return 1
    fi

    echo
    echo "All configurations OK!"
    return 0
}

# Fix symlinks by converting to source
fix_symlinks() {
    local fixed=0

    echo "=== Fixing Symlinks ==="
    echo

    # Fix .zprofile symlink
    if [ -L "$HOME/.zprofile" ]; then
        echo "Converting ~/.zprofile from symlink to source..."
        rm "$HOME/.zprofile"
        echo "source \"$BASEDIR/cfg/zprofile\"" > "$HOME/.zprofile"
        ((fixed++))
    fi

    # Fix .zshrc symlink
    if [ -L "$HOME/.zshrc" ]; then
        echo "Converting ~/.zshrc from symlink to source..."
        rm "$HOME/.zshrc"
        echo "source \"$BASEDIR/cfg/zshrc\"" > "$HOME/.zshrc"
        ((fixed++))
    fi

    if [ $fixed -eq 0 ]; then
        echo "No symlinks to fix."
    else
        echo "Fixed $fixed symlink(s)."
    fi
    echo
}

init() {
    echo "INFO: Starting initialization..."

    # Prevent login messages ("Last login: ...")
    touch $HOME/.hushlogin

    # Install Zprofile and Zshrc if they don't exist
    echo "INFO: Zprofile"
    if [ -L "$HOME/.zprofile" ]; then
        echo "WARN: ~/.zprofile is a symlink, converting to source..."
        rm "$HOME/.zprofile"
        echo "source \"$BASEDIR/cfg/zprofile\"" > "$HOME/.zprofile"
    elif [ ! -f "$HOME/.zprofile" ]; then
        echo "INFO: Creating new $HOME/.zprofile"
        echo "source \"$BASEDIR/cfg/zprofile\"" > "$HOME/.zprofile"
    elif [ ! -s "$HOME/.zprofile" ]; then
        echo "INFO: Creating new $HOME/.zprofile (was empty)"
        echo "source \"$BASEDIR/cfg/zprofile\"" > "$HOME/.zprofile"
    else
        echo "INFO: Adding source line to $HOME/.zprofile"
        grep -qxF "source \"$BASEDIR/cfg/zprofile\"" "$HOME/.zprofile" 2>/dev/null \
            && echo "INFO: Zprofile already sourced" \
            || printf '1i\nsource "%s/cfg/zprofile"\n.\nwq\n' "$BASEDIR" | ed -s "$HOME/.zprofile"
    fi

    echo "INFO: Zshrc"
    if [ -L "$HOME/.zshrc" ]; then
        echo "WARN: ~/.zshrc is a symlink, converting to source..."
        rm "$HOME/.zshrc"
        echo "source \"$BASEDIR/cfg/zshrc\"" > "$HOME/.zshrc"
    elif [ ! -f "$HOME/.zshrc" ]; then
        echo "INFO: Creating new $HOME/.zshrc"
        echo "source \"$BASEDIR/cfg/zshrc\"" > "$HOME/.zshrc"
    elif [ ! -s "$HOME/.zshrc" ]; then
        echo "INFO: Creating new $HOME/.zshrc (was empty)"
        echo "source \"$BASEDIR/cfg/zshrc\"" > "$HOME/.zshrc"
    else
        grep -qxF "source \"$BASEDIR/cfg/zshrc\"" "$HOME/.zshrc" 2>/dev/null \
            && echo "INFO: Zshrc already sourced" \
            || printf '1i\nsource "%s/cfg/zshrc"\n.\nwq\n' "$BASEDIR" | ed -s "$HOME/.zshrc"
    fi

    echo "INFO: Vimrc"
    ln -sf "$BASEDIR/cfg/vimrc" "$HOME/.vimrc" 2>/dev/null || \
        echo "WARN: Failed to link $HOME/.vimrc, it may already exist."

    echo "INFO: Tmux.conf"
    ln -sf "$BASEDIR/cfg/tmux.conf" "$HOME/.tmux.conf" 2>/dev/null || \
        echo "WARN: Failed to link $HOME/.tmux.conf, it may already exist."

    # Core utilities
    echo "INFO: Core utilities"
    typeset -A brew_packages
    brew_packages=(
        bat       "Better cat with syntax highlighting"
        fd        "Modern find alternative"
        ripgrep   "Fast grep alternative"
        tree      "Directory structure viewer"
        jq        "JSON processor"
        tig       "Text-mode interface for git"
        zsh-completions "Additional completions"
        ffmpeg    "Media processor"
    )

    # Init or install Homebrew if missing
    if ! command -v brew >/dev/null 2>&1; then
        echo "WARN: Homebrew not detected. Looking..."
        if [ -x /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -x /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        else
            echo "WARN: Homebrew not found. Installing..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

            if [ -x /opt/homebrew/bin/brew ]; then
                eval "$(/opt/homebrew/bin/brew shellenv)"
            elif [ -x /usr/local/bin/brew ]; then
                eval "$(/usr/local/bin/brew shellenv)"
            fi
        fi

        echo "INFO: Proceeding with installation..."
    fi

    # Install required Brew packages
    brew install ${(k)brew_packages}
    echo "INFO: Core packages installation completed."
}

kbd_install() {
    local layout_name="$1"
    if [ ! -d "$BASEDIR/cfg/kbd/$layout_name" ]; then
        echo "Layout '$layout_name' not found in $BASEDIR. Skipping."
        return 1
    fi

    echo "INFO: Installing $layout_name from $BASEDIR/cfg/kbd to /Library/Keyboard Layouts/..."
    sudo cp -r "$BASEDIR/cfg/kbd/$layout_name" "/Library/Keyboard Layouts/" && \
        echo "INFO: Installation completed successfully." || \
        echo "ERROR: Installation failed."
}

kbd_uninstall() {
    local layout_name="$1"
    echo "INFO: Removing $layout_name from /Library/Keyboard Layouts/"

    sudo rm -rf "/Library/Keyboard Layouts/$layout_name" && \
        echo "INFO: Removal completed successfully." || \
        echo "ERROR: Removal failed."
}

kbd_open_plist() {
    local plist="$HOME/Library/Preferences/com.apple.HIToolbox.plist"

    echo "INFO: Opening HIToolbox preferences plist for editing"
    echo "INFO: Plist path: $plist"
    if [ ! -f "$plist" ]; then
        echo "Plist file not found."
        return 1
    fi

    plutil -convert xml1 "$plist" && \
        ${EDITOR:-vim} "$plist" || \
        echo "ERROR: Failed to open plist for editing."
}

# Interactive menu
interactive_menu() {
    while true; do
        echo
        echo "=== .myhome Configuration Manager ==="
        echo
        echo "1) Check configuration status"
        echo "2) Fix symlinks (convert to source)"
        echo "3) Install keyboard layout (macOS)"
        echo "4) Uninstall keyboard layout (macOS)"
        echo "5) Edit HIToolbox plist (macOS)"
        echo "6) Reinstall/update dotfiles"
        echo "q) Quit"
        echo
        read "choice?Choose an option: "

        case "$choice" in
            1)
                echo
                check_status
                ;;
            2)
                echo
                fix_symlinks
                ;;
            3)
                if [[ "$OSTYPE" != darwin* ]]; then
                    echo "ERROR: Keyboard layout installation only available on macOS"
                else
                    echo
                    kbd_install "$KBD_LAYOUT_NAME"
                fi
                ;;
            4)
                if [[ "$OSTYPE" != darwin* ]]; then
                    echo "ERROR: Keyboard layout management only available on macOS"
                else
                    echo
                    kbd_uninstall "$KBD_LAYOUT_NAME"
                fi
                ;;
            5)
                if [[ "$OSTYPE" != darwin* ]]; then
                    echo "ERROR: HIToolbox plist only available on macOS"
                else
                    echo
                    kbd_open_plist
                fi
                ;;
            6)
                echo
                init
                ;;
            q|Q)
                echo "Exiting."
                exit 0
                ;;
            *)
                echo "Invalid option. Please try again."
                ;;
        esac
    done
}

# Main entry point
if [ $# -eq 0 ]; then
    # No arguments - run interactive menu
    interactive_menu
fi

case "$1" in
    --init)
        init
        ;;
    --help)
        usage
        ;;
    *)
        echo "ERROR: Unknown option '$1'"
        usage
        exit 1
        ;;
esac
