#!/bin/zsh
set -euo pipefail
setopt extended_glob

# Determine the directory of this script so it can be run from anywhere
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
KBD_LAYOUT_NAME="ERDB.bundle"

usage() {
    cat << EOF
Usage: $0 [--init|--kbd_install|--kbd_uninstall|--kbd_open_plist|--help]
    --init             Initialize Homebrew and install core packages
    --kbd_install      Install keyboard layout "$KBD_LAYOUT_NAME"
    --kbd_uninstall    Uninstall keyboard layout "$KBD_LAYOUT_NAME"
    --kbd_open_plist   Open HIToolbox preferences plist for editing
    --help             Display this usage message
EOF
}

init() {
    echo "INFO: Starting initialization..."
    touch .hushlogin

    echo "INFO: Zprofile"

    if [ -e "$HOME/.zprofile" ]; then
        echo "WARN: Zprofile already exists. Skipped."
    else
        ln "$BASEDIR/cfg/zprofile" "$HOME/.zprofile"
        echo "INFO: Zprofile hardlinked into your home directory."
    fi

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
    if [ ! -d "$BASEDIR/kbd/$layout_name" ]; then
        echo "Layout '$layout_name' not found in $BASEDIR. Skipping."
        exit 1
    fi

    echo "INFO: Installing $layout_name from $BASEDIR/kbd to /Library/Keyboard Layouts/..."
    sudo cp -r "$BASEDIR/kbd/$layout_name" "/Library/Keyboard Layouts/" && \
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
        exit 1
    fi

    plutil -convert xml1 "$plist" && \
        ${EDITOR:-nano} "$plist" || \
        echo "ERROR: Failed to open plist for editing."
}

# Function dispatch and usage handling
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case "$1" in
    --init)
        init
        ;;
    --kbd_install)
        kbd_install "$KBD_LAYOUT_NAME"
        ;;
    --kbd_uninstall)
        kbd_uninstall "$KBD_LAYOUT_NAME"
        ;;
    --kbd_open_plist)
        kbd_open_plist
        ;;
    --help)
        usage
        ;;
    *)
        usage

        # Check for Homebrew
        if ! command -v brew >/dev/null 2>&1; then
            # Prompt for fresh install initialization
            read -q "?INFO: Homebrew not detected. Fresh install? Initialize homedir? (y/N) " brew_resp
            echo
            if [[ $brew_resp == [Yy] ]]; then
                init
            fi
        fi

        # Check for keyboard layout installation
        if [ ! -d "/Library/Keyboard Layouts/$KBD_LAYOUT_NAME" ]; then
            read -q "?INFO: Keyboard layout $KBD_LAYOUT_NAME not installed. Install? (y/N) " kbd_resp
            echo
            if [[ $kbd_resp == [Yy] ]]; then
                kbd_install "$KBD_LAYOUT_NAME"
            fi
        fi
        ;;
esac
