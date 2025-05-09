# .myhome

Personal macOS setup — my dotfiles, scripts, and preferences for easy initialization.

## Quickstart

    /bin/bash -c \
      "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone https://github.com/gramoflava/.myhome.git ~/.myhome
    chmod +x ~/.myhome/bin/init.sh && ~/.myhome/bin/init.sh

## What will happen

1. This repository will be cloned to your $HOME
2. Homebrew (and CLI tools) will be installed if not already present
3. Your shell environment, scripts, and configs will be initialized

## Useful preliminary steps

### (Re)generate a new SSH key and copy it to the clipboard for adding to GitHub

    rm -f ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub && ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q && cat ~/.ssh/id_ed25519.pub|pbcopy
