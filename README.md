# .myhome

Personal macOS setup — my dotfiles, scripts, and preferences for easy initialization.

## Quickstart

Install Homebrew, clone repository, and launch auto-configuration all in one copy-paste

- Read-only

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    git clone https://github.com/gramoflava/.myhome.git ~/.myhome
    chmod +x ~/.myhome/manage.sh && ~/.myhome/manage.sh --init
    ```

- Write access

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:gramoflava/.myhome.git ~/.myhome
    chmod +x ~/.myhome/manage.sh && ~/.myhome/manage.sh --init
    ```

## Useful commands

### (Re)generate a new SSH key and copy it to the clipboard for adding to GitHub

> **Note:** This command will overwrite existing `~/.ssh/id_ed25519*` keys. Back up any existing keys before proceeding or adjust the filenames to avoid data loss.

```bash
rm -f ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub && ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q && cat ~/.ssh/id_ed25519.pub | pbcopy
```
