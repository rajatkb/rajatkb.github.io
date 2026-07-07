#!/bin/bash
# Sync vault between WSL repo (data/vault/) and Windows Obsidian vault.
# Always pushes from Repo → Windows so Obsidian sees the latest files.
# --watch: continuously watch repo vault and auto-push on changes (run via npm run watch)
# Default (no args): one-shot push.
#
# Requires VAULT_WIN_PATH env var pointing to the Windows vault (e.g.
# /mnt/c/Users/RAJAT/vault/). Set it in .zshrc / .bashrc.
#
# Runs a git diff first to detect new/changed/untracked files
# and verifies they arrive at the destination.

REPO_VAULT="$(cd "$(dirname "$0")/../data/vault/" && pwd)/"
: "${VAULT_WIN_PATH:?VAULT_WIN_PATH is not set — add it to .zshrc or export it before running sync-vault.sh}"
WIN_VAULT="$VAULT_WIN_PATH"
GIT_REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Detect new + modified + untracked files under data/vault/ relative to git HEAD
detect_repo_changes() {
  cd "$GIT_REPO"
  {
    git diff --name-only --diff-filter=ACMRT HEAD -- data/vault/
    git ls-files --others --exclude-standard -- data/vault/
  } | sort -u | sed 's|^data/vault/||'
}

# Detect files present in Windows vault but missing from repo vault
detect_win_new_files() {
  # Files on Windows side not present in repo
  find "$WIN_VAULT" -type f ! -path '*/.obsidian/*' -printf '%P\n' 2>/dev/null | sort -u > /tmp/_win_files.txt
  find "$REPO_VAULT" -type f -printf '%P\n' 2>/dev/null | sort -u > /tmp/_repo_files.txt
  comm -23 /tmp/_win_files.txt /tmp/_repo_files.txt
}

sync_pull() {
  echo "→ Scanning for new files in Windows vault..."
  local new_files
  new_files=$(detect_win_new_files)
  if [ -n "$new_files" ]; then
    echo "  Files only in Windows vault:"
    echo "$new_files" | sed 's/^/    /'
  else
    echo "  No new files detected."
  fi

  echo "→ Syncing Windows → Repo..."
  rsync -av --delete --exclude '.obsidian/workspace.json' "$WIN_VAULT" "$REPO_VAULT"

  echo "→ Verifying..."
  local missing=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if [ ! -f "${REPO_VAULT}${f}" ]; then
      echo "  MISSING: $f"
      missing=$((missing + 1))
    fi
  done <<< "$new_files"
  if [ "$missing" -gt 0 ]; then
    echo "⚠ $missing file(s) failed to sync from Windows!"
  else
    echo "  All detected files synced OK."
  fi
}

sync_push() {
  echo "→ Scanning for changed/untracked files via git..."
  local changes
  changes=$(detect_repo_changes)
  if [ -n "$changes" ]; then
    echo "  Changed/untracked files in data/vault/:"
    echo "$changes" | sed 's/^/    /'
  else
    echo "  No changes detected in data/vault/ since HEAD."
  fi

  echo "→ Syncing Repo → Windows..."
  rsync -av --delete --exclude '.obsidian/' "$REPO_VAULT" "$WIN_VAULT"

  echo "→ Verifying..."
  local missing=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if [ ! -f "${WIN_VAULT}${f}" ]; then
      echo "  MISSING: $f"
      missing=$((missing + 1))
    fi
  done <<< "$changes"
  if [ "$missing" -gt 0 ]; then
    echo "⚠ $missing file(s) failed to sync to Windows!"
  else
    echo "  All detected files synced OK."
  fi
}

watch_push() {
  echo "→ Watching $REPO_VAULT for changes..."
  inotifywait -m -r -e modify,create,delete,move \
    --exclude '.obsidian/' \
    --format '%w%f' \
    "$REPO_VAULT" 2>/dev/null | \
  while read; do
    # Debounce: wait 1.5s of quiet before syncing
    while read -t 1.5; do true; done
    echo "→ Change detected, pushing to Windows..."
    sync_push
  done
}

case "${1:-}" in
  --watch)
    echo "→ Watching repo for changes and pushing to Windows..."
    watch_push
    ;;
  *)
    echo "→ Repo → Windows (push)..."
    sync_push
    ;;
esac
echo "✓ Done."
