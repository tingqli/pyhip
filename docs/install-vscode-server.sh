#!/bin/bash

COMMIT_ID="7e7950df89d055b5a378379db9ee14290772148a"

INSTALL_DIR="$HOME/.vscode-server/bin/$COMMIT_ID"
mkdir -p "$INSTALL_DIR"

echo "download VS Code Server (Commit: $COMMIT_ID)..."
curl -fsSL "https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-x64/stable" \
  | tar -xz --strip-components=1 -C "$INSTALL_DIR"

touch "$INSTALL_DIR/0"
echo "finished"