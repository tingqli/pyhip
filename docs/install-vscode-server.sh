#!/bin/bash

COMMIT_ID="4fe60c8b1cdac1c4c174f2fb180d0d758272d713"

INSTALL_DIR="$HOME/.vscode-server/bin/$COMMIT_ID"
mkdir -p "$INSTALL_DIR"

echo "download VS Code Server (Commit: $COMMIT_ID)..."
curl -fsSL "https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-x64/stable" \
  | tar -xz --strip-components=1 -C "$INSTALL_DIR"

touch "$INSTALL_DIR/0"
echo "finished"
