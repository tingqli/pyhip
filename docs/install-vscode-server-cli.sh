#!/bin/bash

COMMIT_ID=4fe60c8b1cdac1c4c174f2fb180d0d758272d713

SERVER_DIR="$HOME/.vscode-server/cli/servers/Stable-$COMMIT_ID/server"
mkdir -p "$SERVER_DIR"

echo "正在下载并安装 VS Code Server (Commit: $COMMIT_ID)..."
curl -fsSL "https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-x64/stable" \
  | tar -xz --strip-components=1 -C "$SERVER_DIR"

CLI_TAR="vscode_cli_alpine_x64_cli.tar.gz"
curl -fsSL "https://update.code.visualstudio.com/commit:$COMMIT_ID/cli-linux-x64/stable" -o "$CLI_TAR"

tar -xzf "$CLI_TAR"
chmod +x code
mv code "$HOME/.vscode-server/code-$COMMIT_ID"

rm "$CLI_TAR"

echo "$HOME/.vscode-server/code-$COMMIT_ID tunnel"

