# Enable Copilot in container

```bash
openssl s_client -showcerts -connect api.githubcopilot.com:443 < /dev/null | \
  awk '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/' > /tmp/chain.pem
 
cat /tmp/chain.pem >> /etc/ssl/certs/ca-certificates.crt

# add following command to ~/.bashrc、~/.profile 
export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

```

# Install vscode server in container

```bash
export COMMIT_ID=4fe60c8b1cdac1c4c174f2fb180d0d758272d713

# install server in old folder structure: ~/.vscode-server/bin/$COMMIT_ID
curl -L "https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-x64/stable" -o "vscode-server-linux-x64-$COMMIT_ID.tar.gz" --progress-bar
tar -xzf vscode-server-linux-x64-$COMMIT_ID.tar.gz
mv vscode-server-linux-x64 ~/.vscode-server/bin/$COMMIT_ID

# install server in new folder structure : ~/.vscode-server/cli/servers/Stable-$COMMIT_ID/server
SERVER_DIR="$HOME/.vscode-server/cli/servers/Stable-$COMMIT_ID/server"
mkdir -p "$SERVER_DIR"

curl -fsSL "https://update.code.visualstudio.com/commit:$COMMIT_ID/server-linux-x64/stable" \
  | tar -xz --strip-components=1 -C "$SERVER_DIR"

# install CLI tool (code) for tunneling
CLI_TAR="vscode_cli_alpine_x64_cli.tar.gz"
curl -fsSL "https://update.code.visualstudio.com/commit:$COMMIT_ID/cli-linux-x64/stable" -o "$CLI_TAR"

tar -xzf "$CLI_TAR"
chmod +x code
mv code "$HOME/.vscode-server/code-$COMMIT_ID"

rm "$CLI_TAR"

echo "======= install finshed, run following command to start tunnel ======="
echo "~/.vscode-server/code-$COMMIT_ID tunnel"


# Alpine Linux version of code cli
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz
./code tunnel

```


