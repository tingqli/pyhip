# Start script through debugpy

```bash
python3 -m debugpy --listen localhost:5678 --wait-for-client xxx.py
```


# connect to it in vscode

```json
    {
    "name": "Python Debugger: Attach",
    "type": "debugpy",
    "justMyCode": false, // important
    "request": "attach",
        "connect": {
            "host": "localhost",
            "port": 5678
        }
    }
```