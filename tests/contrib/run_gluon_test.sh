#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm ~/.triton -rf
pytest $SCRIPT_DIR/gluon