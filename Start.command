#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
pip3 install flask psutil vastai -q 2>/dev/null
lsof -ti:8800 | xargs kill -9 2>/dev/null
echo ""
echo "  SplatMaker запускается..."
echo ""
python3 "$DIR/app.py"
