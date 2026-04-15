#!/bin/bash
# flux HTTP 서버 시작 스크립트
# Metal GPU(MPS) 풀 활용 — diffusionkit conda env 사용

PYTHON="/Users/bumsuklee/miniconda3/envs/diffusionkit/bin/python3.10"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "flux-server 시작: port 18188"
exec "$PYTHON" "$DIR/server.py"
