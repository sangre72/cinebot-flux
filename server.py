"""diffusionkit FluxPipeline HTTP 서버.

모델 상주 + 아이들 타임아웃 방식 (mFlux server.py와 동일).
- 첫 요청 시 모델 로드, 이후 재사용
- 마지막 요청 후 3분 경과 시 자동 종료 (Metal GPU 메모리 OS 반환)
- 다음 요청 시 diffusionkit_service가 자동 재기동

포트: 18188
엔드포인트:
    GET  /health    — 상태 확인
    POST /generate  — 이미지 생성
    POST /shutdown  — 명시적 종료
"""

import argparse
import base64
import io
import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Timer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("flux-server")

MODEL_SCHNELL = "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized"
MODEL_DEV = "argmaxinc/mlx-FLUX.1-dev"
DEFAULT_PORT = 18188
IDLE_TIMEOUT = 180  # 3분

_pipelines: dict = {}
_lock = threading.Lock()
_idle_timer: Timer | None = None
_last_request_time = 0.0


def _schedule_idle_shutdown():
    global _idle_timer
    if _idle_timer:
        _idle_timer.cancel()
    _idle_timer = Timer(IDLE_TIMEOUT, _idle_shutdown)
    _idle_timer.daemon = True
    _idle_timer.start()


def _idle_shutdown():
    elapsed = time.time() - _last_request_time
    if elapsed < IDLE_TIMEOUT - 5:
        _schedule_idle_shutdown()
        return
    logger.info(f"아이들 {IDLE_TIMEOUT}초 경과 → 프로세스 종료 (Metal GPU 메모리 반환)")
    os.kill(os.getpid(), signal.SIGTERM)


def get_pipeline(model_version: str = MODEL_SCHNELL):
    with _lock:
        if model_version not in _pipelines:
            logger.info(f"FluxPipeline 로딩: {model_version}")
            t = time.time()
            from diffusionkit.mlx import FluxPipeline
            _pipelines[model_version] = FluxPipeline(model_version=model_version, low_memory_mode=False)
            logger.info(f"FluxPipeline 로딩 완료: {time.time() - t:.1f}s")
    return _pipelines[model_version]


def _shutdown():
    logger.info("프로세스 종료 (Metal GPU 메모리 OS 반환)")
    os.kill(os.getpid(), signal.SIGTERM)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info(format % args)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_SCHNELL, "loaded": bool(_pipelines)})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        global _last_request_time

        if self.path == "/shutdown":
            self._send_json(200, {"success": True, "message": "shutting down"})
            Timer(0.3, _shutdown).start()
            return

        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._send_json(400, {"success": False, "error": "invalid JSON"})
            return

        prompt = body.get("prompt", "")
        width = int(body.get("width", 768))
        height = int(body.get("height", 432))
        steps = int(body.get("steps", 4))
        seed = body.get("seed", None)
        ref_image_b64 = body.get("ref_image_b64", None)
        denoise = float(body.get("denoise", 1.0))
        flux_model = body.get("flux_model", "schnell")
        model_version = MODEL_DEV if flux_model == "dev" else MODEL_SCHNELL

        _last_request_time = time.time()

        try:
            pipeline = get_pipeline(model_version)
            t = time.time()
            logger.info(f"생성 시작: {width}x{height}, steps={steps}, denoise={denoise}, prompt={prompt[:60]}...")

            image_path = None
            if ref_image_b64:
                from PIL import Image
                raw = ref_image_b64.split(",", 1)[-1]
                ref_img = Image.open(io.BytesIO(base64.b64decode(raw))).resize((width, height))
                image_path = f"/tmp/flux_ref_{int(time.time())}.png"
                ref_img.save(image_path)

            images, _ = pipeline.generate_image(
                prompt,
                cfg_weight=0.0,
                num_steps=steps,
                latent_size=(height // 8, width // 8),
                seed=seed,
                image_path=image_path,
                denoise=denoise if image_path else 1.0,
            )

            from PIL import Image as PILImage
            if isinstance(images, PILImage.Image):
                img = images
            elif hasattr(images, '__getitem__'):
                item = images[0]
                img = item if isinstance(item, PILImage.Image) else PILImage.fromarray(item)
            else:
                import mlx.core as mx
                img = PILImage.fromarray((mx.clip(images, 0, 1) * 255).astype(mx.uint8).tolist())

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            elapsed = time.time() - t
            logger.info(f"생성 완료: {elapsed:.1f}s, {len(buf.getvalue()) // 1024}KB")

            self._send_json(200, {
                "success": True,
                "image": f"data:image/png;base64,{b64}",
                "elapsed": elapsed,
            })

        except Exception as e:
            logger.error(f"생성 실패: {e}", exc_info=True)
            self._send_json(500, {"success": False, "error": str(e)})

        finally:
            _last_request_time = time.time()
            _schedule_idle_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    port = args.port

    logger.info(f"flux-server 시작 (port {port}, idle_timeout={IDLE_TIMEOUT}s)")
    if args.preload:
        get_pipeline()
        _last_request_time = time.time()
        _schedule_idle_shutdown()
    server = HTTPServer(("127.0.0.1", port), Handler)
    logger.info(f"Ready: http://127.0.0.1:{port}")
    server.serve_forever()
