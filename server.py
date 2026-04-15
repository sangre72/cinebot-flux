"""diffusionkit FluxPipeline HTTP 서버.

이미지 1장 생성 후 프로세스 자체를 종료한다.
→ Apple Silicon Unified Memory 완전 반환.

요청 전 자동 시작: diffusionkit_service.py가 필요 시 subprocess로 띄움.

실행:
    /Users/bumsuklee/miniconda3/envs/diffusionkit/bin/python3.10 server.py

포트: 18188
엔드포인트:
    GET  /health    — 상태 확인
    POST /generate  — 이미지 생성 (완료 후 프로세스 종료)
"""

import base64
import io
import json
import logging
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("flux-server")

MODEL_VERSION = "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized"
PORT = 18188

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info(f"FluxPipeline 로딩: {MODEL_VERSION}")
        t = time.time()
        from diffusionkit.mlx import FluxPipeline
        _pipeline = FluxPipeline(model_version=MODEL_VERSION, low_memory_mode=False)
        logger.info(f"FluxPipeline 로딩 완료: {time.time() - t:.1f}s")
    return _pipeline


def _shutdown():
    """응답 전송 후 프로세스 종료 — Unified Memory 완전 반환."""
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
            self._send_json(200, {"status": "ok", "model": MODEL_VERSION})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
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

        try:
            pipeline = get_pipeline()
            t = time.time()
            logger.info(
                f"생성 시작: {width}x{height}, steps={steps}, "
                f"denoise={denoise}, prompt={prompt[:60]}..."
            )

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
            # 응답 완료 후 0.5초 뒤 프로세스 종료 → GPU 메모리 OS 반환
            Timer(0.5, _shutdown).start()


if __name__ == "__main__":
    logger.info(f"flux-server 시작 (port {PORT})")
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    logger.info(f"Ready: http://127.0.0.1:{PORT}")
    server.serve_forever()
