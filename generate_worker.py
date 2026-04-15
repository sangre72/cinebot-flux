"""FluxPipeline 이미지 생성 워커 (1회 실행 후 종료).

subprocess로 호출됨. 생성 완료 시 JSON을 stdout에 출력하고 종료.
프로세스 종료 = Metal GPU 메모리 완전 반환.

사용:
    python3.10 generate_worker.py <json_input_file> <json_output_file>
"""

import base64
import io
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("flux-worker")

MODEL_VERSION = "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized"


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "인자 부족: input_file output_file"}))
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file) as f:
        params = json.load(f)

    prompt = params.get("prompt", "")
    width = int(params.get("width", 768))
    height = int(params.get("height", 432))
    steps = int(params.get("steps", 4))
    seed = params.get("seed", None)
    ref_image_b64 = params.get("ref_image_b64", None)
    denoise = float(params.get("denoise", 1.0))

    try:
        logger.info(f"FluxPipeline 로딩: {MODEL_VERSION}")
        t0 = time.time()
        from diffusionkit.mlx import FluxPipeline
        pipeline = FluxPipeline(model_version=MODEL_VERSION, low_memory_mode=False)
        logger.info(f"로딩 완료: {time.time()-t0:.1f}s")

        # img2img 참조 이미지
        image_path = None
        if ref_image_b64:
            from PIL import Image
            raw = ref_image_b64.split(",", 1)[-1]
            ref_img = Image.open(io.BytesIO(base64.b64decode(raw))).resize((width, height))
            image_path = f"/tmp/flux_ref_{int(time.time())}.png"
            ref_img.save(image_path)

        t1 = time.time()
        logger.info(f"생성 시작: {width}x{height}, steps={steps}, denoise={denoise}")
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
        elapsed = time.time() - t1
        logger.info(f"생성 완료: {elapsed:.1f}s, {len(buf.getvalue())//1024}KB")

        result = {
            "success": True,
            "image": f"data:image/png;base64,{b64}",
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"생성 실패: {e}", exc_info=True)
        result = {"success": False, "error": str(e)}

    with open(output_file, "w") as f:
        json.dump(result, f)

    # 프로세스 종료 → Metal GPU 메모리 OS 완전 반환
    logger.info("워커 종료 (GPU 메모리 반환)")
    sys.exit(0)


if __name__ == "__main__":
    main()
