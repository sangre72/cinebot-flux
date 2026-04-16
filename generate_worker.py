"""SSDFluxPipeline 이미지 생성 워커 (1회 실행 후 종료).

subprocess로 호출됨. 생성 완료 시 JSON을 stdout에 출력하고 종료.
프로세스 종료 = Metal GPU 메모리 완전 반환.

SSD 블록 스트리밍으로 RAM 사용량을 ~24GB → ~4GB로 절감.

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

# 4bit 양자화 safetensors 경로
import os
from pathlib import Path

_REPO_ID = "models--argmaxinc--mlx-FLUX.1-schnell-4bit-quantized"
_SFT_FILENAME = "flux-schnell-4bit-quantized.safetensors"


def _hf_cache_roots() -> list[Path]:
    """HuggingFace 캐시 루트 후보 목록 (우선순위 순)."""
    candidates = []
    for env_var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        v = os.environ.get(env_var)
        if v:
            candidates.append(Path(v).expanduser())
    # 알려진 고정 경로 후보
    candidates.append(Path("~/hf_home").expanduser())
    candidates.append(Path("~/.cache/huggingface").expanduser())
    return candidates


def _find_safetensors() -> Path:
    """HuggingFace 캐시에서 FLUX safetensors 파일 경로 탐색."""
    searched = []
    for cache_root in _hf_cache_roots():
        snapshots_dir = cache_root / "hub" / _REPO_ID / "snapshots"
        searched.append(str(snapshots_dir))
        if snapshots_dir.exists():
            for snapshot in sorted(snapshots_dir.iterdir()):
                candidate = snapshot / _SFT_FILENAME
                if candidate.exists():
                    return candidate

    raise FileNotFoundError(
        f"{_SFT_FILENAME} 파일을 찾을 수 없습니다.\n"
        f"검색한 경로:\n" + "\n".join(f"  {p}" for p in searched) + "\n"
        f"\n모델 다운로드 방법:\n"
        f"  python3.10 -c \"\n"
        f"  from diffusionkit.mlx import FluxPipeline\n"
        f"  FluxPipeline(model_version='{MODEL_VERSION}', low_memory_mode=True)\n"
        f"  \""
    )


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
        sft_path = _find_safetensors()
        logger.info(f"SSDFluxPipeline 로딩: {MODEL_VERSION}")
        logger.info(f"safetensors: {sft_path}")
        t0 = time.time()
        from ssd_stream import SSDFluxPipeline
        pipeline = SSDFluxPipeline(
            sft_path=sft_path,
            model_version=MODEL_VERSION,
            prefetch_depth=2,
            verbose=True,
        )
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
