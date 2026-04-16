"""
SSD Block Streaming for FLUX.1 DiT (ltx-flash 방식 포팅).

ltx-flash의 Flash-MoE 원리를 FLUX.1 MMDiT에 적용:
  - 전체 모델을 RAM에 올리는 대신 블록을 SSD에서 하나씩 스트리밍
  - Non-block 가중치만 RAM 상주

구조:
  flux-schnell-4bit-quantized.safetensors (~6.6GB)
    multimodal_transformer_blocks.{N}.*   (19개, Double Stream)
    unified_transformer_blocks.{N}.*      (38개, Single Stream)
  + 비블록 가중치 → RAM 상주

실행 시 메모리:
  - RAM 상주: 비블록 가중치 ~0.4GB + 활성화 ~1GB
  - 스트리밍: 블록 1개 on-demand (~191MB double / ~80MB single)
  - 목표 총 RAM: ~3~4GB (기존 ~24GB 대비 85%+ 절약)

핵심 설계:
  - cache_modulation_params() 제거: 블록 없이 adaLN 계산 불가
  - 대신 블록 forward 직전 modulation_params를 블록 객체에 직접 주입
  - diffusionkit MMDiT.__call__() 인터페이스 완전 호환
"""

from __future__ import annotations

import mmap
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from diffusionkit.mlx.config import FLUX_SCHNELL, MMDiTConfig, PositionalEncoding
from diffusionkit.mlx.mmdit import (
    FinalLayer,
    LatentImageAdapter,
    MultiModalTransformerBlock,
    PooledTextEmbeddingAdapter,
    RoPE,
    TimestepAdapter,
    UnifiedTransformerBlock,
    affine_transform,
    unpatchify,
)


# ─────────────────────────────────────────────
# safetensors dtype 매핑
# ─────────────────────────────────────────────

_DTYPE_MAP = {
    "BF16": mx.bfloat16,
    "F16": mx.float16,
    "F32": mx.float32,
    "I32": mx.int32,
    "U32": mx.uint32,
    "U8": mx.uint8,
    "I8": mx.int8,
}


# ─────────────────────────────────────────────
# BlockIndex — safetensors 텐서 오프셋 인덱스
# ─────────────────────────────────────────────

class FluxBlockIndex:
    """
    safetensors 파일에서 FLUX 블록 텐서의 바이트 오프셋 인덱스.

    FLUX diffusionkit 변환 후 키 패턴:
      multimodal_transformer_blocks.{N}.*   (Double Stream)
      unified_transformer_blocks.{N}.*      (Single Stream)
    """

    DOUBLE_PREFIX = "multimodal_transformer_blocks."
    SINGLE_PREFIX = "unified_transformer_blocks."

    def __init__(self, sft_path: str | Path):
        self.path = Path(sft_path)
        self._offsets: dict[str, tuple[int, int, str, tuple]] = {}
        self._header_size: int = 0
        self._fd: int = -1
        self._mmap: mmap.mmap | None = None
        self._build_index()

    def _build_index(self):
        import json
        import struct

        with open(self.path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_len).decode("utf-8")
            self._header_size = 8 + header_len

        header = json.loads(header_json)
        for key, info in header.items():
            if key == "__metadata__":
                continue
            dtype = info["dtype"]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            self._offsets[key] = (
                self._header_size + start,
                self._header_size + end,
                dtype,
                shape,
            )

    def block_keys(self, block_type: str, block_idx: int) -> list[str]:
        """블록 타입('multimodal' 또는 'unified')과 인덱스로 키 목록 반환."""
        if block_type == "multimodal":
            prefix = f"{self.DOUBLE_PREFIX}{block_idx}."
        else:
            prefix = f"{self.SINGLE_PREFIX}{block_idx}."
        return [k for k in self._offsets if k.startswith(prefix)]

    def non_block_keys(self) -> list[str]:
        """블록 가중치가 아닌 비블록 키 반환 (RAM 상주 대상)."""
        return [
            k for k in self._offsets
            if self.DOUBLE_PREFIX not in k and self.SINGLE_PREFIX not in k
        ]

    def get_offset(self, key: str) -> tuple[int, int, str, tuple]:
        return self._offsets[key]

    def all_keys(self) -> list[str]:
        return list(self._offsets.keys())

    def open_fd(self) -> int:
        if self._fd < 0:
            self._fd = os.open(str(self.path), os.O_RDONLY)
        return self._fd

    def open_mmap(self) -> mmap.mmap:
        """mmap으로 전체 파일 매핑 — OS page cache 직접 참조."""
        if self._mmap is None:
            fd = self.open_fd()
            self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        return self._mmap

    def close_fd(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


# ─────────────────────────────────────────────
# 저수준 텐서 로드 함수
# ─────────────────────────────────────────────

def _mmap_tensor(mm: mmap.mmap, start: int, end: int, dtype_str: str, shape: tuple) -> mx.array:
    """
    mmap 슬라이스로 텐서를 읽어 mx.array로 변환.
    mmap은 멀티스레드에서 내부 포지션 공유 이슈 → bytes()로 복사 후 사용.
    """
    buf = bytes(mm[start:end])
    dtype_upper = dtype_str.upper()

    if dtype_upper == "BF16":
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        return mx.array(arr).view(mx.bfloat16)
    elif dtype_upper == "U32":
        arr = np.frombuffer(buf, dtype=np.uint32).reshape(shape)
        return mx.array(arr)
    elif dtype_upper == "F16":
        arr = np.frombuffer(buf, dtype=np.float16).reshape(shape)
        return mx.array(arr)
    elif dtype_upper == "F32":
        arr = np.frombuffer(buf, dtype=np.float32).reshape(shape)
        return mx.array(arr)
    else:
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        return mx.array(arr)


def _pread_tensor(fd: int, start: int, end: int, dtype_str: str, shape: tuple) -> mx.array:
    """pread()로 SSD에서 텐서를 직접 읽음 (mmap 불가 환경 fallback)."""
    n_bytes = end - start
    buf = bytearray(n_bytes)
    view = memoryview(buf)
    total = 0
    while total < n_bytes:
        chunk = os.pread(fd, n_bytes - total, start + total)
        if not chunk:
            break
        view[total:total + len(chunk)] = chunk
        total += len(chunk)

    dtype_upper = dtype_str.upper()
    if dtype_upper == "BF16":
        arr = np.frombuffer(bytes(buf), dtype=np.uint16).reshape(shape)
        return mx.array(arr).view(mx.bfloat16)
    elif dtype_upper == "U32":
        arr = np.frombuffer(bytes(buf), dtype=np.uint32).reshape(shape)
        return mx.array(arr)
    elif dtype_upper == "F16":
        arr = np.frombuffer(bytes(buf), dtype=np.float16).reshape(shape)
        return mx.array(arr)
    elif dtype_upper == "F32":
        arr = np.frombuffer(bytes(buf), dtype=np.float32).reshape(shape)
        return mx.array(arr)
    else:
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(shape)
        return mx.array(arr)


# ─────────────────────────────────────────────
# SSDFluxBlockLoader
# ─────────────────────────────────────────────

class SSDFluxBlockLoader:
    """
    FLUX.1 Transformer Block SSD 스트리머.

    최적화:
      1. N-ahead prefetch: prefetch_depth 블록 미리 읽기
      2. 즉시 GPU 전송: 로드 후 mx.eval()
      3. mmap: OS page cache 직접 참조
      4. 블록 객체 풀링: MultiModalTransformerBlock / UnifiedTransformerBlock 각 1개 재사용
    """

    def __init__(
        self,
        sft_path: str | Path,
        config: MMDiTConfig,
        num_io_threads: int = 8,
        prefetch_depth: int = 2,
        use_mmap: bool = True,
    ):
        self.sft_path = Path(sft_path)
        self.config = config
        self.prefetch_depth = prefetch_depth
        self._index = FluxBlockIndex(sft_path)
        self._executor = ThreadPoolExecutor(
            max_workers=num_io_threads, thread_name_prefix="ssd-flux"
        )

        if use_mmap:
            self._mm = self._index.open_mmap()
            self._fd = -1
        else:
            self._fd = self._index.open_fd()
            self._mm = None

    def _load_tensor(self, start: int, end: int, dtype: str, shape: tuple) -> mx.array:
        if self._mm is not None:
            return _mmap_tensor(self._mm, start, end, dtype, shape)
        else:
            return _pread_tensor(self._fd, start, end, dtype, shape)

    def load_block_weights(self, block_type: str, block_idx: int) -> dict[str, mx.array]:
        """
        블록의 모든 텐서를 ThreadPoolExecutor로 병렬 로드.
        키는 블록 prefix를 제거한 로컬 키로 반환.
        """
        keys = self._index.block_keys(block_type, block_idx)
        if block_type == "multimodal":
            prefix = f"{FluxBlockIndex.DOUBLE_PREFIX}{block_idx}."
        else:
            prefix = f"{FluxBlockIndex.SINGLE_PREFIX}{block_idx}."

        futures: dict[str, Future] = {}
        for key in keys:
            start, end, dtype, shape = self._index.get_offset(key)
            local_key = key[len(prefix):]
            futures[local_key] = self._executor.submit(
                self._load_tensor, start, end, dtype, shape
            )

        return {k: f.result() for k, f in futures.items()}

    def load_non_block_weights(self) -> dict[str, mx.array]:
        """비블록 가중치를 병렬 로드 — RAM 상주용."""
        keys = self._index.non_block_keys()
        futures: dict[str, Future] = {}
        for key in keys:
            start, end, dtype, shape = self._index.get_offset(key)
            futures[key] = self._executor.submit(
                self._load_tensor, start, end, dtype, shape
            )
        return {k: f.result() for k, f in futures.items()}

    def stream_blocks(
        self,
        block_type: str,
        num_blocks: int,
    ) -> Iterator[tuple[nn.Module, dict, float]]:
        """
        블록을 SSD에서 순차 스트리밍.

        N-ahead prefetch + 블록 객체 풀링.

        Yields:
            (block_module, weights_dict, t_load_seconds)
        """
        # 블록 객체 1개 생성 후 재사용
        # 4bit 양자화 체크포인트: nn.quantize(block) 후 block.update()로 가중치 주입
        if block_type == "multimodal":
            # depth_unified > 0 이면 마지막 double block도 text 처리 필요 → skip_text_post_sdpa=False
            block = MultiModalTransformerBlock(self.config, skip_text_post_sdpa=False)
        else:
            block = UnifiedTransformerBlock(self.config)

        nn.quantize(block)  # 4bit 구조로 변환 (1회만 실행, 이후 update()로 가중치 교체)

        # N-ahead prefetch 큐
        prefetch_queue: deque[tuple[int, Future]] = deque()

        def _enqueue(idx: int):
            if idx < num_blocks:
                f = self._executor.submit(self.load_block_weights, block_type, idx)
                prefetch_queue.append((idx, f))

        # 초기 prefetch_depth개 선발
        for i in range(min(self.prefetch_depth, num_blocks)):
            _enqueue(i)

        for block_idx in range(num_blocks):
            t_load_start = time.perf_counter()

            queued_idx, future = prefetch_queue.popleft()
            assert queued_idx == block_idx
            weights = future.result()
            t_load = time.perf_counter() - t_load_start

            # 다음 prefetch 추가 (슬라이딩 윈도우)
            _enqueue(block_idx + self.prefetch_depth)

            # 블록 재사용: 가중치만 교체 (quantize 구조 유지, update로 값만 갱신)
            block.update(tree_unflatten(tree_flatten(weights)))
            mx.eval(block.parameters())

            yield block, weights, t_load

            del weights
            mx.metal.clear_cache()

        del block

    def close(self):
        self._index.close_fd()
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# SSDStreamingMMDiT — MMDiT 드롭인 대체
# ─────────────────────────────────────────────

class SSDStreamingMMDiT(nn.Module):
    """
    FLUX.1 MMDiT SSD 스트리밍 버전.

    diffusionkit MMDiT.__call__()과 완전히 동일한 인터페이스.

    핵심 차이:
      - multimodal/unified 블록 가중치는 SSD에서 on-demand 스트리밍
      - 비블록 가중치(x_embedder, t_embedder, y_embedder 등)는 RAM 상주
      - cache_modulation_params() 제거 → 블록 forward 직전 실시간 계산 주입

    adaLN 실시간 계산 전략:
      TransformerBlock.pre_sdpa()는 self._modulation_params[timestep.item()]을 참조.
      블록 가중치가 로드된 후, 해당 블록의 adaLN_modulation을 직접 호출해
      _modulation_params를 주입한 뒤 forward를 수행.
    """

    def __init__(
        self,
        sft_path: str | Path,
        config: MMDiTConfig | None = None,
        prefetch_depth: int = 2,
        use_mmap: bool = True,
        num_io_threads: int = 8,
        verbose: bool = True,
    ):
        super().__init__()
        self.config = config or FLUX_SCHNELL
        self.verbose = verbose
        self.sft_path = Path(sft_path)

        if not self.sft_path.exists():
            raise FileNotFoundError(f"safetensors 파일 없음: {self.sft_path}")

        # ── 비블록 가중치 구성 요소 선언 (load_weights용) ──
        if self.config.guidance_embed:
            from diffusionkit.mlx.mmdit import MLPEmbedder
            self.guidance_in = MLPEmbedder(
                in_dim=self.config.frequency_embed_dim,
                hidden_dim=self.config.hidden_size,
            )
        else:
            self.guidance_in = nn.Identity()

        self.x_embedder = LatentImageAdapter(self.config)
        self.y_embedder = PooledTextEmbeddingAdapter(self.config)
        self.t_embedder = TimestepAdapter(self.config)
        self.context_embedder = nn.Linear(
            self.config.token_level_text_embed_dim,
            self.config.hidden_size,
        )
        self.pre_sdpa_rope = RoPE(
            theta=10000,
            axes_dim=self.config.rope_axes_dim,
        )
        self.final_layer = FinalLayer(self.config)

        # 블록은 스트리밍 — 실제 nn.Module 리스트로 두지 않음
        # (parameters()를 스캔할 때 블록 가중치가 포함되지 않도록)
        self.multimodal_transformer_blocks = []
        self.unified_transformer_blocks = []

        # ── SSD 로더 초기화 ──
        if verbose:
            print(f"[SSDStreamingMMDiT] 비블록 가중치 로드 중...", flush=True)
            print(f"  safetensors: {self.sft_path}", flush=True)
            print(f"  prefetch depth: {prefetch_depth}블록", flush=True)

        t0 = time.perf_counter()
        self._loader = SSDFluxBlockLoader(
            sft_path=sft_path,
            config=self.config,
            num_io_threads=num_io_threads,
            prefetch_depth=prefetch_depth,
            use_mmap=use_mmap,
        )

        # 비블록 가중치 로드 및 주입
        # 4bit 양자화 체크포인트: nn.quantize(self) 후 update()로 주입
        nn.quantize(self)
        non_block_weights = self._loader.load_non_block_weights()
        self.update(tree_unflatten(tree_flatten(non_block_weights)))
        mx.eval(self.parameters())

        elapsed = time.perf_counter() - t0
        if verbose:
            print(
                f"[SSDStreamingMMDiT] 비블록 가중치 로드 완료 "
                f"({elapsed:.1f}초, {len(non_block_weights)}개 텐서)",
                flush=True,
            )
            print(
                f"[SSDStreamingMMDiT] 블록 가중치: SSD 스트리밍 모드",
                flush=True,
            )

    def _compute_timestep_cond(self, pooled_text_embeddings: mx.array, timestep: mx.array) -> mx.array:
        """
        timestep conditioning 계산 (비블록 가중치 사용).
        MMDiT.__call__ 내부의 동일 연산 추출.
        """
        y_embed = self.y_embedder(pooled_text_embeddings)
        t_embed = self.t_embedder(timestep)
        return y_embed[:, None, None, :] + t_embed

    def _inject_modulation_params(
        self,
        block: nn.Module,
        timestep_key: float,
        modulation_inputs: mx.array,
        is_multimodal: bool,
    ):
        """
        블록의 adaLN_modulation을 직접 호출해 _modulation_params를 주입.
        TransformerBlock.pre_sdpa()가 _modulation_params를 캐시에서 읽으므로
        블록 forward 전에 반드시 주입해야 함.
        """
        if is_multimodal:
            # MultiModalTransformerBlock: image + text 두 TransformerBlock
            img_block = block.image_transformer_block
            txt_block = block.text_transformer_block

            if not hasattr(img_block, "_modulation_params"):
                img_block._modulation_params = {}
            if not hasattr(txt_block, "_modulation_params"):
                txt_block._modulation_params = {}

            img_block._modulation_params[timestep_key] = img_block.adaLN_modulation(modulation_inputs)
            txt_block._modulation_params[timestep_key] = txt_block.adaLN_modulation(modulation_inputs)
            mx.eval(
                img_block._modulation_params[timestep_key],
                txt_block._modulation_params[timestep_key],
            )
        else:
            # UnifiedTransformerBlock: transformer_block 1개
            tb = block.transformer_block
            if not hasattr(tb, "_modulation_params"):
                tb._modulation_params = {}
            tb._modulation_params[timestep_key] = tb.adaLN_modulation(modulation_inputs)
            mx.eval(tb._modulation_params[timestep_key])

    def __call__(
        self,
        latent_image_embeddings: mx.array,
        token_level_text_embeddings: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        """
        MMDiT.__call__()과 완전히 동일한 인터페이스.

        FLUX.1 forward 흐름:
          1. x_embedder: latent → image_embeds
          2. context_embedder: T5 → text_embeds
          3. y_embedder + t_embedder: timestep conditioning
          4. pre_sdpa_rope: RoPE positional encodings
          5. 19개 MultiModalTransformerBlock (SSD 스트리밍)
          6. concat → 38개 UnifiedTransformerBlock (SSD 스트리밍)
          7. final_layer → output latent
        """
        batch, latent_height, latent_width, _ = latent_image_embeddings.shape

        # ── 1. 텍스트 임베딩 프로젝션 ──
        token_level_text_embeddings = self.context_embedder(token_level_text_embeddings)

        # ── 2. 이미지 패치 임베딩 (patchify_via_reshape) ──
        latent_image_embeddings = self.x_embedder(latent_image_embeddings)
        latent_image_embeddings = latent_image_embeddings.reshape(
            batch, -1, 1, self.config.hidden_size
        )

        # ── 3. RoPE positional encodings ──
        positional_encodings = self.pre_sdpa_rope(
            text_sequence_length=token_level_text_embeddings.shape[1],
            latent_image_resolution=(
                latent_height // self.config.patch_size,
                latent_width // self.config.patch_size,
            ),
        )

        # ── 4. adaLN modulation_inputs 조회 ──
        # cache_modulation_params()에서 timestep별 modulation_inputs를 사전 계산해
        # _cached_modulation_inputs에 저장해 둠.
        # (sample_euler → CFGDenoiser.cache_modulation_params() → 여기 호출 경로)
        if not hasattr(self, "_cached_modulation_inputs") or self._cached_modulation_inputs is None:
            raise RuntimeError(
                "cache_modulation_params()를 먼저 호출해야 합니다. "
                "SSDStreamingMMDiT는 CFGDenoiser.cache_modulation_params()를 통해 "
                "modulation_inputs를 사전 계산합니다."
            )

        # timestep key: 현재 timestep 스칼라 값 (pre_sdpa()에서 lookup용)
        if timestep.size > 1:
            timestep_key = timestep[0].item()
        else:
            timestep_key = timestep.item()

        modulation_inputs = self._cached_modulation_inputs.get(timestep_key)
        if modulation_inputs is None:
            # fallback: 가장 가까운 key 사용
            available_keys = list(self._cached_modulation_inputs.keys())
            if available_keys:
                closest = min(available_keys, key=lambda k: abs(k - timestep_key))
                modulation_inputs = self._cached_modulation_inputs[closest]
            else:
                raise RuntimeError(
                    f"timestep_key={timestep_key}에 대한 modulation_inputs 없음. "
                    f"사용 가능한 키: {list(self._cached_modulation_inputs.keys())}"
                )

        # FLUX-dev: guidance_embed=True → timestep을 guidance_in 통과
        # FLUX-schnell: guidance_embed=False → timestep 그대로
        # 블록 forward에서 timestep은 _modulation_params dict lookup 키로만 사용됨
        if self.config.guidance_embed:
            timestep_for_blocks = self.guidance_in(self.t_embedder(timestep))
        else:
            timestep_for_blocks = timestep

        # ── 5. Double Stream 블록 (19개) — SSD 스트리밍 ──
        for block_idx, (block, weights, t_load) in enumerate(
            self._loader.stream_blocks("multimodal", self.config.depth_multimodal)
        ):
            t_gpu_start = time.perf_counter()

            # adaLN modulation params 주입
            self._inject_modulation_params(
                block, timestep_key, modulation_inputs, is_multimodal=True
            )

            # 마지막 double block: skip_text_post_sdpa 처리
            # (unified block이 있으면 마지막 double block에서 text 계속 처리)
            latent_image_embeddings, token_level_text_embeddings = block(
                latent_image_embeddings,
                token_level_text_embeddings,
                timestep_for_blocks,
                positional_encodings=positional_encodings,
            )
            mx.eval(latent_image_embeddings)
            if token_level_text_embeddings is not None:
                mx.eval(token_level_text_embeddings)

            t_gpu = time.perf_counter() - t_gpu_start
            if self.verbose and (block_idx % 5 == 0 or block_idx == self.config.depth_multimodal - 1):
                print(
                    f"  double block {block_idx:2d}/{self.config.depth_multimodal}: "
                    f"SSD={t_load*1000:.0f}ms GPU={t_gpu*1000:.0f}ms",
                    flush=True,
                )

        # ── 6. Single Stream 블록 (38개) — SSD 스트리밍 ──
        if self.config.depth_unified > 0:
            # text + image concat
            latent_unified_embeddings = mx.concatenate(
                (token_level_text_embeddings, latent_image_embeddings), axis=1
            )
            txt_seq_len = token_level_text_embeddings.shape[1]

            for block_idx, (block, weights, t_load) in enumerate(
                self._loader.stream_blocks("unified", self.config.depth_unified)
            ):
                t_gpu_start = time.perf_counter()

                self._inject_modulation_params(
                    block, timestep_key, modulation_inputs, is_multimodal=False
                )

                latent_unified_embeddings = block(
                    latent_unified_embeddings,
                    timestep_for_blocks,
                    positional_encodings=positional_encodings,
                )
                mx.eval(latent_unified_embeddings)

                t_gpu = time.perf_counter() - t_gpu_start
                if self.verbose and (block_idx % 10 == 0 or block_idx == self.config.depth_unified - 1):
                    print(
                        f"  single block {block_idx:2d}/{self.config.depth_unified}: "
                        f"SSD={t_load*1000:.0f}ms GPU={t_gpu*1000:.0f}ms",
                        flush=True,
                    )

            latent_image_embeddings = latent_unified_embeddings[:, txt_seq_len:, ...]

        # ── 7. Final layer ──
        # FinalLayer.__call__도 _modulation_params를 사용 → 주입
        if not hasattr(self.final_layer, "_modulation_params"):
            self.final_layer._modulation_params = {}
        self.final_layer._modulation_params[timestep_key] = self.final_layer.adaLN_modulation(
            modulation_inputs
        )
        mx.eval(self.final_layer._modulation_params[timestep_key])

        latent_image_embeddings = self.final_layer(
            latent_image_embeddings,
            timestep_for_blocks,
        )

        # ── 8. Unpatchify ──
        if self.config.patchify_via_reshape:
            latent_image_embeddings = self.x_embedder.unpack(
                latent_image_embeddings, (latent_height, latent_width)
            )
        else:
            latent_image_embeddings = unpatchify(
                latent_image_embeddings,
                patch_size=self.config.patch_size,
                target_height=latent_height,
                target_width=latent_width,
                vae_latent_dim=self.config.vae_latent_dim,
            )

        return latent_image_embeddings

    def cache_modulation_params(
        self,
        pooled_text_embeddings: mx.array,
        timesteps: mx.array,
    ):
        """
        CFGDenoiser.cache_modulation_params()와 동일한 시그니처.

        MMDiT.cache_modulation_params()를 오버라이드:
          - 블록 adaLN 가중치를 오프로드하지 않음 (어차피 SSD에서 로드)
          - 대신 각 timestep별 modulation_inputs만 미리 계산해 저장

        sample_euler() → CFGDenoiser.cache_modulation_params() 경로로 호출됨.
        """
        y_embed = self.y_embedder(pooled_text_embeddings)

        self._cached_modulation_inputs = {}
        for timestep in timesteps:
            timestep_key = timestep.item()
            batch_size = pooled_text_embeddings.shape[0]
            modulation_inputs = y_embed[:, None, None, :] + self.t_embedder(
                mx.repeat(timestep[None], batch_size, axis=0)
            )
            mx.eval(modulation_inputs)
            self._cached_modulation_inputs[timestep_key] = modulation_inputs

        if self.verbose:
            print(
                f"[SSDStreamingMMDiT] modulation_inputs 사전 계산 완료 "
                f"({len(self._cached_modulation_inputs)}개 timestep)",
                flush=True,
            )

    def load_weights(self, weights, strict: bool = True):
        """
        CFGDenoiser.clear_cache()에서 load_weights([])를 호출하면 no-op.
        비어있지 않은 경우 상위 클래스에 위임.
        """
        if not weights:
            return self  # 빈 가중치 → 무시
        return super().load_weights(weights, strict=strict)

    def clear_modulation_params_cache(self):
        """modulation_inputs 캐시 정리."""
        self._cached_modulation_inputs = None
        if hasattr(self.final_layer, "_modulation_params"):
            del self.final_layer._modulation_params

    def close(self):
        self._loader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# VAE Tiled Decoding
# ─────────────────────────────────────────────

def _blend_v(a: mx.array, b: mx.array, blend_extent: int) -> mx.array:
    """
    세로(H) 경계 linear blending. NHWC 포맷.
    a: 위쪽 타일, b: 아래쪽 타일.
    b의 상단 blend_extent 행을 a의 하단과 선형 혼합.
    """
    blend_extent = min(a.shape[1], b.shape[1], blend_extent)
    # weight: 0/N ... (N-1)/N  → b 비중이 점점 커짐
    weight = (mx.arange(blend_extent, dtype=mx.float32) / blend_extent).reshape(1, blend_extent, 1, 1)
    blended = a[:, -blend_extent:, :, :] * (1.0 - weight) + b[:, :blend_extent, :, :] * weight
    return mx.concatenate([blended, b[:, blend_extent:, :, :]], axis=1)


def _blend_h(a: mx.array, b: mx.array, blend_extent: int) -> mx.array:
    """
    가로(W) 경계 linear blending. NHWC 포맷.
    a: 왼쪽 타일, b: 오른쪽 타일.
    """
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    weight = (mx.arange(blend_extent, dtype=mx.float32) / blend_extent).reshape(1, 1, blend_extent, 1)
    blended = a[:, :, -blend_extent:, :] * (1.0 - weight) + b[:, :, :blend_extent, :] * weight
    return mx.concatenate([blended, b[:, :, blend_extent:, :]], axis=2)


def vae_tiled_decode(
    decoder,
    z: mx.array,
    tile_size: int = 32,
    overlap: int = 8,
    verbose: bool = False,
) -> mx.array:
    """
    VAE decoder tiled decoding.

    latent를 겹치는 타일로 분할해 순차 디코딩 후 linear blending으로 seam 제거.
    해상도에 무관하게 peak 메모리를 타일 1개 분량으로 고정.

    Args:
        decoder: diffusionkit VAEDecoder (또는 Autoencoder.decoder)
        z: latent (B, H_lat, W_lat, C), post_quant_proj 적용 후 값
        tile_size: latent 공간 기준 타일 크기 (기본 32 = 256px)
        overlap: latent 공간 기준 타일 간 겹침 (기본 8 = 64px)
        verbose: 진행 상황 출력

    Returns:
        decoded: (B, H_pix, W_pix, 3)
    """
    B, H_lat, W_lat, C = z.shape
    stride = tile_size - overlap
    vae_scale = 8  # FLUX VAE: 8× upsample
    blend_extent = overlap * vae_scale  # pixel 공간 blending 폭
    row_limit = tile_size * vae_scale - blend_extent  # 타일에서 실제 사용할 행/열 수

    # latent가 타일 하나보다 작으면 그냥 통째로 디코딩
    if H_lat <= tile_size and W_lat <= tile_size:
        return decoder(z)

    # 타일 좌상단 좌표 목록 생성 (stride 간격, 마지막 타일은 경계에 맞춤)
    def tile_coords(total, ts, st):
        coords = list(range(0, total - ts, st))
        coords.append(total - ts)  # 마지막 타일: 오른쪽/아래쪽 경계에 정렬
        # 중복 제거 (total <= ts 인 경우 대비)
        seen = []
        for c in coords:
            if not seen or c != seen[-1]:
                seen.append(max(0, c))
        return seen

    h_coords = tile_coords(H_lat, tile_size, stride)
    w_coords = tile_coords(W_lat, tile_size, stride)
    n_tiles = len(h_coords) * len(w_coords)

    if verbose:
        print(
            f"[vae_tiled_decode] latent={H_lat}×{W_lat} → "
            f"tile={tile_size}(overlap={overlap}) → {n_tiles}개 타일",
            flush=True,
        )

    # ── 1단계: 각 타일 디코딩 ──
    # rows[i][j] = decoded tile (B, tile_size*8, tile_size*8, 3)
    rows = []
    tile_idx = 0
    for i, hi in enumerate(h_coords):
        row_tiles = []
        for j, wi in enumerate(w_coords):
            tile = z[:, hi : hi + tile_size, wi : wi + tile_size, :]
            decoded = decoder(tile)
            mx.eval(decoded)
            mx.metal.clear_cache()
            row_tiles.append(decoded)
            tile_idx += 1
            if verbose:
                print(f"  tile {tile_idx}/{n_tiles} ({hi},{wi})", flush=True)
        rows.append(row_tiles)

    # ── 2단계: 세로(H) 방향 blending ──
    for i in range(1, len(rows)):
        for j in range(len(rows[i])):
            rows[i][j] = _blend_v(rows[i - 1][j], rows[i][j], blend_extent)

    # ── 3단계: 가로(W) 방향 blending + crop ──
    result_rows = []
    for i, row in enumerate(rows):
        blended_row = row[0]
        for j in range(1, len(row)):
            blended_row = _blend_h(blended_row, row[j], blend_extent)
            # 왼쪽 타일에서 row_limit 열만 사용 (blending 영역 이후 crop)
            keep_w = min(row_limit, blended_row.shape[2] - row[j].shape[2] + row_limit)
            if j < len(row) - 1:
                blended_row = mx.concatenate(
                    [blended_row[:, :, :row_limit, :], row[j][:, :, :, :]],
                    axis=2,
                )
                blended_row = _blend_h(blended_row[:, :, :-(blended_row.shape[2] - row_limit - row[j].shape[2] + blend_extent), :], row[j], blend_extent)

        # 행 blending: 위 행에서 row_limit 행만 사용
        if i < len(rows) - 1:
            result_rows.append(blended_row[:, :row_limit, :, :])
        else:
            result_rows.append(blended_row)

    result = mx.concatenate(result_rows, axis=1)
    mx.eval(result)
    return result


def vae_tiled_decode_simple(
    decoder,
    z: mx.array,
    tile_size: int = 64,
    overlap: int = 16,
    verbose: bool = False,
) -> mx.array:
    """
    VAE decoder tiled decoding.

    diffusers AutoencoderKL.tiled_decode 알고리즘을 MLX NHWC로 포팅.

    핵심 아이디어:
      - latent를 (tile_size × tile_size) 타일로 stride=tile_size-overlap 간격으로 분할
      - 각 타일 독립 디코딩 후 mx.eval() → 중간 텐서 즉시 해제
      - 인접 타일 경계를 linear blending으로 seamless 합성

    Args:
        decoder: VAEDecoder 인스턴스
        z: latent (B, H_lat, W_lat, C) — NHWC, 이미 scaling 적용된 값
        tile_size: latent 타일 크기 (기본 64 = 512px). 클수록 빠름, 메모리 증가
        overlap: latent 타일 간 겹침 (기본 16 = 128px). 클수록 seam 감소
        verbose: 진행 로그 출력

    Returns:
        decoded: (B, H_pix, W_pix, 3)
    """
    B, H_lat, W_lat, C = z.shape
    vae_scale = 8  # FLUX VAE: 8× spatial upsample
    blend_extent = overlap * vae_scale   # pixel 공간 blending 폭 (예: 16*8=128px)
    row_limit = tile_size * vae_scale - blend_extent  # 타일당 실제 사용 px (예: 512-128=384px)
    stride = tile_size - overlap         # 타일 간격 (예: 64-16=48 latent)

    # 타일 하나보다 작으면 그냥 통째로
    if H_lat <= tile_size and W_lat <= tile_size:
        return decoder(z)

    # 타일 시작 좌표: 마지막 타일은 경계에 맞춤
    # 한 축이 tile_size 이하면 타일 1개 (전체)
    def tile_starts(total: int) -> list[int]:
        if total <= tile_size:
            return [0]
        coords = list(range(0, total - tile_size, stride))
        if not coords or coords[-1] + tile_size < total:
            coords.append(total - tile_size)
        return coords

    h_starts = tile_starts(H_lat)
    w_starts = tile_starts(W_lat)
    n_h, n_w = len(h_starts), len(w_starts)

    if verbose:
        print(
            f"[vae_tiled_decode] latent={H_lat}×{W_lat} px={H_lat*vae_scale}×{W_lat*vae_scale} "
            f"tile={tile_size}(overlap={overlap}) → {n_h}×{n_w}={n_h*n_w}타일",
            flush=True,
        )

    # ── 1단계: 타일별 디코딩 → rows[i][j] = (B, h_px, w_px, 3) ──
    rows: list[list[mx.array]] = []
    for i, hi in enumerate(h_starts):
        tile_h = min(tile_size, H_lat - hi)  # 경계 타일은 실제 크기로
        row: list[mx.array] = []
        for j, wi in enumerate(w_starts):
            tile_w = min(tile_size, W_lat - wi)
            tile_z = z[:, hi : hi + tile_h, wi : wi + tile_w, :]
            decoded = decoder(tile_z)
            mx.eval(decoded)
            mx.metal.clear_cache()
            row.append(decoded)
            if verbose:
                print(f"  [{i},{j}] latent({hi}:{hi+tile_h},{wi}:{wi+tile_w}) → px={decoded.shape[1]}×{decoded.shape[2]}", flush=True)
        rows.append(row)

    # ── 2단계: blending + crop (diffusers 알고리즘) ──
    #
    # 핵심: result_cols[j]에는 blend 후 원본 타일 크기 그대로 보존.
    # concat 시 중간 타일은 row_limit px만, 마지막 타일은 전체.
    # 최종적으로 H_out × W_out으로 crop하여 정확한 크기 보장.
    result_rows: list[mx.array] = []
    for i in range(n_h):
        # blend된 타일을 원본 크기로 보관 (crop은 concat 단계에서)
        blended_cols: list[mx.array] = []
        for j in range(n_w):
            tile = rows[i][j]
            # (a) 위 타일과 세로 blending
            if i > 0:
                tile = _blend_v(rows[i - 1][j], tile, blend_extent)
            # (b) 왼쪽 blended 타일과 가로 blending (원본 크기 타일 사용)
            if j > 0:
                tile = _blend_h(blended_cols[j - 1], tile, blend_extent)
            blended_cols.append(tile)

        # 가로 방향: 중간 타일은 row_limit, 마지막은 전체 → concat → W crop
        row_parts: list[mx.array] = []
        for j in range(n_w):
            if j < n_w - 1:
                row_parts.append(blended_cols[j][:, :, :row_limit, :])
            else:
                row_parts.append(blended_cols[j])
        row_cat = mx.concatenate(row_parts, axis=2) if len(row_parts) > 1 else row_parts[0]

        # 세로 방향: 중간 row는 row_limit, 마지막은 전체 → concat → H crop
        if i < n_h - 1:
            result_rows.append(row_cat[:, :row_limit, :, :])
        else:
            result_rows.append(row_cat)

    result = mx.concatenate(result_rows, axis=1) if len(result_rows) > 1 else result_rows[0]

    # 타일 배치 오차를 정확히 제거: 원본 latent 해상도로 crop
    H_out, W_out = H_lat * vae_scale, W_lat * vae_scale
    result = result[:, :H_out, :W_out, :]

    mx.eval(result)
    return result


# ─────────────────────────────────────────────
# SSDFluxPipeline — FluxPipeline 대체
# ─────────────────────────────────────────────

def _make_ssd_flux_pipeline(
    sft_path: str | Path,
    model_version: str = "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
    config: MMDiTConfig | None = None,
    prefetch_depth: int = 2,
    use_mmap: bool = True,
    num_io_threads: int = 8,
    verbose: bool = True,
    vae_tile_size: int = 64,
    vae_overlap: int = 16,
):
    """
    SSD 스트리밍 FLUX.1 파이프라인 팩토리 함수.

    FluxPipeline을 서브클래싱해서:
      1. load_mmdit()를 오버라이드 → SSDStreamingMMDiT 로드
      2. load_mmdit(only_modulation_dict=True) → 빈 리스트 반환
         (CFGDenoiser.clear_cache()가 호출해도 no-op)

    이렇게 하면 diffusionkit의 sample_euler 루프가 그대로 동작함.
    """
    from diffusionkit.mlx import FluxPipeline

    _sft_path = Path(sft_path).expanduser()
    _ssd_config = config or FLUX_SCHNELL
    _tile_size = vae_tile_size
    _overlap = vae_overlap
    if "dev" in model_version.lower():
        from diffusionkit.mlx.config import FLUX_DEV
        _ssd_config = config or FLUX_DEV

    class _SSDFluxPipeline(FluxPipeline):
        def load_mmdit(self, only_modulation_dict=False):
            if only_modulation_dict:
                # CFGDenoiser.clear_cache()에서 호출: adaLN 가중치 복원 → 필요 없음
                # SSDStreamingMMDiT는 adaLN 가중치를 오프로드하지 않으므로 no-op
                return []
            # 처음 로드 시: SSDStreamingMMDiT 생성
            self.mmdit = SSDStreamingMMDiT(
                sft_path=_sft_path,
                config=_ssd_config,
                prefetch_depth=prefetch_depth,
                use_mmap=use_mmap,
                num_io_threads=num_io_threads,
                verbose=verbose,
            )

        def decode_latents_to_image(self, x_t):
            # x_t: latent_format.process_out() 이후 값 — VAEDecoder에 바로 전달 가능
            # tiled decode: 해상도 무관하게 peak ~100MB 고정
            x = vae_tiled_decode_simple(
                self.decoder,
                x_t,
                tile_size=_tile_size,
                overlap=_overlap,
                verbose=verbose,
            )
            return mx.clip(x / 2 + 0.5, 0, 1)

    if verbose:
        print(f"[SSDFluxPipeline] 초기화 중: {model_version}", flush=True)

    pipeline = _SSDFluxPipeline(
        model_version=model_version,
        low_memory_mode=True,
    )

    if verbose:
        print(f"[SSDFluxPipeline] 준비 완료!", flush=True)

    return pipeline


# 사용하기 편하도록 별칭 제공
class SSDFluxPipeline:
    """
    SSD 스트리밍 FLUX.1 파이프라인.

    FluxPipeline(low_memory_mode=False) 드롭인 대체:
      - MMDiT 블록 가중치: SSD on-demand 스트리밍
      - 텍스트 인코더(T5, CLIP) / VAE: diffusionkit 원본
      - RAM 목표: ~3~4GB (기존 ~24GB 대비 85%+ 절약)

    사용:
        pipeline = SSDFluxPipeline(
            sft_path="~/.cache/.../flux-schnell-4bit-quantized.safetensors",
            model_version="argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
        )
        images, log = pipeline.generate_image(prompt, ...)
    """

    def __new__(
        cls,
        sft_path: str | Path,
        model_version: str = "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
        config: MMDiTConfig | None = None,
        prefetch_depth: int = 2,
        use_mmap: bool = True,
        num_io_threads: int = 8,
        verbose: bool = True,
        vae_tile_size: int = 64,
        vae_overlap: int = 16,
    ):
        # 실제로는 FluxPipeline 서브클래스 인스턴스를 반환
        return _make_ssd_flux_pipeline(
            sft_path=sft_path,
            model_version=model_version,
            config=config,
            prefetch_depth=prefetch_depth,
            use_mmap=use_mmap,
            num_io_threads=num_io_threads,
            verbose=verbose,
            vae_tile_size=vae_tile_size,
            vae_overlap=vae_overlap,
        )
