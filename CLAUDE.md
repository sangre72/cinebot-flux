# flux-flash

FLUX.1 DiT SSD Block Streaming — ltx-flash 방식을 FLUX.1에 포팅.  
diffusionkit의 `MMDiT`을 SSD 블록 스트리밍 버전으로 교체해 RAM 사용량을 대폭 줄인다.

---

## 현재 상태

```
~/git/flux/
├── generate_worker.py   # subprocess 워커 — diffusionkit FluxPipeline 직접 사용
├── server.py            # HTTP 서버 (포트 18188) — 생성 후 프로세스 종료
├── start.sh             # 서버 실행 스크립트
└── out.png              # 테스트 출력 이미지
```

**현재 문제**: `FluxPipeline(low_memory_mode=False)`로 전체 모델을 RAM에 로드 → ~24GB RAM 필요.  
M4 Max 36GB에서는 돌아가지만 다른 앱과 동시 사용 시 메모리 압박이 심함.

**목표**: ltx-flash의 SSD 블록 스트리밍을 적용 → RAM ~3~4GB 수준으로 줄이기.

---

## FLUX.1 모델 구조 (diffusionkit 기준)

**safetensors 파일 경로:**
```
~/.cache/huggingface/hub/models--argmaxinc--mlx-FLUX.1-schnell-4bit-quantized/
  snapshots/.../flux-schnell-4bit-quantized.safetensors   (~6.6GB)
  snapshots/.../ae.safetensors                            (VAE)
```

**블록 구조:**

| 블록 타입 | safetensors 키 패턴 | 개수 | 블록당 텐서 수 | 블록당 크기(4bit) |
|---------|-------------------|-----|------------|----------------|
| Double Stream | `multimodal_transformer_blocks.{N}.*` | 19개 | 60개 | ~191 MB |
| Single Stream | `unified_transformer_blocks.{N}.*` | 38개 | 30개 | ~80 MB |

> 주의: 원본 FLUX.1의 `double_blocks` / `single_blocks` 키가 diffusionkit 로딩 시  
> `multimodal_transformer_blocks` / `unified_transformer_blocks`로 변환되어 저장됨.

**비블록 가중치 (30개 텐서, RAM 상주 대상):**
```
x_embedder.proj.*                         — 이미지 patch embedding (Conv2d)
context_embedder.*                         — T5 텍스트 임베딩 프로젝션
y_embedder.mlp.layers.{0,2}.*             — CLIP pooled text embedding MLP
t_embedder.mlp.layers.{0,2}.*             — timestep embedding MLP
final_layer.adaLN_modulation.layers.1.*   — 출력 adaLN
final_layer.linear.*                       — 출력 linear
```

**4bit 양자화 텐서 구조 (각 가중치마다 4개 텐서):**
```
*.weight  (U32)  — 실제 4bit 양자화 가중치
*.scales  (BF16) — 그룹별 스케일
*.biases  (BF16) — 그룹별 오프셋
*.bias    (BF16) — 레이어 바이어스 (별도)
```
→ 블록 로드 시 4개 텐서를 함께 읽어야 제대로 복원됨.

**블록 내부 구조:**
```
multimodal_transformer_blocks.N:        (Double Stream — 이미지+텍스트 공동 attention)
  image_transformer_block.adaLN_modulation.layers.1.*   — timestep AdaLN
  image_transformer_block.attn.{q,k,v,o}_proj.*         — self-attention
  image_transformer_block.mlp.{fc1,fc2}.*               — feed-forward
  image_transformer_block.qk_norm.{q,k}_norm.weight     — QK RMSNorm
  text_transformer_block.*                               — 위와 동일 구조

unified_transformer_blocks.N:           (Single Stream — 이미지+텍스트 concat)
  transformer_block.adaLN_modulation.layers.1.*
  transformer_block.attn.{q,k,v,o}_proj.*
  transformer_block.mlp.{fc1,fc2}.*
  transformer_block.qk_norm.{q,k}_norm.weight
```

**MMDiT forward 흐름:**
```
입력: latent_image (B,H,W,C), token_text (B,T,4096), timestep (B,)

1. x_embedder(latent_image)          → image_embeds  (B, H*W, 1, 3072)  [비블록]
2. context_embedder(token_text)      → text_embeds   (B, T, 3072)        [비블록]
3. y_embedder(pooled_text)           → timestep_cond (B, 1, 1, 3072)     [비블록]
4. t_embedder(timestep)              → timestep_cond +=                   [비블록]
5. pre_sdpa_rope(...)                → positional_encodings               [비블록, FLUX는 PreSDPARope]

  ↓ cache_modulation_params() 없이 실시간 계산 필요 (스트리밍 시)

6. for N in 0..18:                   multimodal_transformer_blocks[N]     [SSD 스트리밍]
     image_embeds, text_embeds = block(image_embeds, text_embeds, timestep_cond)

7. concat(text_embeds, image_embeds) → unified_embeds                    [비블록 연산]

8. for N in 0..37:                   unified_transformer_blocks[N]        [SSD 스트리밍]
     unified_embeds = block(unified_embeds, timestep_cond)

9. image_embeds = unified_embeds[:, T:, ...]                              [비블록]
10. final_layer(image_embeds, timestep_cond)  → output latent             [비블록]
```

---

## 구현 계획

### 1단계: `ssd_stream.py` 작성 (핵심)

ltx-flash의 `ssd_stream.py` 구조를 참고해 FLUX 전용으로 작성.

**`BlockIndex`**  
- `block_keys(block_type, block_idx)`: `"multimodal"` 또는 `"unified"` + 인덱스로 키 반환
- `non_block_keys()`: 블록 키 패턴 미포함 키 반환
- `open_mmap()` / `close_fd()`: 파일 디스크립터 관리

**`SSDBlockLoader`**  
- `load_block_weights(block_type, block_idx)`: 한 블록 텐서를 ThreadPoolExecutor로 병렬 로드
- `stream_blocks(block_type, num_blocks)`: N-ahead prefetch 슬라이딩 윈도우 제너레이터
- 블록 풀: `MultiModalTransformerBlock` 1개 + `UnifiedTransformerBlock` 1개 재사용

**`SSDStreamingMMDiT`** — `MMDiT` 드롭인 대체
- 비블록 가중치(30개)는 RAM 상주
- `cache_modulation_params()` 제거 (블록 없이 adaLN 계산 불가 → 실시간 계산)
- `__call__()`: MMDiT와 동일 인터페이스 유지

### 2단계: `FluxPipeline` 연결

diffusionkit의 `FluxPipeline`이 내부적으로 `MMDiT`을 생성하는 방식 파악 후,  
`pipeline.mmdit`을 `SSDStreamingMMDiT` 인스턴스로 교체하거나  
`FluxPipeline` 서브클래스를 작성.

### 3단계: `generate_worker.py` 수정

`FluxPipeline(low_memory_mode=False)` → `SSDFluxPipeline(...)` 교체.

---

## 참고: ltx-flash 핵심 설계 원칙

(ltx-flash에서 검증된 기법 — 그대로 적용)

1. **mmap 우선**: OS page cache 직접 참조 — warm 상태에서 I/O ≈ 0ms
2. **N-ahead prefetch**: 슬라이딩 윈도우, 다음 2개 블록 미리 읽기
3. **블록 객체 풀링**: 블록 1개 생성 후 재사용, 가중치만 `load_weights()` 교체
4. **즉시 해제**: `del weights; mx.clear_cache()` — Metal GPU 메모리도 명시적 해제
5. **Metal 스레드 안전**: `mx.eval()`은 반드시 메인 스레드에서 호출
6. **mmap 멀티스레드 주의**: `bytes(mm[start:end])`로 복사 후 사용

---

## 주의사항

- **`cache_modulation_params()` 충돌**: diffusionkit `MMDiT`은 모든 timestep의 adaLN 파라미터를  
  추론 시작 전 미리 계산해두는 최적화가 있음. 블록을 스트리밍하면 블록 가중치가 없어  
  이 함수 사용 불가 → 제거하고 블록 forward 내에서 실시간 계산해야 함.
- **adaLN 가중치 위치**: `adaLN_modulation.layers.1.weight`는 블록 내부에 있음 (블록과 함께 스트리밍).  
  `layers.0`은 SiLU activation — 파라미터 없음.
- **4bit 역양자화**: diffusionkit이 내부적으로 처리하는지 확인 필요.  
  `load_weights()`에 `(weight, scales, biases)` 세트를 넘기면 자동 복원될 가능성 높음.
- **`guidance_in`**: FLUX-dev는 `guidance_embed=True` — `guidance_in` 레이어 추가됨.  
  schnell은 없음. 비블록에 포함.
- **`pre_sdpa_rope`**: FLUX는 `PreSDPARope` 방식 — `RoPE` 모듈이 비블록 가중치 없는 순수 계산.

---

## 환경

```bash
# Python 인터프리터
/Users/bumsuklee/miniconda3/envs/diffusionkit/bin/python3.10

# diffusionkit 설치 위치
/Users/bumsuklee/miniconda3/lib/python3.10/site-packages/diffusionkit/

# 참고할 핵심 파일
diffusionkit/mlx/mmdit.py      — MMDiT, MultiModalTransformerBlock, UnifiedTransformerBlock
diffusionkit/mlx/config.py     — FLUX_SCHNELL, FLUX_DEV MMDiTConfig
diffusionkit/mlx/model_io.py   — flux_state_dict_adjustments (키 변환 로직)
diffusionkit/mlx/__init__.py   — FluxPipeline

# 참고할 ltx-flash 구현
~/git/ltx-flash/ltx_flash/ssd_stream.py
```

---

## 작업 순서 체크리스트

- [ ] diffusionkit `FluxPipeline` 내부에서 `MMDiT` 생성/교체 포인트 파악
- [ ] `SSDFluxBlockIndex` 구현 (키 패턴: `multimodal_transformer_blocks.N.*`, `unified_transformer_blocks.N.*`)
- [ ] `SSDFluxBlockLoader` 구현 (4bit 텐서 4종 함께 로드)
- [ ] `SSDStreamingMMDiT` 구현 (`MMDiT.__call__`과 동일 인터페이스)
- [ ] `cache_modulation_params()` 없이 동작 검증
- [ ] `FluxPipeline`에 스트리밍 모델 주입
- [ ] `generate_worker.py` 수정 및 동작 테스트
- [ ] RAM 사용량 측정 (목표: ~4GB 이하)
