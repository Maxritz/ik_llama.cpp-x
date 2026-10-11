#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ExpertCache — persistent VRAM arena for hot MoE experts (>VRAM / >RAM models).
 *
 * Design (revised): the cache OWNS a dedicated VRAM arena, laid out as
 *   [layer0: up_experts | gate_experts | down_experts][layer1: ...] ...
 * Each region is full-width (n_expert slots of the weight's per-expert stride),
 * so the existing MoE kernels can read an expert at (arena_region + expert*stride)
 * with no kernel changes. Residency is tracked per (layer, kind, expert).
 *
 * Why the previous design failed: it reused the scheduler's shared staging buffer
 * and skipped re-copies on "hit". That buffer is aliased/reused across layers, so
 * a hit read another layer's bytes -> NaN logits -> sampler crash. A dedicated,
 * non-aliased, persistent arena fixes correctness and gives real reuse.
 *
 * Papers: SeqMoE (2609.12978) 96.97% hit @ 45% residency; Mira (2609.38090) HOT
 * eviction; Fate (2502.12224) predictive prefetch; Expert Reduction->Divergence
 * (2607.28097) determinism.
 */

#define GGML_EXPERT_KIND_UP   0
#define GGML_EXPERT_KIND_GATE 1
#define GGML_EXPERT_KIND_DOWN 2

typedef struct ggml_expert_cache {
    void *    base;       /* device arena base */
    size_t    size;       /* arena bytes */

    int32_t   n_layers;   /* number of cached layers (layers [0, n_layers) ) */
    int64_t   n_expert;   /* experts per layer */

    /* Per-layer geometry (models may vary weight shapes between layers). */
    int64_t * lay_off;    /* [n_layers+1] byte offset of each layer arena */
    int64_t * up_es_l;    /* [n_layers] bytes/expert for up   (0 = none) */
    int64_t * gate_es_l;  /* [n_layers] bytes/expert for gate (0 = none) */
    int64_t * down_es_l;  /* [n_layers] bytes/expert for down (0 = none) */

    uint8_t * resident;   /* [(layer*3 + kind)*n_expert], 1 = expert bytes present */

    uint64_t  hits;
    uint64_t  misses;
    uint64_t  disabled;   /* access outside cached layers */
} ggml_expert_cache_t;

/*
 * Open the cache over a caller-allocated device arena (ggml_backend_buffer).
 *   n_layers : number of leading model layers to cache
 *   up_es/gate_es/down_es : per-layer per-expert byte stride arrays (length n_layers,
 *                           entries may be 0 for dense/non-MoE layers)
 * Returns 0 on success, -1 on error.
 */
int ggml_expert_cache_open(
    ggml_expert_cache_t * cache,
    void * vram_base,
    size_t vram_bytes,
    int32_t n_layers,
    int64_t n_expert,
    const int64_t * up_es,
    const int64_t * gate_es,
    const int64_t * down_es
);

void ggml_expert_cache_close(ggml_expert_cache_t * cache);

/* Padded byte size of one (layer) arena given its weight strides, including the
 * per-region slack the kernels need (they memset row padding past each region). */
int64_t ggml_expert_cache_layer_bytes(int64_t n_expert, int64_t up_es, int64_t gate_es, int64_t down_es);

/* True if `layer` is covered by the arena. */
int ggml_expert_cache_layer_cached(const ggml_expert_cache_t * cache, int64_t layer);

/* Device base of a (layer, kind) full-width region; NULL if not cached. */
void * ggml_expert_cache_kind_base(const ggml_expert_cache_t * cache, int64_t layer, int kind);

/* Per-expert byte stride for a kind in a layer (0 if none). */
int64_t ggml_expert_cache_kind_stride(const ggml_expert_cache_t * cache, int64_t layer, int kind);

/* Residency query / update. */
int  ggml_expert_cache_resident(const ggml_expert_cache_t * cache, int64_t layer, int kind, int64_t expert);
void ggml_expert_cache_mark(ggml_expert_cache_t * cache, int64_t layer, int kind, int64_t expert);

/* Stats. */
void ggml_expert_cache_note_hit(ggml_expert_cache_t * cache);
void ggml_expert_cache_note_miss(ggml_expert_cache_t * cache);
void ggml_expert_cache_dump(const ggml_expert_cache_t * cache);

#ifdef __cplusplus
}
#endif
