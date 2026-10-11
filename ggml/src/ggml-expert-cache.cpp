#include "ggml-expert-cache.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * ExpertCache — persistent VRAM arena for hot MoE experts.
 *
 * Per-layer arena (models are NOT uniform: weight shapes can vary by layer):
 *   lay_off[l] .. lay_off[l+1]
 *     = [up region][slack][gate region][slack][down region][slack]
 *   region = n_expert contiguous per-expert blobs of stride *_es_l[l]
 * The MoE kernel reads an expert at (region_base + expert*stride). Resident
 * experts are uploaded once and reused on later steps. Residency is tracked
 * per (layer, kind, expert). See header for why scheduler staging was dropped.
 *
 * SLACK: mmq_id/mul_mat_q clear row padding at (region_base + nbytes) via
 * cudaMemsetAsync; without a gap that write lands in the next region (or past
 * the arena) and faults. XCA_SLACK covers MATRIX_ROW_PADDING plus alignment.
 */
#define XCA_SLACK 8192
#define XCA_ALIGN 256

static int64_t xca_region_bytes(int64_t n_expert, int64_t es) {
    if (es <= 0) return 0;
    int64_t b = n_expert * es;
    b = (b + (XCA_ALIGN - 1)) & ~(int64_t)(XCA_ALIGN - 1);
    return b + XCA_SLACK;
}

int64_t ggml_expert_cache_layer_bytes(int64_t n_expert, int64_t up_es, int64_t gate_es, int64_t down_es) {
    return xca_region_bytes(n_expert, up_es) +
           xca_region_bytes(n_expert, gate_es) +
           xca_region_bytes(n_expert, down_es);
}

int ggml_expert_cache_open(
    ggml_expert_cache_t * cache,
    void * vram_base,
    size_t vram_bytes,
    int32_t n_layers,
    int64_t n_expert,
    const int64_t * up_es,
    const int64_t * gate_es,
    const int64_t * down_es
) {
    if (!cache || !vram_base || vram_bytes == 0) return -1;
    if (n_layers <= 0 || n_expert <= 0) return -1;
    if (!up_es || !gate_es || !down_es) return -1;

    memset(cache, 0, sizeof(*cache));
    cache->base     = vram_base;
    cache->size     = vram_bytes;
    cache->n_expert = n_expert;

    cache->lay_off   = (int64_t *)calloc((size_t)n_layers + 1, sizeof(int64_t));
    cache->up_es_l   = (int64_t *)calloc((size_t)n_layers, sizeof(int64_t));
    cache->gate_es_l = (int64_t *)calloc((size_t)n_layers, sizeof(int64_t));
    cache->down_es_l = (int64_t *)calloc((size_t)n_layers, sizeof(int64_t));
    if (!cache->lay_off || !cache->up_es_l || !cache->gate_es_l || !cache->down_es_l) {
        ggml_expert_cache_close(cache);
        return -1;
    }

    /* Keep whole layers that fit within the arena. */
    int32_t kept = 0;
    int64_t off = 0;
    for (int32_t l = 0; l < n_layers; ++l) {
        const int64_t lb = ggml_expert_cache_layer_bytes(n_expert, up_es[l], gate_es[l], down_es[l]);
        if (lb <= 0) { /* dense layer: no arena space, still addressable as empty */
            cache->up_es_l[l] = cache->gate_es_l[l] = cache->down_es_l[l] = 0;
            cache->lay_off[l] = off;
            kept = l + 1;
            continue;
        }
        if ((size_t)(off + lb) > vram_bytes) break;
        cache->up_es_l[l]   = up_es[l];
        cache->gate_es_l[l] = gate_es[l];
        cache->down_es_l[l] = down_es[l];
        cache->lay_off[l]   = off;
        off += lb;
        kept = l + 1;
    }
    cache->lay_off[kept] = off;
    cache->n_layers = kept;
    if (kept <= 0) { ggml_expert_cache_close(cache); return -1; }

    const size_t res_bytes = (size_t)kept * 3 * (size_t)n_expert;
    cache->resident = (uint8_t *)calloc(res_bytes, 1);
    if (!cache->resident) { ggml_expert_cache_close(cache); return -1; }

    return 0;
}

void ggml_expert_cache_close(ggml_expert_cache_t * cache) {
    if (!cache) return;
    free(cache->lay_off);
    free(cache->up_es_l);
    free(cache->gate_es_l);
    free(cache->down_es_l);
    free(cache->resident);
    memset(cache, 0, sizeof(*cache));
}

int ggml_expert_cache_layer_cached(const ggml_expert_cache_t * cache, int64_t layer) {
    return cache && layer >= 0 && layer < cache->n_layers;
}

int64_t ggml_expert_cache_kind_stride(const ggml_expert_cache_t * cache, int64_t layer, int kind) {
    if (!cache || !ggml_expert_cache_layer_cached(cache, layer)) return 0;
    switch (kind) {
        case GGML_EXPERT_KIND_UP:   return cache->up_es_l[layer];
        case GGML_EXPERT_KIND_GATE: return cache->gate_es_l[layer];
        case GGML_EXPERT_KIND_DOWN: return cache->down_es_l[layer];
        default: return 0;
    }
}

void * ggml_expert_cache_kind_base(const ggml_expert_cache_t * cache, int64_t layer, int kind) {
    if (!cache || !ggml_expert_cache_layer_cached(cache, layer)) return NULL;
    if (kind < 0 || kind > 2) return NULL;
    const int64_t stride = ggml_expert_cache_kind_stride(cache, layer, kind);
    if (stride <= 0) return NULL;
    int64_t off = cache->lay_off[layer];
    if (kind == GGML_EXPERT_KIND_GATE) {
        off += xca_region_bytes(cache->n_expert, cache->up_es_l[layer]);
    } else if (kind == GGML_EXPERT_KIND_DOWN) {
        off += xca_region_bytes(cache->n_expert, cache->up_es_l[layer]) +
               xca_region_bytes(cache->n_expert, cache->gate_es_l[layer]);
    }
    return (uint8_t *)cache->base + off;
}

int ggml_expert_cache_resident(const ggml_expert_cache_t * cache, int64_t layer, int kind, int64_t expert) {
    if (!cache || !cache->resident) return 0;
    if (!ggml_expert_cache_layer_cached(cache, layer)) return 0;
    if (kind < 0 || kind > 2) return 0;
    if (expert < 0 || expert >= cache->n_expert) return 0;
    return cache->resident[((size_t)layer * 3 + kind) * (size_t)cache->n_expert + (size_t)expert] != 0;
}

void ggml_expert_cache_mark(ggml_expert_cache_t * cache, int64_t layer, int kind, int64_t expert) {
    if (!cache || !cache->resident) return;
    if (!ggml_expert_cache_layer_cached(cache, layer)) return;
    if (kind < 0 || kind > 2) return;
    if (expert < 0 || expert >= cache->n_expert) return;
    cache->resident[((size_t)layer * 3 + kind) * (size_t)cache->n_expert + (size_t)expert] = 1;
}

void ggml_expert_cache_note_hit(ggml_expert_cache_t * cache)  { if (cache) cache->hits++; }
void ggml_expert_cache_note_miss(ggml_expert_cache_t * cache) { if (cache) cache->misses++; }

void ggml_expert_cache_dump(const ggml_expert_cache_t * cache) {
    if (!cache) return;
    const uint64_t tot = cache->hits + cache->misses;
    fprintf(stderr, "xca: hits=%llu misses=%llu (%.1f%% hit) layers=%d experts=%lld arena=%.2fGiB\n",
            (unsigned long long)cache->hits,
            (unsigned long long)cache->misses,
            tot ? 100.0 * (double)cache->hits / (double)tot : 0.0,
            cache->n_layers,
            (long long)cache->n_expert,
            (double)cache->size / 1073741824.0);
}
