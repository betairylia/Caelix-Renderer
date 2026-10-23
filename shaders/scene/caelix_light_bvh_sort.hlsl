// Light BVH: the shared part of the stable LSD radix sort of (key, value) pairs, 4 bits per pass
// (8 passes for 32-bit keys). One pass = caelix_light_bvh_sort_histogram (16 digit counts per block
// of 1024 elements, stored digit-major), caelix_light_bvh_scan over the 16 * block_count counts, and
// caelix_light_bvh_sort_scatter (every block writes its elements to the scanned digit offsets,
// keeping the input order among equal digits). The element count is read from count_address; a
// non-zero gate_address whose u32 is 0 skips the pass.
#ifndef CAELIX_LIGHT_BVH_SORT_HLSL
#define CAELIX_LIGHT_BVH_SORT_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_light_tree_layout.hlsl"

// Push constants of both sort kernels (LightSortPush). 64 bytes.
struct LightSortPush
{
    uint64_t keys_in;
    uint64_t values_in;
    uint64_t keys_out;
    uint64_t values_out;
    uint64_t histogram;  // u32[kLtSortDigits * block_count], digit-major
    uint64_t count_address;
    uint64_t gate_address;
    uint shift;  // digit = (key >> shift) & kLtSortDigitMask
    uint block_count;
};

CAELIX_PUSH_CONSTANTS(LightSortPush, g_push);

static const uint kLtSortThreads = 256;
static const uint kLtSortDigitMask = kLtSortDigits - 1;
static const uint kLtSortMaxWaves = kLtSortThreads / 4;  // at the smallest wave size

bool LtSortSkipped()
{
    return g_push.gate_address != 0 && CAELIX_LOAD_U32(g_push.gate_address) == 0;
}

#endif  // CAELIX_LIGHT_BVH_SORT_HLSL
