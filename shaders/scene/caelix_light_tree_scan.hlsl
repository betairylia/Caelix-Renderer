// Light tree group pass, kernel 2 of 4 (docs/LIGHT_TREE.md section 3): one workgroup turns the
// block sizes in offsets[0, group_count) into their exclusive prefix sum (the offsets in the target
// heap buffer) and decides the pass (a pass with changes also resets the record counters):
//   no released or flagged group -> kLtPassNoChange (the build returns at once, no flip)
//   total <= capacity_units      -> kLtPassOk
//   otherwise                    -> kLtPassFailed (nothing changes on the GPU; the flags stay)
#include "scene/caelix_light_tree_group.hlsl"

groupshared uint gs_flags;

[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_GroupThreadID)
{
    const uint t = thread_id.x;
    const uint64_t control = LtDescriptorU64(g_push.descriptor, kLtDescControl);
    if (t == 0)
    {
        gs_flags = CAELIX_LOAD_U32(control + kLtCtlPassFlags);
    }
    // Chunks of 1024 sizes, 4 per thread; the carry is 64-bit so that an overflow fails the pass.
    uint64_t carry = 0;
    for (uint base = 0; base < g_push.group_count; base += 4 * kLtThreads)
    {
        uint sizes[4];
        CAELIX_UNROLL
        for (uint i = 0; i < 4; ++i)
        {
            const uint g = base + 4 * t + i;
            sizes[i] = g < g_push.group_count ? CAELIX_LOAD_U32(g_push.offsets + uint64_t(g) * 4) : 0u;
        }
        uint total;
        WorkgroupExclusiveSum4(sizes, t, kLtThreads, total);
        CAELIX_UNROLL
        for (uint j = 0; j < 4; ++j)
        {
            const uint g = base + 4 * t + j;
            if (g < g_push.group_count)
            {
                const uint64_t offset = carry + sizes[j];
                CAELIX_STORE_U32(g_push.offsets + uint64_t(g) * 4, uint(min(offset, uint64_t(0xFFFFFFFFu))));
            }
        }
        carry += total;
    }
    if (t == 0)
    {
        const uint changed = gs_flags & kLtPassFlagChanged;
        const uint needed = uint(min(carry, uint64_t(0xFFFFFFFFu)));
        const bool fits = carry <= uint64_t(g_push.capacity_units);
        const uint result = changed == 0 ? kLtPassNoChange : (fits ? kLtPassOk : kLtPassFailed);
        CAELIX_STORE_U32(control + kLtCtlPassResult, result);
        if (changed != 0)
        {
            // A pass with changes: its size and the build kernel's record counters.
            CAELIX_STORE_U32(control + kLtCtlNeededUnits, needed);
            CAELIX_STORE_U32(control + kLtCtlRecordsRecomputed, 0u);
            CAELIX_STORE_U32(control + kLtCtlRecordsCopied, 0u);
        }
        const uint trees = (gs_flags >> kLtPassFlagTreeShift) & ((1u << kLtTrees) - 1u);
        CAELIX_STORE_U32(control + kLtCtlTreesChanged, trees);
        CAELIX_STORE_U32(control + kLtCtlPassFlags, 0u);
    }
}
