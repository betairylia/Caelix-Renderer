// Light BVH gather, kernel 1 of 2 (docs/LIGHT_TREE.md section 3): one thread per group index writes
// the number of leaves the group gives to this tree: its block's record count when the group's
// entity has this tree's motion, else 0 (no block, or a release still pending after a failed pass).
// caelix_light_bvh_scan turns the counts into leaf offsets.
#include "scene/caelix_light_bvh_common.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint group = g_push.group_base + dispatch_id.x;
    if (group >= g_push.group_count || !LtBuilding())
    {
        return;
    }
    const uint64_t descriptor = g_push.descriptor;
    const uint64_t row_address =
        LtDescriptorU64(descriptor, kLtDescGroupRows) + uint64_t(group) * kLtGroupRowBytes;
    const uint4 row = CAELIX_LOAD_U32x4(row_address);  // block_offset, block_units, generation, gpu_flags
    const uint cpu_flags = CAELIX_LOAD_U32(row_address + kLtRowCpuFlags);
    uint count = 0;
    if (row.y != 0 && (cpu_flags & kLtCpuReleased) == 0)
    {
        const uint entity = CAELIX_LOAD_U32(LtDescriptorU64(descriptor, kLtDescGroupInfos) +
                                            uint64_t(group) * kGroupInfoBytes + kGroupInfoEntity);
        const uint64_t motions = LtDescriptorU64(descriptor, kLtDescEntityMotion);
        const uint motion = CAELIX_LOAD_U32(motions + uint64_t(entity) * 4) & 1u;
        if (motion == g_push.tree)
        {
            const uint current = CAELIX_LOAD_U32(LtControl() + kLtCtlCurrent) & 1u;
            const uint64_t heap = LtDescriptorU64(descriptor, current != 0 ? kLtDescHeap1 : kLtDescHeap0);
            count = CAELIX_LOAD_U32(heap + uint64_t(row.x) * kLtUnitBytes);  // LightBlockHeader::record_count
        }
    }
    CAELIX_STORE_U32(g_push.offsets + uint64_t(group) * 4, count);
}
