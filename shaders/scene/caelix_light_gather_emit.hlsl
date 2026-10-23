// Light BVH gather, kernel 2 of 2 (docs/LIGHT_TREE.md section 3): one thread per group index writes
// the leaves of its light bricks at the group's offset (leaves at or above the leaf capacity are
// dropped and counted) and their leaf index into the bvh_leaf word of the index entries in the
// CURRENT heap buffer. A leaf is the brick light record's box and face sums moved to world space:
// the box is the AABB of the rotated local box, the world face sums project the local ones,
//   W[+a] = sum_f F[f] * max(0, (R d_f)_a),  W[-a] = sum_f F[f] * max(0, -(R d_f)_a)
// (an upper bound of the cosine-weighted power in every direction; exact for axis-aligned
// rotations).
#include "scene/caelix_light_bvh_common.hlsl"

LtSummary LtWorldSummary(LtSummary local, float3x3 rotation, float3 translation)
{
    const float3 centre = 0.5f * (local.lo + local.hi);
    const float3 half_size = 0.5f * (local.hi - local.lo);
    const float3 world_centre = mul(rotation, centre) + translation;
    const float3 world_half = mul(abs(rotation), half_size);
    LtSummary world = LtEmptySummary();
    world.lo = world_centre - world_half;
    world.hi = world_centre + world_half;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        const uint axis = f >> 1;
        const float sign = (f & 1) != 0 ? -1.0f : 1.0f;
        const float3 d = sign * float3(rotation[0][axis], rotation[1][axis], rotation[2][axis]);
        CAELIX_UNROLL
        for (uint b = 0; b < 3; ++b)
        {
            world.face[2 * b] += local.face[f] * max(0.0f, d[b]);
            world.face[2 * b + 1] += local.face[f] * max(0.0f, -d[b]);
        }
    }
    return world;
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint group = g_push.group_base + dispatch_id.x;
    if (group >= g_push.group_count || !LtBuilding())
    {
        return;
    }
    // The header's leaf_count and the dropped count are written by caelix_light_bvh_bounds.
    const uint found = CAELIX_LOAD_U32(LtTreeState() + kLtStLeavesFound);
    const uint leaf_count = min(found, g_push.leaf_capacity);
    const uint first = CAELIX_LOAD_U32(g_push.offsets + uint64_t(group) * 4);
    const uint next =
        group + 1 < g_push.group_count ? CAELIX_LOAD_U32(g_push.offsets + uint64_t(group + 1) * 4) : found;
    if (next == first)
    {
        return;
    }
    const uint64_t descriptor = g_push.descriptor;
    // block_offset, block_units, generation, gpu_flags
    const uint4 row =
        CAELIX_LOAD_U32x4(LtDescriptorU64(descriptor, kLtDescGroupRows) + uint64_t(group) * kLtGroupRowBytes);
    const uint current = CAELIX_LOAD_U32(LtControl() + kLtCtlCurrent) & 1u;
    const uint64_t heap = LtDescriptorU64(descriptor, current != 0 ? kLtDescHeap1 : kLtDescHeap0);
    const uint64_t block = heap + uint64_t(row.x) * kLtUnitBytes;
    uint entity;
    const float3 origin = LtGroupOrigin(LtDescriptorU64(descriptor, kLtDescGroupInfos), group,
                                        LtDescriptorU32(descriptor, kLtDescGroupShift), entity);
    const LtFrame frame = LtEntityFrame(LtDescriptorU64(descriptor, kLtDescEntityInfos), entity);
    const uint64_t leaves = LtLeaves();
    for (uint k = 0; k < next - first; ++k)
    {
        const uint leaf = first + k;
        const uint64_t index_entry = block + kLtBlockIndexOffset + uint64_t(k) * kLtIndexEntryBytes;
        if (leaf >= leaf_count)
        {
            CAELIX_STORE_U32(index_entry + kLtIndexBvhLeaf, kLtNone);
            continue;
        }
        const LtRecord record = LtLoadEntryRecord(block, k);
        LtSummary local;
        local.face = record.face;
        local.lo = origin + float3(record.lo);
        local.hi = origin + float3(record.hi);
        const uint64_t address = leaves + uint64_t(leaf) * kLtLeafBytes;
        LtStoreSummary(address, LtWorldSummary(local, frame.rotation, frame.translation));
        // group, entry, pad, parent (written by the collapse)
        CAELIX_STORE_U32x4(address + kLtLeafGroup, uint4(group, k, 0u, kLtNone));
        CAELIX_STORE_U32(index_entry + kLtIndexBvhLeaf, (g_push.tree << kLtBvhLeafTreeShift) | leaf);
    }
}
