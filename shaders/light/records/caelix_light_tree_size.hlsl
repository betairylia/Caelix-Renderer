// Light tree group pass, kernel 1 of 4 (docs/LIGHT_TREE.md section 3): the size of every group's
// light block after the pass, into offsets[group]. One workgroup per group index.
// First the CPU's staged flags are merged: cpu_flags = (cpu_flags & fresh) | staged when
// staged != 0 (a later state replaces rebuild / released but keeps a pending fresh), and the staged
// word is cleared. Then:
//   released (kLtCpuReleased)                -> 0
//   flagged (kLtCpuRebuild or gpu_flags)     -> the size of the rebuilt block: the light brick scan
//                                               and the per-brick plan (recompute or copy the brick
//                                               light record) shared with the build kernel
//   otherwise                                -> the current block_units (the block is copied)
// A released or flagged group sets kLtPassFlagChanged in the control's pass_flags (without one, the
// pass changes nothing: no copy, no heap flip) and the flag of the BVH of its entity's motion (only
// those BVHs are gathered and rebuilt after a committed pass).
#include "light/records/caelix_light_tree_group.hlsl"

static const uint kModeKeep = 0;
static const uint kModeRebuild = 1;
static const uint kModeReleased = 2;

groupshared uint gs_mode;
groupshared uint gs_fresh;
groupshared uint2 gs_block;  // offset, units
groupshared uint gs_slot_base;
groupshared uint gs_high_water;
groupshared uint gs_entity;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    const uint group = g_push.group_base + group_id.x;
    if (group >= g_push.group_count)
    {
        return;  // uniform for the whole workgroup
    }
    const uint t = thread_id.x;
    const uint64_t descriptor = g_push.descriptor;
    const uint shift = LtDescriptorU32(descriptor, kLtDescGroupShift);
    const SceneLookup lookup = LtLookup(shift);
    const uint64_t row = LtDescriptorU64(descriptor, kLtDescGroupRows) + uint64_t(group) * kLtGroupRowBytes;
    if (t == 0)
    {
        const uint4 block = CAELIX_LOAD_U32x4(row);  // block_offset, block_units, generation, gpu_flags
        uint cpu_flags = CAELIX_LOAD_U32(row + kLtRowCpuFlags);
        const uint64_t staged_address = g_push.staged_flags + uint64_t(group) * 4;
        const uint staged = CAELIX_LOAD_U32(staged_address);
        if (staged != 0)
        {
            cpu_flags = (cpu_flags & kLtCpuFresh) | staged;
            CAELIX_STORE_U32(row + kLtRowCpuFlags, cpu_flags);
            CAELIX_STORE_U32(staged_address, 0u);
        }
        gs_mode = (cpu_flags & kLtCpuReleased) != 0                                         ? kModeReleased
                  : ((cpu_flags & kLtCpuRebuild) != 0 || (block.w & kLtGpuLightChanged) != 0) ? kModeRebuild
                                                                                             : kModeKeep;
        gs_fresh = cpu_flags & kLtCpuFresh;
        gs_block = block.xy;
        const uint64_t info = lookup.group_infos + uint64_t(group) * kGroupInfoBytes;
        const uint4 bases = CAELIX_LOAD_U32x4(info + kGroupInfoSlotRowBase);
        gs_slot_base = bases.x;
        gs_high_water = bases.w;
        gs_entity = CAELIX_LOAD_U32(info + kGroupInfoEntity);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint mode = gs_mode;

    uint units = mode == kModeKeep ? gs_block.y : 0u;
    if (mode == kModeRebuild)
    {
        const LtGroupScan scan = LtScanGroup(lookup, gs_slot_base, gs_high_water, t);
        const uint64_t control = LtDescriptorU64(descriptor, kLtDescControl);
        const uint current = CAELIX_LOAD_U32(control + kLtCtlCurrent) & 1u;
        const uint64_t heap = LtDescriptorU64(descriptor, current != 0 ? kLtDescHeap1 : kLtDescHeap0);
        const LtOldBlock old = LtOpenOldBlock(heap, gs_block.x, gs_block.y, gs_fresh != 0);
        uint record_units = 0;
        for (uint id = t; id < gs_high_water; id += kLtThreads)
        {
            uint pos;
            uint64_t record;
            const uint word1 = LtBrickWord1(lookup, gs_slot_base, id, pos, record);
            const uint m = LtMorton(BrickOfPos(pos, shift));
            if (LtLightFaceCount(word1) != 0 && LtBitSet(m))
            {
                record_units += LtPlanBrick(word1, m, old, shift).units;
            }
        }
        uint total;
        WorkgroupExclusiveSum(record_units, t, kLtThreads, total);
        units = LtBlockUnits(scan.bricks, total);
    }
    if (t == 0)
    {
        CAELIX_STORE_U32(g_push.offsets + uint64_t(group) * 4, units);
        if (mode != kModeKeep)
        {
            // The tree of the group's entity must be gathered again (a released group: its last entity).
            const uint64_t motions = LtDescriptorU64(descriptor, kLtDescEntityMotion);
            const uint motion = CAELIX_LOAD_U32(motions + uint64_t(gs_entity) * 4) & 1u;
            CAELIX_ATOMIC_OR_U32(LtDescriptorU64(descriptor, kLtDescControl) + kLtCtlPassFlags,
                                 kLtPassFlagChanged | (1u << (kLtPassFlagTreeShift + motion)));
        }
    }
}
