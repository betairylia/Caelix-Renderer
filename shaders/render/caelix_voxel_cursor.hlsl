// The voxel cursor of the render kernels: a voxel of an entity's grid that moves voxel by voxel
// through the scene's record lookups (scene/caelix_scene_lookup.hlsl), and the entity transform
// helpers by entity id. Data layer: render/caelix_scene_trace.hlsl includes this file after
// TraceResult; the integrator above the boundary uses the cursor only through the functions below
// (the transmissive shadow ray's straight SSS walk, TraceShadowTransmissive; the subsurface walk of
// docs/SUBSURFACE.md).
//
// All coordinates are entity-local block coordinates (object space of the entity's instances; the
// entity transforms are rigid, scale 1). A brick is the 8^3 block cube at voxel >> 3; an absent brick
// (no resident record) is air.
#ifndef CAELIX_VOXEL_CURSOR_HLSL
#define CAELIX_VOXEL_CURSOR_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "render/caelix_brick_trace.hlsl"
#include "scene/caelix_scene_lookup.hlsl"

// The scene tables of the frame as the record lookups of scene/caelix_scene_lookup.hlsl take them.
SceneLookup FrameSceneLookup(CaelixFrameConstants fc)
{
    SceneLookup s;
    s.page_table = fc.page_table;
    s.slot_tables = fc.slot_tables;
    s.reverse_tables = fc.reverse_tables;
    s.group_infos = fc.group_infos;
    s.entity_infos = fc.entity_infos;
    s.group_grids = fc.group_grids;
    s.records_per_page_shift = fc.records_per_page_shift;
    s.record_bytes = fc.record_bytes;
    s.group_shift = fc.group_shift;
    return s;
}

// A voxel of an entity's grid: its entity-local coordinate, the record of its brick and the brick's
// group (the brick coordinate, voxel >> 3, and the voxel in the brick, voxel & 7, are derived where
// needed: fewer live registers in a walk).
struct VoxelCursor
{
    uint group;       // the scene-wide group index of the brick (unspecified in an absent brick)
    uint entity;
    uint64_t record;  // 0 = absent brick (every voxel air)
    int3 voxel;       // entity-local voxel coordinate
};

// The voxel of the hit `hit` (the voxel whose block the DDA reported: the brick origin, as
// TraceCandidate derives it, plus hit.cell; exact, no floor of object_hit), in the brick that the
// hit's candidate traced (it is resident).
VoxelCursor VoxelCursorFromHit(CaelixFrameConstants fc, TraceResult hit)
{
    const uint64_t info = fc.group_infos + uint64_t(hit.group) * kGroupInfoBytes;
    const uint4 w0 = CAELIX_LOAD_U32x4(info);  // coord, entity (kGroupInfoCoord = 0)
    const uint slot_row_base = CAELIX_LOAD_U32(info + kGroupInfoSlotRowBase);
    const uint4 row = CAELIX_LOAD_U32x4(fc.slot_tables + (uint64_t(slot_row_base) + hit.id) * kSlotRowBytes);
    VoxelCursor c;
    c.group = hit.group;
    c.entity = w0.w;  // kGroupInfoEntity
    c.record = SceneRecordAddress(FrameSceneLookup(fc), row.z);
    // Entity-local block coordinate of the brick's corner (as in TraceCandidate).
    const uint shift = fc.group_shift;
    const int mask = int((1u << shift) - 1u);
    const uint pos = SlotRowPos(row.x);
    const int3 brick = int3(int(pos) & mask, int(pos >> shift) & mask, int(pos >> (2 * shift)) & mask);
    const int3 origin = (int3(w0.xyz) * int(1u << shift) + brick) * int(kBrickEdge);
    const uint cell = hit.cell;
    c.voxel = origin + int3(int(cell & 7u), int((cell >> 3) & 7u), int((cell >> 6) & 7u));
    return c;
}

// Moves the cursor by `delta` voxels (each component -1, 0 or 1). False: the new voxel lies in an
// absent brick (air); `voxel` still describes the new voxel. A brick crossing resolves the new brick
// (SceneBrickAt); once absent, the cursor stays absent (the callers stop at air).
bool VoxelCursorStep(CaelixFrameConstants fc, inout VoxelCursor c, int3 delta)
{
    const int3 old_brick = c.voxel >> 3;
    c.voxel += delta;
    const int3 new_brick = c.voxel >> 3;
    CAELIX_BRANCH
    if (any(new_brick != old_brick))
    {
        c.record = c.record != 0 ? SceneBrickAt(FrameSceneLookup(fc), c.group, c.entity, new_brick) : 0;
    }
    return c.record != 0;
}

// The raw block value of the cursor's voxel: the payload word read (0 in an absent brick). The payload
// holds every voxel, also the interior voxels that the occupancy leaves out (scene/caelix_finalize.hlsl).
uint VoxelCursorBlock(VoxelCursor c)
{
    uint block = 0u;
    CAELIX_BRANCH
    if (c.record != 0)
    {
        const uint3 q = uint3(c.voxel & int(kBrickEdge - 1));
        const uint word = CAELIX_LOAD_U32(c.record + uint64_t(RecordPayloadWord(q.x, q.y, q.z)) * kRecordWordBytes);
        block = PayloadPairBlock(word, q.x);
    }
    return block;
}

// The world direction of the entity-local direction `d` under the entity's current transform (rigid:
// the rotation rows of EntityInfo cur).
float3 EntityWorldDirectionById(CaelixFrameConstants fc, uint entity, float3 d)
{
    const uint64_t entity_info = fc.entity_infos + uint64_t(entity) * kEntityInfoBytes;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow0));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow1));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow2));
    return float3(dot(r0.xyz, d), dot(r1.xyz, d), dot(r2.xyz, d));
}

// The entity-local direction of the world direction `d` (the transpose of the rotation).
float3 EntityObjectDirectionById(CaelixFrameConstants fc, uint entity, float3 d)
{
    const uint64_t entity_info = fc.entity_infos + uint64_t(entity) * kEntityInfoBytes;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow0));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow1));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow2));
    return float3(dot(float3(r0.x, r1.x, r2.x), d), dot(float3(r0.y, r1.y, r2.y), d),
                  dot(float3(r0.z, r1.z, r2.z), d));
}

// The world point of the entity-local point `p` under the entity's current transform.
float3 EntityWorldPointById(CaelixFrameConstants fc, uint entity, float3 p)
{
    const uint64_t entity_info = fc.entity_infos + uint64_t(entity) * kEntityInfoBytes;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow0));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow1));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow2));
    const float4 h = float4(p, 1.0f);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

#endif  // CAELIX_VOXEL_CURSOR_HLSL
