// Scene traversal of the render kernels: the boundary between the integrator and
// the scene's data layer. A kernel that includes this file gets the TLAS (set 0,
// binding 0) and reads the scene's tables (page table, slot rows, group infos,
// entity infos) through the device addresses in CaelixFrameConstants. The
// integrator above this file uses only TraceScene, TraceShadow, the entity
// transform helpers and the block code in TraceResult::attrib; nothing above it
// knows what a record, a slot row or a group is.
//
// Instances: instanceCustomIndex (CandidateInstanceID()) is the scene-wide group
// index, the primitive index is the dense brick id inside the group, and the
// object space is the entity-local block space. Before the scene's first TLAS
// build binding 0 holds no valid descriptor (PARTIALLY_BOUND): the callers check
// fc.has_tlas and a ray misses.
#ifndef CAELIX_SCENE_TRACE_HLSL
#define CAELIX_SCENE_TRACE_HLSL
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "render/caelix_brick_trace.hlsl"

CAELIX_BINDING(0, 0) RaytracingAccelerationStructure g_tlas;

uint64_t RecordAddress(CaelixFrameConstants fc, uint record_index)
{
    const uint page = record_index >> fc.records_per_page_shift;
    const uint slot = record_index & ((1u << fc.records_per_page_shift) - 1u);
    const uint64_t page_address = CAELIX_LOAD_U64(fc.page_table + uint64_t(page) * 8);
    return page_address + uint64_t(slot) * fc.record_bytes;
}

// The committed hit of one ray through the scene.
struct TraceResult
{
    bool hit;
    float T;            // world distance (the ray direction is unit length)
    uint attrib;        // AttributeData::matID_faceNormal: block << 16 | face flags
    uint group;         // scene-wide group index
    uint id;            // dense brick id in the group
    uint candidates;    // procedural candidates visited
    float3 normal;      // world-space face normal
    float3 object_hit;  // entity-local block coordinates of the hit
};

// The brick DDA on one procedural candidate (group, dense id) of a ray query:
// returns the DDA's t (object space = world distance, scale 1) and its attribute;
// attrib.matID_faceNormal == 0 means no hit (or the brick is not resident).
float TraceCandidate(CaelixFrameConstants fc, uint group, uint id, float3 object_origin,
                     float3 object_direction, float t_current, out AttributeData attrib)
{
    attrib.matID_faceNormal = 0u;
    // GroupInfo: coord (words 0..2), entity, slot_row_base (word 4), ...
    const uint64_t info = fc.group_infos + uint64_t(group) * 32;
    const uint4 w0 = CAELIX_LOAD_U32x4(info);
    const uint slot_row_base = CAELIX_LOAD_U32(info + 16);
    const uint4 row = CAELIX_LOAD_U32x4(fc.slot_tables + (uint64_t(slot_row_base) + id) * kSlotRowBytes);
    if (SlotRowState(row.x) != kSlotStateResident || row.z == kInvalidRecord)
    {
        return t_current;
    }
    g_recordBase = RecordAddress(fc, row.z);
    // Entity-local block coordinate of the brick's corner (same formula as the
    // finalize kernel's AABB origin).
    const uint shift = fc.group_shift;
    const int mask = int((1u << shift) - 1u);
    const uint pos = SlotRowPos(row.x);
    const int3 brick = int3(int(pos) & mask, int(pos >> shift) & mask, int(pos >> (2 * shift)) & mask);
    const float3 brick_origin = float3((int3(w0.xyz) * int(1u << shift) + brick) * 8);
    return CaelixTraceBrickPrimitiveCore(0, object_origin, object_direction, t_current, brick_origin,
                                         attrib);
}

// Traces `ray` through the TLAS: every procedural candidate runs the brick DDA on
// its record, and a DDA hit inside [TMin, CommittedRayT()) is committed. Before
// the scene's first TLAS build binding 0 holds no valid descriptor
// (PARTIALLY_BOUND): the ray query does not start and the ray misses.
TraceResult TraceScene(CaelixFrameConstants fc, RayDesc ray)
{
    TraceResult r;
    r.hit = false;
    r.T = 0.0f;
    r.attrib = 0;
    r.group = 0;
    r.id = 0;
    r.candidates = 0;
    r.normal = float3(0.0f, 0.0f, 0.0f);
    r.object_hit = float3(0.0f, 0.0f, 0.0f);
    CAELIX_BRANCH
    if (fc.has_tlas != 0u)
    {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, ray);

        while (q.Proceed())
        {
            if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                continue;
            }
            ++r.candidates;
            const uint group = q.CandidateInstanceID();
            const uint id = q.CandidatePrimitiveIndex();
            AttributeData attrib;
            const float t = TraceCandidate(fc, group, id, q.CandidateObjectRayOrigin(),
                                           q.CandidateObjectRayDirection(), q.CommittedRayT(), attrib);
            // Nothing rejects a t outside [TMin, TCurrent] for us, so the test is done here.
            if (attrib.matID_faceNormal != 0u && t >= ray.TMin && t < q.CommittedRayT())
            {
                q.CommitProceduralPrimitiveHit(t);
                r.attrib = attrib.matID_faceNormal;
                r.group = group;
                r.id = id;
            }
        }
        if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
        {
            r.hit = true;
            r.T = q.CommittedRayT();
            const float3x4 object_to_world = q.CommittedObjectToWorld3x4();
            r.normal = normalize(mul((float3x3)object_to_world, UnpackObjectNormal(r.attrib)));
            // The object ray direction is the world direction in object space (not
            // normalized), so the same T applies.
            r.object_hit = q.CommittedObjectRayOrigin() + q.CommittedObjectRayDirection() * r.T;
        }
    }
    return r;
}

// Shadow ray: true when a voxel lies on `ray` inside [TMin, TMax). The first
// candidate whose DDA reports a hit in range is committed and ends the search
// (RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH); no closest-hit data is needed.
bool TraceShadow(CaelixFrameConstants fc, RayDesc ray)
{
    bool occluded = false;
    CAELIX_BRANCH
    if (fc.has_tlas != 0u)
    {
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, ray);
        while (q.Proceed())
        {
            if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE)
            {
                continue;
            }
            AttributeData attrib;
            const float t = TraceCandidate(fc, q.CandidateInstanceID(), q.CandidatePrimitiveIndex(),
                                           q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(),
                                           q.CommittedRayT(), attrib);
            if (attrib.matID_faceNormal != 0u && t >= ray.TMin && t < q.CommittedRayT())
            {
                q.CommitProceduralPrimitiveHit(t);
                occluded = true;
                break;
            }
        }
    }
    return occluded;
}

// The entity that owns `group` (GroupInfo word 3).
uint GroupEntity(CaelixFrameConstants fc, uint group)
{
    return CAELIX_LOAD_U32(fc.group_infos + uint64_t(group) * 32 + 12);
}

// The entity-local point `p` of the entity that owns `group` under the entity's
// current transform (EntityInfo cur rows, bytes 32..79 of the 128-byte row).
float3 EntityCurWorld(CaelixFrameConstants fc, uint group, float3 p)
{
    const uint64_t entity_info = fc.entity_infos + uint64_t(GroupEntity(fc, group)) * kEntityInfoBytes;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + 32));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + 48));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + 64));
    const float4 h = float4(p, 1.0f);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

// The same point under the entity's previous-frame transform (EntityInfo prev
// rows, bytes 80..127).
float3 EntityPrevWorld(CaelixFrameConstants fc, uint group, float3 p)
{
    const uint64_t entity_info = fc.entity_infos + uint64_t(GroupEntity(fc, group)) * kEntityInfoBytes;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + 80));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + 96));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + 112));
    const float4 h = float4(p, 1.0f);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

#endif  // CAELIX_SCENE_TRACE_HLSL
