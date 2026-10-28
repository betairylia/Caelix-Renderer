// Scene traversal of the render kernels: the boundary between the integrator and
// the scene's data layer. A kernel that includes this file gets the TLAS (set 0,
// binding 0) and reads the scene's tables (page table, slot rows, group infos,
// entity infos) through the device addresses in CaelixFrameConstants. The
// integrator above this file uses only TraceScene, TraceShadow, TraceShadowTransmissive (the
// transmissive shadow ray of docs/SHADOW_TRANSMISSION.md: a chain of closest-hit queries through
// glass; in the CAELIX_SUBSURFACE kernels with kRenderFlagShadowSubsurface also a straight walk through
// SSS voxels), the voxel cursor
// and the entity transform helpers (render/caelix_voxel_cursor.hlsl, included here; the subsurface
// walk of docs/SUBSURFACE.md and the shadow ray's SSS walk move through the voxel grid with them),
// VoxelIdHash and the block code in TraceResult::attrib; nothing above it knows what a record, a slot
// row or a group is.
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
#include "scene/caelix_scene_lookup.hlsl"
#include "common/caelix_material.hlsl"
#include "render/caelix_ray.hlsl"

CAELIX_BINDING(0, 0) RaytracingAccelerationStructure g_tlas;

uint64_t RecordAddress(CaelixFrameConstants fc, uint record_index)
{
    const uint page = record_index >> fc.records_per_page_shift;
    const uint slot = record_index & ((1u << fc.records_per_page_shift) - 1u);
    const uint64_t page_address = CAELIX_LOAD_U64(fc.page_table + uint64_t(page) * kPageTableEntryBytes);
    return page_address + uint64_t(slot) * fc.record_bytes;
}

// The committed hit of one ray through the scene.
struct TraceResult
{
    bool hit;
    float T;            // world distance (the ray direction is unit length)
    uint attrib;        // AttributeData::matID_faceNormal: block << 16 | cell << 6 | face flags
    uint group;         // scene-wide group index
    uint id;            // dense brick id in the group
    uint cell;          // the hit voxel inside its brick, x | y << 3 | z << 6 (0 on a miss)
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
    const uint64_t info = fc.group_infos + uint64_t(group) * kGroupInfoBytes;
    const uint4 w0 = CAELIX_LOAD_U32x4(info);  // coord, entity (kGroupInfoCoord = 0)
    const uint slot_row_base = CAELIX_LOAD_U32(info + kGroupInfoSlotRowBase);
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
    const float3 brick_origin = float3((int3(w0.xyz) * int(1u << shift) + brick) * int(kBrickEdge));
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
    r.cell = 0;
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
                r.cell = (attrib.matID_faceNormal >> 6) & 0x1FFu;
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

// One word of the voxel id: avalanche hashes of the entity and the voxel's coordinates, chained
// from `seed`; 0xFFFFFFFF (the miss value) becomes 0xFFFFFFFE.
uint VoxelIdWord(uint seed, uint entity, int3 voxel)
{
    uint h = CaelixHashAvalanche(seed ^ entity);
    h = CaelixHashAvalanche(h ^ uint(voxel.x));
    h = CaelixHashAvalanche(h ^ uint(voxel.y));
    h = CaelixHashAvalanche(h ^ uint(voxel.z));
    return h == 0xFFFFFFFFu ? 0xFFFFFFFEu : h;
}

// The voxel id of a hit: two independent 32-bit hashes of (entity id, entity-local integer
// voxel position); the voxel is the brick of the hit (group coordinate and slot row position,
// as TraceCandidate derives the brick origin) plus the DDA cell of the hit. Neither word is
// ever 0xFFFFFFFF (the miss value of the voxel id target).
uint2 VoxelIdHash(CaelixFrameConstants fc, TraceResult r)
{
    const uint64_t info = fc.group_infos + uint64_t(r.group) * kGroupInfoBytes;
    const uint4 w0 = CAELIX_LOAD_U32x4(info);  // coord, entity (kGroupInfoCoord = 0)
    const uint slot_row_base = CAELIX_LOAD_U32(info + kGroupInfoSlotRowBase);
    const uint row_x = CAELIX_LOAD_U32(fc.slot_tables + (uint64_t(slot_row_base) + r.id) * kSlotRowBytes);
    const uint shift = fc.group_shift;
    const int mask = int((1u << shift) - 1u);
    const uint pos = SlotRowPos(row_x);
    const int3 brick = int3(int(pos) & mask, int(pos >> shift) & mask, int(pos >> (2 * shift)) & mask);
    const int3 cell = int3(int(r.cell & 7u), int((r.cell >> 3) & 7u), int((r.cell >> 6) & 7u));
    const int3 voxel = (int3(w0.xyz) * int(1u << shift) + brick) * int(kBrickEdge) + cell;
    const uint entity = w0.w;  // kGroupInfoEntity
    return uint2(VoxelIdWord(0x9E3779B9u, entity, voxel), VoxelIdWord(0x85EBCA6Bu, entity, voxel));
}

// The entity that owns `group` (GroupInfo entity).
uint GroupEntity(CaelixFrameConstants fc, uint group)
{
    return CAELIX_LOAD_U32(fc.group_infos + uint64_t(group) * kGroupInfoBytes + kGroupInfoEntity);
}

// The entity-local point `p` of the entity that owns `group` under the entity's
// current transform (EntityInfo cur rows).
float3 EntityCurWorld(CaelixFrameConstants fc, uint group, float3 p)
{
    const uint64_t entity_info = fc.entity_infos + uint64_t(GroupEntity(fc, group)) * kEntityInfoBytes;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow0));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow1));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoCurRow2));
    const float4 h = float4(p, 1.0f);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

// The same point under the entity's previous-frame transform (EntityInfo prev
// rows).
float3 EntityPrevWorld(CaelixFrameConstants fc, uint group, float3 p)
{
    const uint64_t entity_info = fc.entity_infos + uint64_t(GroupEntity(fc, group)) * kEntityInfoBytes;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoPrevRow0));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoPrevRow1));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + kEntityInfoPrevRow2));
    const float4 h = float4(p, 1.0f);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

// The voxel cursor and the entity transform helpers by entity id (they read TraceResult).
#include "render/caelix_voxel_cursor.hlsl"
// RandomFloat01: the entry draw of an SSS voxel in TraceShadowTransmissive.
#include "render/caelix_random.hlsl"

#ifndef CAELIX_SUBSURFACE
#define CAELIX_SUBSURFACE 0
#endif

// The cap of closest-hit queries (hops) of the transmissive shadow ray.
static const uint kShadowMaxHops = 8;
// A ray whose remaining length is at least this is unbounded (the sun's TMax): a miss inside a medium
// adds no segment.
static const float kShadowUnboundedT = 1e29f;
// Max channel of the transmittance below which a shadow ray counts as occluded.
static const float kShadowTransmitCutoff = 1e-3f;

// The transmittance of a segment of length t inside the transparent medium `medium_id` (a face-free
// transparent id; 0 = air: 1): exp(-(1 - albedo) extinction t), the formula of MediumExtinction (the
// air's fog is ignored: shadow rays ignore fog).
float3 ShadowGlassTransmittance(uint medium_id, float t)
{
    if (medium_id == 0u)
    {
        return float3(1.0f, 1.0f, 1.0f);
    }
    const CaelixMaterial material = GET_MATERIAL(medium_id);
    return exp(-(1.0f - material.albedo) * t * material.extinction);
}

// The transmittance is below the cutoff (max channel): the ray counts as occluded.
bool ShadowOccluded(float3 transmittance)
{
    return max(transmittance.r, max(transmittance.g, transmittance.b)) < kShadowTransmitCutoff;
}

#if CAELIX_SUBSURFACE

// The subsurface entry of the block `block`: true for an SSS voxel of the HG walk (opaque, model
// kSssModelHg, sss_probability > 0 and sss_steps > 0) with its sss_albedo, probability and step count;
// one 16-byte load of the subsurface entry (w0..w3) for an opaque code, none for the others (albedo 0,
// probability 0, steps 0 when it is not an SSS voxel).
bool SubsurfaceEntry(uint block, out float3 albedo, out float probability, out uint steps)
{
    albedo = float3(0.0f, 0.0f, 0.0f);
    probability = 0.0f;
    steps = 0u;
    CAELIX_BRANCH
    if (!IsOpaque(int(block)))
    {
        return false;
    }
    const uint4 w = CAELIX_LOAD_U32x4(g_subsurfaceTable + uint64_t(CaelixMaterialCode(block)) * kSubsurfaceBytes);
    const float p = CaelixHalfLo(w.w);
    const uint s = w.z & 0xFFu;
    const bool sss = ((w.w >> 16) & 0xFFu) == kSssModelHg && p > 0.0f && s != 0u;
    albedo = sss ? 1.0f - float3(CaelixHalfLo(w.x), CaelixHalfHi(w.x), CaelixHalfLo(w.y)) : albedo;
    probability = sss ? p : 0.0f;
    steps = sss ? s : 0u;
    return sss;
}

// SubsurfaceEntry with the albedo only (the walk's per-voxel read).
bool SubsurfaceVoxel(uint block, out float3 albedo)
{
    float probability;
    uint steps;
    return SubsurfaceEntry(block, albedo, probability, steps);
}

// ShadowSubsurfaceWalk results.
static const uint kShadowWalkOccluded = 0;  // a non-SSS opaque voxel, or the step cap: the ray is occluded
static const uint kShadowWalkExit = 1;      // the ray left the SSS voxels: the chain continues at `exit_origin`
static const uint kShadowWalkEnd = 2;       // the ray's end point lies inside the SSS voxels (a lamp ray): visible

// The straight walk of a shadow ray with unit world direction `direction` through the SSS voxels it
// entered at `hit` (an SSS voxel whose subsurface entry gave `entry_albedo` and `steps`; the chain has
// already drawn the entry): a 3D DDA through the hit entity's voxel grid from hit.object_hit, with the
// voxel cursor (other entities count as air, as in SubsurfaceWalk). Every SSS voxel visited (the entry
// voxel included) multiplies `transmittance` by its sss_albedo; a non-SSS opaque voxel occludes; more
// than `steps` SSS voxels occlude (the walk is dead, the rule of SubsurfaceWalk); any other voxel
// (air, glass, an absent brick) is the exit: `exit_origin` = the world point on the leaving face pushed
// off along the face normal, `exit_block` its block value, `travelled` the distance along the ray from
// the hit to the pushed-off origin. `remaining` = the ray length left after the hit: a boundary at or
// beyond it means the ray's end lies inside the current voxel (kShadowWalkEnd).
uint ShadowSubsurfaceWalk(CaelixFrameConstants fc, TraceResult hit, float3 direction, float3 entry_albedo,
                          uint steps, float remaining, inout float3 transmittance, out float3 exit_origin,
                          out uint exit_block, out float travelled)
{
    exit_origin = float3(0.0f, 0.0f, 0.0f);
    exit_block = 0u;
    travelled = 0.0f;
    VoxelCursor c = VoxelCursorFromHit(fc, hit);
    const float3 d = EntityObjectDirectionById(fc, c.entity, direction);
    // The entry point, clamped into the entry voxel (object_hit lies on its face up to rounding).
    const float3 p = clamp(hit.object_hit, float3(c.voxel), float3(c.voxel) + 1.0f);
    const float3 t_delta = 1.0f / max(abs(d), 1e-8f);
    const int3 step = int3(d.x >= 0.0f ? 1 : -1, d.y >= 0.0f ? 1 : -1, d.z >= 0.0f ? 1 : -1);
    const float3 next = float3(c.voxel) + float3(d.x >= 0.0f ? 1.0f : 0.0f, d.y >= 0.0f ? 1.0f : 0.0f,
                                                 d.z >= 0.0f ? 1.0f : 0.0f);
    float3 t_max = abs(next - p) * t_delta;
    transmittance *= entry_albedo;
    CAELIX_LOOP
    for (uint i = 0; i < steps; ++i)
    {
        // The nearest voxel boundary (ties: x, then y, then z), then the step across it.
        float t;
        int3 delta;
        if (t_max.x <= t_max.y && t_max.x <= t_max.z)
        {
            t = t_max.x;
            t_max.x += t_delta.x;
            delta = int3(step.x, 0, 0);
        }
        else if (t_max.y <= t_max.z)
        {
            t = t_max.y;
            t_max.y += t_delta.y;
            delta = int3(0, step.y, 0);
        }
        else
        {
            t = t_max.z;
            t_max.z += t_delta.z;
            delta = int3(0, 0, step.z);
        }
        if (t >= remaining)
        {
            return kShadowWalkEnd;
        }
        VoxelCursorStep(fc, c, delta);
        const uint block = VoxelCursorBlock(c);
        float3 albedo;
        CAELIX_BRANCH
        if (SubsurfaceVoxel(block, albedo))
        {
            transmittance *= albedo;
            continue;
        }
        if (IsOpaque(int(block)))
        {
            return kShadowWalkOccluded;
        }
        // The exit through the face crossed by this step: into the exit voxel, pushed off along the face
        // normal (the push-off moves the origin along the ray too).
        const float3 n = EntityWorldDirectionById(fc, c.entity, float3(delta));
        exit_origin = EntityWorldPointById(fc, c.entity, p + d * t) + K_RAY_ORIGIN_PUSH_OFF * n;
        exit_block = block;
        travelled = t + K_RAY_ORIGIN_PUSH_OFF * dot(n, direction);
        return kShadowWalkExit;
    }
    return kShadowWalkOccluded;
}

#endif  // CAELIX_SUBSURFACE

// The transmittance of the shadow ray `ray` (unit direction; [0, TMax)) that starts in the transparent
// medium `medium` (a face-free transparent id, 0 = air), in a straight line (no refraction,
// docs/SHADOW_TRANSMISSION.md): opaque voxels occlude, transparent media attenuate by their extinction.
// A chain of up to kShadowMaxHops closest-hit queries, interface to interface (a glass exit is a hit on
// an air cell with face bits): the common case (air, nothing or an opaque voxel on the ray) costs one
// query. After the hop cap: 0.
// SSS voxels: in the CAELIX_SUBSURFACE kernels with kRenderFlagShadowSubsurface and kRenderFlagSubsurface
// (the vertex walk's flag) the ray enters an SSS voxel with its sss_probability (one number of `rng`;
// Russian roulette without a weight) and walks straight through the SSS voxels
// (ShadowSubsurfaceWalk: sss_albedo per voxel visited, dead beyond the entry material's sss_steps
// voxels, a non-SSS opaque voxel occludes); the walk and its exit use one hop of the cap, and the exit
// point starts the next hop. Without CAELIX_SUBSURFACE or without both flags an SSS voxel occludes like
// any opaque voxel (`rng` is not used).
float3 TraceShadowTransmissive(CaelixFrameConstants fc, RayDesc ray, uint medium, inout uint rng)
{
    float3 transmittance = float3(1.0f, 1.0f, 1.0f);
    uint cur = medium;
    float remaining = ray.TMax;
    RayDesc hop = ray;
    CAELIX_LOOP
    for (uint hop_index = 0; hop_index < kShadowMaxHops; ++hop_index)
    {
        hop.TMax = remaining;
        const TraceResult h = TraceScene(fc, hop);
        if (!h.hit)
        {
            if (cur != 0u && remaining < kShadowUnboundedT)
            {
                transmittance *= ShadowGlassTransmittance(cur, remaining);
            }
            return ShadowOccluded(transmittance) ? float3(0.0f, 0.0f, 0.0f) : transmittance;
        }
        transmittance *= ShadowGlassTransmittance(cur, h.T);
        if (ShadowOccluded(transmittance))
        {
            return float3(0.0f, 0.0f, 0.0f);
        }
        remaining -= h.T;
        const uint block = (h.attrib >> 16) & 0xFFFFu;
        if (IsOpaque(int(block)))
        {
#if CAELIX_SUBSURFACE
            // An SSS voxel: with its sss_probability the ray enters and walks straight through the SSS
            // voxels (Russian roulette, no weight: the estimate of p times the transmittance is unbiased);
            // otherwise it occludes.
            float3 sss_albedo;
            float sss_probability;
            uint sss_steps;
            const uint sss_flags = kRenderFlagSubsurface | kRenderFlagShadowSubsurface;
            CAELIX_BRANCH
            if ((fc.flags & sss_flags) == sss_flags &&
                SubsurfaceEntry(block, sss_albedo, sss_probability, sss_steps))
            {
                if (!(RandomFloat01(rng) < sss_probability))
                {
                    return float3(0.0f, 0.0f, 0.0f);
                }
                float3 exit_origin;
                uint exit_block;
                float travelled;
                const uint walk = ShadowSubsurfaceWalk(fc, h, hop.Direction, sss_albedo, sss_steps, remaining,
                                                       transmittance, exit_origin, exit_block, travelled);
                if (walk == kShadowWalkOccluded || ShadowOccluded(transmittance))
                {
                    return float3(0.0f, 0.0f, 0.0f);
                }
                if (walk == kShadowWalkEnd)
                {
                    return transmittance;
                }
                cur = CaelixTransparentId(exit_block);
                hop.Origin = exit_origin;
                remaining -= travelled;
                continue;
            }
#endif
            // An opaque voxel occludes.
            return float3(0.0f, 0.0f, 0.0f);
        }
        // An interface voxel (glass entry, glass exit = an air cell with face bits, glass to glass):
        // into the entered voxel (h.normal faces the ray). The push-off moves the origin along the ray
        // too: the remaining length keeps the ray's end point (a lamp ray must stop short of the lamp's
        // face).
        cur = CaelixTransparentId(block);
        const float3 offset = -K_RAY_ORIGIN_PUSH_OFF * h.normal;
        hop.Origin = hop.Origin + hop.Direction * h.T + offset;
        remaining -= dot(offset, hop.Direction);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

#endif  // CAELIX_SCENE_TRACE_HLSL
