// Cheap subsurface scattering of the trace kernel (EXPERIMENT, branch sss-expr; the model and the owner
// decisions in docs/SSS_EXPERIMENT.md).
//
// An OPAQUE block code whose material has sss_probability = p > 0 is an SSS voxel. A path vertex on an
// SSS face (the primary surface or a bounce hit, with kRenderFlagSubsurface) draws one number: with
// probability p it runs a random walk through the entity's voxel grid instead of the surface BSDF
// (SubsurfaceVertex). The walk (SubsurfaceWalk) is an integer lattice walk over the 6 face directions
// with no ray query: per step one scattering event in the current voxel (weight *= the voxel's
// sss_albedo, a new continuous direction from the Henyey-Greenstein lobe of the entry material), then
// a look at the face neighbour along the dominant axis of that direction: an SSS voxel is entered, any
// other opaque voxel is a wall (the walk stays), anything else (air, glass, an absent brick) is the
// exit. The exit becomes a diffuse vertex on the leaving face (uniform point, cosine direction around
// the face normal) with the walk's weight as its albedo; a walk that reaches the entry material's step
// cap is absorbed (dead). The walk stays inside the hit entity (other entities count as air). The
// per-voxel row (sss_albedo, sss_probability) comes from the subsurface table (fc.subsurface_table), the
// entry material's anisotropy and step cap from the main material table.
//
// CAELIX_SUBSURFACE 1 only in the trace kernels caelix_path_trace_sss(_no_light_nee) (the renderer binds
// them for lighting frames with the flag whose scene has an SSS material); the default 0 compiles
// nothing of this file, so the main trace kernels keep their code.
#ifndef CAELIX_SUBSURFACE_HLSL
#define CAELIX_SUBSURFACE_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "common/caelix_material.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "render/caelix_shading.hlsl"

#ifndef CAELIX_SUBSURFACE
#define CAELIX_SUBSURFACE 0
#endif

#if CAELIX_SUBSURFACE

// SubsurfaceVertex results.
static const uint kSubsurfaceNone = 0;  // not an SSS material: the shade stands
static const uint kSubsurfaceBsdf = 1;  // an SSS material whose draw chose the surface BSDF: the shade stands
static const uint kSubsurfaceExit = 2;  // the walk left the medium: the shade is the exit vertex
static const uint kSubsurfaceDead = 3;  // the walk reached its step cap: the path ends at this vertex

// ---------------------------------------------------------------------------
// The voxel cursor

// A voxel of an entity's grid: its entity-local coordinate, the brick's record and its group (the brick
// coordinate, voxel >> 3, and the voxel in the brick, voxel & 7, are derived where needed: fewer live
// registers in the walk).
struct MediumCursor
{
    uint group;       // the scene-wide group index of the brick (unspecified in an absent brick)
    uint entity;
    uint64_t record;  // 0 = absent brick (every voxel air)
    int3 voxel;       // entity-local voxel coordinate
};

// The voxel that the face hit `hit` entered (half a voxel behind the hit point along the object-space face
// normal), in the brick that the hit's candidate traced (it is resident).
MediumCursor MediumCursorFromHit(CaelixFrameConstants fc, SceneLookup s, TraceResult hit)
{
    const uint64_t info = fc.group_infos + uint64_t(hit.group) * kGroupInfoBytes;
    const uint4 w0 = CAELIX_LOAD_U32x4(info);                          // coord, entity (kGroupInfoCoord = 0)
    const uint4 w1 = CAELIX_LOAD_U32x4(info + kGroupInfoSlotRowBase);  // slot_row_base, reverse_base, ...
    const uint4 row = CAELIX_LOAD_U32x4(fc.slot_tables + (uint64_t(w1.x) + hit.id) * kSlotRowBytes);
    MediumCursor c;
    c.group = hit.group;
    c.entity = w0.w;
    c.record = SceneRecordAddress(s, row.z);
    // Entity-local block coordinate of the brick's corner (as in TraceCandidate).
    const uint shift = fc.group_shift;
    const int mask = int((1u << shift) - 1u);
    const uint pos = SlotRowPos(row.x);
    const int3 brick = int3(int(pos) & mask, int(pos >> shift) & mask, int(pos >> (2 * shift)) & mask);
    const int3 origin = (int3(w0.xyz) * int(1u << shift) + brick) * int(kBrickEdge);
    const float3 n = float3(UnpackObjectNormal(hit.attrib));
    c.voxel = origin + clamp(int3(floor(hit.object_hit - 0.5f * n)) - origin, 0, int(kBrickEdge) - 1);
    return c;
}

// Moves the cursor by `delta` voxels (each component -1, 0 or 1). False: the new voxel lies in an absent brick
// (air); `voxel` still describes the new voxel. A brick crossing resolves the new brick (SceneBrickAt).
bool MediumStep(SceneLookup s, inout MediumCursor c, int3 delta)
{
    const int3 old_brick = c.voxel >> 3;
    c.voxel += delta;
    const int3 new_brick = c.voxel >> 3;
    CAELIX_BRANCH
    if (any(new_brick != old_brick))
    {
        c.record = c.record != 0 ? SceneBrickAt(s, c.group, c.entity, new_brick) : 0;
    }
    return c.record != 0;
}

// The raw block value of the cursor's voxel (0 in an absent brick).
uint MediumBlock(MediumCursor c)
{
    uint block = 0u;
    CAELIX_BRANCH
    if (c.record != 0)
    {
        const uint3 q = uint3(c.voxel & int(kBrickEdge - 1));
        const uint word =
            CAELIX_LOAD_U32(c.record + uint64_t(RecordPayloadWord(q.x, q.y, q.z)) * kRecordWordBytes);
        block = PayloadPairBlock(word, q.x);
    }
    return block;
}

// ---------------------------------------------------------------------------
// Phase function

// A direction from the Henyey-Greenstein lobe of anisotropy g around the unit direction w (inversion of
// the cosine's distribution; g > 0 scatters forward, g = 0 is the uniform sphere).
float3 SampleHenyeyGreenstein(float3 w, float g, inout uint rng)
{
    const float u1 = RandomFloat01(rng);
    const float u2 = RandomFloat01(rng);
    float cos_theta;
    if (abs(g) < 1e-3f)
    {
        cos_theta = 1.0f - 2.0f * u1;
    }
    else
    {
        const float sq = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
        cos_theta = (1.0f + g * g - sq * sq) / (2.0f * g);
    }
    cos_theta = clamp(cos_theta, -1.0f, 1.0f);
    const float sin_theta = sqrt(max(1.0f - cos_theta * cos_theta, 0.0f));
    const float phi = K_TWO_PI * u2;
    float3 t;
    float3 b;
    OrthonormalBasis(w, t, b);
    return t * (cos(phi) * sin_theta) + b * (sin(phi) * sin_theta) + w * cos_theta;
}

// ---------------------------------------------------------------------------
// The walk

// The subsurface row (sss_albedo, sss_probability) of a raw block value, from the subsurface table
// (fc.subsurface_table, kSubsurfaceRowBytes per code). True when the block is an SSS voxel: an opaque code
// whose sss_probability is > 0. One 16-byte load for an opaque code, none for the others (`row` is 0 then).
bool SubsurfaceRow(CaelixFrameConstants fc, uint block, out float4 row)
{
    row = float4(0.0f, 0.0f, 0.0f, 0.0f);
    CAELIX_BRANCH
    if (!IsOpaque(int(block)))
    {
        return false;
    }
    row = asfloat(CAELIX_LOAD_U32x4(fc.subsurface_table + uint64_t(CaelixMaterialCode(block)) * kSubsurfaceRowBytes));
    return row.w > 0.0f;
}

// The walk from the SSS face hit `hit` of a ray with world direction `ray_direction`, in the hit entity's
// voxel grid (docs/SSS_EXPERIMENT.md, "Walk"). True: the walk left the medium; origin (pushed off), normal
// and the cosine-distributed direction of the exit vertex in world space, weight = the product of the
// sss_albedo of every step's voxel. False: the step cap was reached (absorbed; the outputs are 0). The
// anisotropy and the step cap come from the entry material (main table, kMaterialSssAnisotropy and
// kMaterialSssSteps).
// Entry jitter (kRenderFlagSssEntryJitter): the start voxel is the voxel of the hit point moved by a
// uniform offset in [-0.5, 0.5] along the two face-tangent axes (object space, half a voxel behind the
// face): a one-voxel box filter over the face, against the per-voxel brightness steps of a walk that
// always starts in the hit voxel. A start voxel that is not an SSS voxel of the entity (absent brick,
// air, an opaque voxel without SSS) falls back to the hit voxel. Two numbers of `rng`.
bool SubsurfaceWalk(CaelixFrameConstants fc, TraceResult hit, float3 ray_direction, inout uint rng,
                    out float3 origin, out float3 normal, out float3 direction, out float3 weight)
{
    origin = float3(0.0f, 0.0f, 0.0f);
    normal = float3(0.0f, 0.0f, 0.0f);
    direction = float3(0.0f, 0.0f, 0.0f);
    weight = float3(0.0f, 0.0f, 0.0f);
    const SceneLookup s = FrameSceneLookup(fc);
    MediumCursor c = MediumCursorFromHit(fc, s, hit);
    const uint block = (hit.attrib >> 16) & 0xFFFFu;
    const uint64_t entry = g_materialTable + uint64_t(CaelixMaterialCode(block)) * kMaterialBytes;
    const uint2 walk = CAELIX_LOAD_U32x2(entry + kMaterialSssAnisotropy);  // sss_anisotropy, sss_steps
    const float g = asfloat(walk.x);
    const uint steps = walk.y;
    float4 entry_row;
    SubsurfaceRow(fc, block, entry_row);
    float3 a = entry_row.xyz;
    CAELIX_BRANCH
    if ((fc.flags & kRenderFlagSssEntryJitter) != 0u)
    {
        const float3 face_normal = float3(UnpackObjectNormal(hit.attrib));
        const float u1 = RandomFloat01(rng) - 0.5f;
        const float u2 = RandomFloat01(rng) - 0.5f;
        const float3 offset = face_normal.x != 0.0f
                                  ? float3(0.0f, u1, u2)
                                  : (face_normal.y != 0.0f ? float3(u1, 0.0f, u2) : float3(u1, u2, 0.0f));
        const int3 target = int3(floor(hit.object_hit - 0.5f * face_normal + offset));
        // At most one voxel along each tangent axis (one MediumStep resolves a neighbour brick).
        const int3 delta = clamp(target - c.voxel, -1, 1) * int3(face_normal == 0.0f);
        CAELIX_BRANCH
        if (any(delta != 0))
        {
            MediumCursor start = c;
            MediumStep(s, start, delta);
            float4 row;
            if (SubsurfaceRow(fc, MediumBlock(start), row))
            {
                c = start;
                a = row.xyz;
            }
        }
    }
    float3 d = EntityObjectDirectionById(fc, c.entity, ray_direction);
    float3 w = float3(1.0f, 1.0f, 1.0f);
    CAELIX_LOOP
    for (uint i = 0; i < steps; ++i)
    {
        // One scattering event in the current voxel, then a new direction.
        w *= a;
        d = SampleHenyeyGreenstein(d, g, rng);
        // The face neighbour along the dominant axis of d (ties: x, then y, then z).
        const float3 ad = abs(d);
        int3 delta;
        if (ad.x >= ad.y && ad.x >= ad.z)
        {
            delta = int3(d.x >= 0.0f ? 1 : -1, 0, 0);
        }
        else if (ad.y >= ad.z)
        {
            delta = int3(0, d.y >= 0.0f ? 1 : -1, 0);
        }
        else
        {
            delta = int3(0, 0, d.z >= 0.0f ? 1 : -1);
        }
        MediumCursor next = c;
        MediumStep(s, next, delta);
        const uint next_block = MediumBlock(next);
        CAELIX_BRANCH
        if (IsOpaque(int(next_block)))
        {
            // An SSS voxel is entered; any other opaque voxel is a wall: the cursor stays.
            float4 row;
            if (SubsurfaceRow(fc, next_block, row))
            {
                c = next;
                a = row.xyz;
            }
            continue;
        }
        // Exit through the face of the cursor's voxel in direction delta: a uniform point on the face.
        const float3 n = float3(delta);
        const float u1 = RandomFloat01(rng) - 0.5f;
        const float u2 = RandomFloat01(rng) - 0.5f;
        const float3 offset = delta.x != 0 ? float3(0.0f, u1, u2) : (delta.y != 0 ? float3(u1, 0.0f, u2)
                                                                                     : float3(u1, u2, 0.0f));
        const float3 exit_point = float3(c.voxel) + 0.5f + 0.5f * n + offset;
        normal = EntityWorldDirectionById(fc, c.entity, n);
        origin = EntityWorldPointById(fc, c.entity, exit_point) + K_RAY_ORIGIN_PUSH_OFF * normal;
        // The cosine lobe around the exit normal (the diffuse lobe of ShadeVoxelHit).
        direction = SafeNormalize(normal + RandomUnitVector(rng));
        direction = (dot(direction, direction) > 0.0f) ? direction : normal;
        weight = w;
        return true;
    }
    return false;
}

// The subsurface choice at a path vertex: the hit `hit` of a ray with world direction `ray_direction`,
// after ShadeVoxelHit gave `shade`. An SSS material (SubsurfaceRow; `row` = its row, 0 otherwise) draws
// one number of `rng`: below its sss_probability the walk runs and its exit replaces the vertex (albedo
// = the walk's weight, the exit's bounce origin and direction, the diffuse lobe with roughness 1, and
// `vertex_normal` = the exit normal); a dead walk sets the albedo to 0. Other materials draw nothing.
uint SubsurfaceVertex(CaelixFrameConstants fc, TraceResult hit, float3 ray_direction, inout uint rng,
                      inout CaelixShadeResult shade, inout float3 vertex_normal, out float4 row)
{
    const uint block = (hit.attrib >> 16) & 0xFFFFu;
    if (!SubsurfaceRow(fc, block, row))
    {
        return kSubsurfaceNone;
    }
    if (!(RandomFloat01(rng) < row.w))
    {
        return kSubsurfaceBsdf;
    }
    float3 origin;
    float3 normal;
    float3 direction;
    float3 weight;
    if (!SubsurfaceWalk(fc, hit, ray_direction, rng, origin, normal, direction, weight))
    {
        shade.albedo = float3(0.0f, 0.0f, 0.0f);
        return kSubsurfaceDead;
    }
    shade.albedo = weight;
    shade.bounce_origin = origin;
    shade.bounce_direction = direction;
    shade.specular_lobe = false;
    shade.roughness = 1.0f;
    vertex_normal = normal;
    return kSubsurfaceExit;
}

#endif  // CAELIX_SUBSURFACE

#endif  // CAELIX_SUBSURFACE_HLSL
