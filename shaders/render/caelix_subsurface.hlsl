// Subsurface scattering of the trace kernel: the Henyey-Greenstein lattice walk (docs/SUBSURFACE.md).
//
// An SSS voxel is an OPAQUE block code whose material has sss_model kSssModelHg, sss_probability p > 0
// and sss_steps > 0. A path vertex on an SSS face (the primary surface or a bounce hit, with
// kRenderFlagSubsurface) draws one number: with probability p it runs a random walk through the hit
// entity's voxel grid instead of the surface BSDF (SubsurfaceVertex). The walk (SubsurfaceWalk) is an
// integer lattice walk over the 6 face directions with no ray query: per step one scattering event in
// the current voxel (weight *= the voxel's sss_albedo, a new continuous direction from the
// Henyey-Greenstein lobe of the entry material), then a look at the face neighbour along the dominant
// axis of that direction: an SSS voxel is entered, any other opaque voxel is a wall (the walk stays),
// anything else (air, glass, an absent brick) is the exit. The exit becomes a diffuse vertex on the
// leaving face (uniform point, cosine direction around the face normal) with the walk's weight as its
// albedo; a walk that reaches the entry material's step cap is absorbed (dead). The walk stays inside
// the hit entity (other entities count as air). It moves with the voxel cursor of
// render/caelix_voxel_cursor.hlsl (included by render/caelix_scene_trace.hlsl), so it stays above the
// scene-trace boundary.
//
// Material reads: the vertex test loads the hot entry's sss word (CaelixMaterialSssWord, 4 bytes, a
// cache hit after ShadeVoxelHit); an SSS candidate then loads w0..w3 of its subsurface entry (16 bytes:
// one minus sss_albedo, anisotropy, steps, probability and model), and so does every voxel the walk
// looks at (SubsurfaceVoxel, defined with SubsurfaceEntry in render/caelix_scene_trace.hlsl under
// CAELIX_SUBSURFACE: the transmissive shadow ray reads the same entries).
//
// Compiled only with CAELIX_SUBSURFACE 1: the trace kernel variants caelix_path_trace_sss and
// caelix_path_trace_sss_no_light_nee (render/caelix_path_trace_sss*.hlsl), which the renderer binds
// for the lighting frames of a scene whose material table has an SSS entry. The default 0 (defined
// below) compiles nothing of this file, so the main trace kernels keep their code. The primary vertex's
// transmissive shadow ray enters SSS voxels with the same probability and walks straight through them
// (docs/SHADOW_TRANSMISSION.md); the other shadow rays stop at SSS voxels like at any opaque voxel.
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
static const uint kSubsurfaceNone = 0;  // not an SSS vertex (or the last vertex): the shade stands
static const uint kSubsurfaceBsdf = 1;  // an SSS material whose draw chose the surface BSDF: the shade stands
static const uint kSubsurfaceExit = 2;  // the walk left the medium: the shade is the exit vertex
static const uint kSubsurfaceDead = 3;  // the walk reached its step cap: the path ends at this vertex

// The anisotropy range of the Henyey-Greenstein inversion, and the |g| below which the lobe is the
// uniform sphere.
static const float kSubsurfaceMaxAnisotropy = 0.99f;
static const float kSubsurfaceIsotropicG = 1e-3f;

// ---------------------------------------------------------------------------
// Phase function

// A direction from the Henyey-Greenstein lobe of anisotropy g around the unit direction w (inversion of
// the cosine's distribution; g > 0 scatters forward, |g| < kSubsurfaceIsotropicG is the uniform
// sphere). Two numbers of `rng`.
float3 SampleHenyeyGreenstein(float3 w, float g, inout uint rng)
{
    const float u1 = RandomFloat01(rng);
    const float u2 = RandomFloat01(rng);
    float cos_theta;
    if (abs(g) < kSubsurfaceIsotropicG)
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

// The walk from the SSS face hit `hit` of a ray with world direction `ray_direction`, in the hit entity's
// voxel grid (docs/SUBSURFACE.md section 2). `g` (clamped) and `steps` (> 0) come from the entry
// material's subsurface entry, `entry_albedo` is its sss_albedo. True: the walk left the medium; origin
// (pushed off), normal and the cosine-distributed direction of the exit vertex in world space, weight =
// the product of the sss_albedo of every step's voxel. False: the step cap was reached (absorbed; the
// outputs are 0).
// Entry jitter (always on): the start voxel is the voxel of the hit point moved by a uniform offset in
// [-0.5, 0.5] along the two face-tangent axes (object space, half a voxel behind the face), reached by
// one cursor step of at most one voxel per tangent axis: a one-voxel box filter over the face, against
// the per-voxel brightness steps of a walk that always starts in the hit voxel. A start voxel that is
// not an SSS voxel of the entity (absent brick, air, an opaque voxel without SSS) falls back to the hit
// voxel. Two numbers of `rng`; then two per step and four at the exit.
bool SubsurfaceWalk(CaelixFrameConstants fc, TraceResult hit, float3 ray_direction, float g, uint steps,
                    float3 entry_albedo, inout uint rng, out float3 origin, out float3 normal, out float3 direction,
                    out float3 weight)
{
    origin = float3(0.0f, 0.0f, 0.0f);
    normal = float3(0.0f, 0.0f, 0.0f);
    direction = float3(0.0f, 0.0f, 0.0f);
    weight = float3(0.0f, 0.0f, 0.0f);
    VoxelCursor c = VoxelCursorFromHit(fc, hit);
    float3 a = entry_albedo;
    {
        const float3 face_normal = float3(UnpackObjectNormal(hit.attrib));
        const float u1 = RandomFloat01(rng) - 0.5f;
        const float u2 = RandomFloat01(rng) - 0.5f;
        const float3 offset = face_normal.x != 0.0f
                                  ? float3(0.0f, u1, u2)
                                  : (face_normal.y != 0.0f ? float3(u1, 0.0f, u2) : float3(u1, u2, 0.0f));
        const int3 target = int3(floor(hit.object_hit - 0.5f * face_normal + offset));
        // At most one voxel along each tangent axis (one cursor step resolves a neighbour brick).
        const int3 delta = clamp(target - c.voxel, -1, 1) * int3(face_normal == 0.0f);
        CAELIX_BRANCH
        if (any(delta != 0))
        {
            VoxelCursor start = c;
            VoxelCursorStep(fc, start, delta);
            float3 start_albedo;
            if (SubsurfaceVoxel(VoxelCursorBlock(start), start_albedo))
            {
                c = start;
                a = start_albedo;
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
        VoxelCursor next = c;
        VoxelCursorStep(fc, next, delta);
        const uint next_block = VoxelCursorBlock(next);
        CAELIX_BRANCH
        if (IsOpaque(int(next_block)))
        {
            // An SSS voxel is entered; any other opaque voxel is a wall: the cursor stays.
            float3 next_albedo;
            if (SubsurfaceVoxel(next_block, next_albedo))
            {
                c = next;
                a = next_albedo;
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
// after ShadeVoxelHit gave `shade`. Not an SSS material (the hot entry's sss word: opaque, model
// kSssModelHg, sss_probability > 0; then sss_steps > 0 from the subsurface entry) or `last_vertex` (no
// bounce left: only the vertex's emission counts, the same estimate without the walk): kSubsurfaceNone,
// nothing drawn. Otherwise `guide_albedo` = lerp(albedo, sss_albedo, p) (the primary vertex's
// demodulation albedo; the shade's albedo otherwise) and one number u of `rng`: u >= p ->
// kSubsurfaceBsdf (the shade stands); else the walk: its exit replaces the vertex (kSubsurfaceExit:
// albedo = the walk's weight, the exit's bounce origin and direction, the diffuse lobe with roughness 1,
// and `vertex_normal` = the exit normal); a dead walk sets the albedo to 0 (kSubsurfaceDead).
uint SubsurfaceVertex(CaelixFrameConstants fc, TraceResult hit, float3 ray_direction, inout uint rng,
                      inout CaelixShadeResult shade, inout float3 vertex_normal, out float3 guide_albedo,
                      bool last_vertex)
{
    guide_albedo = shade.albedo;
    const uint block = (hit.attrib >> 16) & 0xFFFFu;
    if (last_vertex || !IsOpaque(int(block)))
    {
        return kSubsurfaceNone;
    }
    const uint code = CaelixMaterialCode(block);
    const uint sss_word = CaelixMaterialSssWord(g_materialTable, code);
    const float probability = CaelixHalfLo(sss_word);
    if (((sss_word >> 16) & 0xFFu) != kSssModelHg || !(probability > 0.0f))
    {
        return kSubsurfaceNone;
    }
    // The entry material's subsurface entry (w0..w3): sss_albedo, anisotropy, steps.
    const uint4 entry = CAELIX_LOAD_U32x4(g_subsurfaceTable + uint64_t(code) * kSubsurfaceBytes);
    const uint steps = entry.z & 0xFFu;
    if (steps == 0u)
    {
        return kSubsurfaceNone;
    }
    const float3 entry_albedo = 1.0f - float3(CaelixHalfLo(entry.x), CaelixHalfHi(entry.x), CaelixHalfLo(entry.y));
    guide_albedo = lerp(shade.albedo, entry_albedo, probability);
    if (!(RandomFloat01(rng) < probability))
    {
        return kSubsurfaceBsdf;
    }
    const float g = clamp(CaelixHalfHi(entry.y), -kSubsurfaceMaxAnisotropy, kSubsurfaceMaxAnisotropy);
    float3 origin;
    float3 normal;
    float3 direction;
    float3 weight;
    if (!SubsurfaceWalk(fc, hit, ray_direction, g, steps, entry_albedo, rng, origin, normal, direction, weight))
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
