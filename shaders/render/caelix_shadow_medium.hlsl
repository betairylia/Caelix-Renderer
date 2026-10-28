// The material rules of the transmissive shadow ray (docs/SHADOW_TRANSMISSION.md section 1): the
// transmittance of a straight shadow ray through transparent media and subsurface-scattering (SSS)
// voxels, and its SSS budget. render/caelix_scene_trace.hlsl (TraceShadowTransmissive) applies them per
// segment. The materials come from the scene's tables (common/caelix_material.hlsl: g_materialTable,
// g_subsurfaceTable, set by the kernel).
//   medium of transparent id g != 0, segment of length t: exp(-(1 - albedo) extinction t) (the formula
//     of MediumExtinction; the air's fog is ignored: shadow rays ignore fog);
//   SSS voxel (an opaque code with sss_model kSssModelHg and sss_probability > 0): entering an SSS run
//     multiplies by sss_probability once; a segment of length l multiplies by sss_albedo^l and adds
//     l / shadow_length to the budget; at a budget of 1 the ray dies;
//   a transmittance below kShadowTransmitCutoff (max channel) is occluded.
// The SSS rules exist only in the kernels compiled with CAELIX_SUBSURFACE 1 (the trace variants
// caelix_path_trace_sss*, docs/SUBSURFACE.md section 4); with the default 0 no voxel is an SSS voxel
// (ShadowVoxelIsSubsurface folds to false), so the main trace kernels carry no SSS code.
#ifndef CAELIX_SHADOW_MEDIUM_HLSL
#define CAELIX_SHADOW_MEDIUM_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_block.hlsl"
#include "common/caelix_material.hlsl"

// 1: the kernel carries the subsurface code (the SSS rules of the shadow ray here and the walk of
// render/caelix_subsurface.hlsl). The variant files define it before their include.
#ifndef CAELIX_SUBSURFACE
#define CAELIX_SUBSURFACE 0
#endif

// Max channel of the transmittance below which a shadow ray counts as occluded.
static const float kShadowTransmitCutoff = 1e-3f;
// The smallest SSS albedo of the attenuation (log2 of 0 is -inf; 1e-4 per voxel is black anyway).
static const float kShadowMinSssAlbedo = 1e-4f;

struct ShadowMedium
{
    float3 transmittance;
    float sss_budget;  // the distance travelled in SSS voxels over their shadow length; dead at 1
};

ShadowMedium ShadowMediumInit()
{
    ShadowMedium m;
    m.transmittance = float3(1.0f, 1.0f, 1.0f);
    m.sss_budget = 0.0f;
    return m;
}

// A segment of length t inside the transparent medium `medium_id` (a face-free transparent id; 0 = air:
// nothing).
void ShadowGlassSegment(inout ShadowMedium m, uint medium_id, float t)
{
    if (medium_id == 0u)
    {
        return;
    }
    const CaelixMaterial material = GET_MATERIAL(medium_id);
    m.transmittance *= exp(-(1.0f - material.albedo) * t * material.extinction);
}

// A segment of length l (voxel units) inside the SSS voxel of block value `block` (opaque): times
// sss_albedo^l, budget + l / shadow_length (a shadow length of 0 kills the ray). One 16-byte load of
// the subsurface entry (w0..w3).
void ShadowSssSegment(inout ShadowMedium m, uint block, float l)
{
    const uint4 w = CAELIX_LOAD_U32x4(g_subsurfaceTable + uint64_t(CaelixMaterialCode(block)) * kSubsurfaceBytes);
    const float3 albedo = 1.0f - float3(CaelixHalfLo(w.x), CaelixHalfHi(w.x), CaelixHalfLo(w.y));
    const float shadow_length = CaelixHalfHi(w.z);  // kSubsurfaceSteps word, high half
    m.transmittance *= exp2(log2(max(albedo, kShadowMinSssAlbedo)) * l);
    m.sss_budget += shadow_length > 0.0f ? l / shadow_length : 1.0f;
}

// Entering an SSS run (from air, glass or the ray's start) at the voxel of block value `block`: times its
// sss_probability (the part of the light that enters the medium in the walk model).
void ShadowSssEnter(inout ShadowMedium m, uint block)
{
    const uint w3 = CAELIX_LOAD_U32(g_subsurfaceTable + uint64_t(CaelixMaterialCode(block)) * kSubsurfaceBytes +
                                    kSubsurfaceProbability);
    m.transmittance *= CaelixHalfLo(w3);
}

// The block value is an SSS voxel: opaque, sss_model kSssModelHg, sss_probability > 0. One 4-byte load of
// the subsurface entry's w3 (the mirror of the hot entry's sss word). Always false (no load) in a kernel
// without CAELIX_SUBSURFACE.
bool ShadowVoxelIsSubsurface(uint block)
{
#if CAELIX_SUBSURFACE
    if (!IsOpaque(int(block)))
    {
        return false;
    }
    const uint w3 = CAELIX_LOAD_U32(g_subsurfaceTable + uint64_t(CaelixMaterialCode(block)) * kSubsurfaceBytes +
                                    kSubsurfaceProbability);
    return ((w3 >> 16) & 0xFFu) == kSssModelHg && CaelixHalfLo(w3) > 0.0f;
#else
    return false;
#endif
}

// The ray is dead: the SSS budget is used up or the transmittance is below the cutoff.
bool ShadowMediumDead(ShadowMedium m)
{
    return m.sss_budget >= 1.0f || max(m.transmittance.r, max(m.transmittance.g, m.transmittance.b)) < kShadowTransmitCutoff;
}

#endif  // CAELIX_SHADOW_MEDIUM_HLSL
