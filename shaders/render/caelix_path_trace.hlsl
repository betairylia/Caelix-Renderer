// Trace kernel: one primary ray per pixel through the scene's TLAS with an inline
// ray query; the brick DDA (caelix_brick_trace.hlsl) runs on every procedural
// candidate, as in Unity's CaelixRayQueryTrace.hlsl.
//
// Every mode writes the host's geometry targets that are present
// (CaelixFrameConstants::target_mask): albedo, normal, device depth, motion and
// linear view depth. With kRenderFlagFlipY every HOST target write goes to the
// mirrored row (CaelixOutputPixel) and the host motion's y is negated; the ray,
// the random stream and the internal targets use the unflipped pixel.
//   kModePrimary:   the colour target gets a debug view of the primary hit.
//   kModePathTrace, kModeReference: the path tracer of Unity's
//                   CaelixPathTraceCommon.hlsl (opaque surfaces): the primary hit
//                   is shaded (CaelixShadeVoxelHit), its emission goes to the
//                   deterministic target, and up to bounce_count stochastic
//                   segments gather the incident radiance into the diffuse OR
//                   the specular target (by the lobe chosen at the primary hit),
//                   packed for NRD REBLUR (REBLUR_FrontEnd_PackRadianceAndNormHitDist:
//                   YCoCg radiance, normalized hit distance of the first segment in
//                   alpha). The internal G-buffer (view_z, normal_roughness packed by
//                   NRD_FrontEnd_PackNormalAndRoughness, motion, albedo) is written
//                   for the composite and for NRD. The colour target is written by
//                   caelix_composite.hlsl.
//                   With kRenderFlagSunNee every diffuse vertex also samples the
//                   sun cone and traces a shadow ray (TracePath, SunNextEvent).
//                   Before the shading, the delta chain (RunDeltaChain, Unity's
//                   AdvanceDeltaRay loop) carries the camera ray through up to
//                   bounce_count_transparent smooth transparent interfaces
//                   (stochastic Fresnel choice between reflection and refraction,
//                   medium extinction); the surface where it stops replaces the
//                   primary hit (PSR): G-buffer, shading and emission, all times the
//                   chain's transmission, depth = chain length along the camera ray.
//                   Materials come from the scene's material table (GET_MATERIAL).
//                   Fog (fog_color.w > 0): every segment in the air that ends on
//                   a surface (the camera ray, the air gaps of the delta chain,
//                   every bounce segment) is attenuated by exp(-density t) and
//                   gains fog_color (1 - that) (TraverseSegment): the chain's in-
//                   scattering goes to the deterministic target, a bounce
//                   segment's to the stochastic radiance. Misses and shadow rays
//                   see no fog. Sky (sky_cubemap = 1): primary misses and the
//                   chain's sky end sample the host cubemap at mip 0, a specular
//                   bounce that misses at the mip of its roughness, a diffuse
//                   bounce that misses receives the irradiance SH / pi that
//                   caelix_sky_sh.hlsl projected before this dispatch
//                   (SkyDiffuseMiss, SkySpecularMiss).
//                   The internal device depth (binding 14, the host depth target's
//                   expression; the upscaler's depth) is written in every lighting
//                   frame; with fc.upscaler = kUpscalerDlssRr the four DLSS RR guides
//                   (bindings 15..18): diffuse albedo albedo * (1 - metallic) (sky 0.5),
//                   specular albedo EnvBRDFApprox2(lerp(0.04, albedo, metallic),
//                   roughness^2, N.V) (sky 0), world normal + linear roughness (sky 0) and
//                   the hit distance of the first bounce segment (sky and no segment 65504),
//                   all of the surface where the delta chain stopped, the albedos times the
//                   chain's transmission. A transparent surface (rough glass) gets diffuse 0
//                   and specular = the transmission. APPROXIMATION: the hit distance is the
//                   one of the lobe the path sampled (diffuse or specular), not always a
//                   specular segment.
//                   kGiFlagDiffuseOnly: every opaque vertex takes the diffuse lobe
//                   (g_forceDiffuse). ReSTIR GI (fc.gi_method = kGiRestir, render/
//                   caelix_restir_gi.hlsl): the diffuse lobe everywhere, no sun NEE and no fog
//                   on the path; the first bounce of the path is stored as the pixel's initial
//                   reservoir (fc.gi_initial) and the diffuse and specular radiance targets get
//                   0: caelix_restir_gi_shade.hlsl writes the diffuse one after the reuse passes.
//                   kGiFlagLightNee (docs/LIGHT_LIST.md, render/caelix_light_sample.hlsl): every
//                   diffuse vertex with a next segment also samples the scene's light list
//                   (LightNextEvent) and the segment that leaves it does not count the emission
//                   of a list light it hits when the NEE can sample that point (the brick has a
//                   table entry and the voxel's mask bit is set, LightListCanSample); in ReSTIR
//                   mode the primary vertex's light NEE goes to the diffuse target unpacked (the
//                   shade pass adds the reservoir to it).
//                   kGiFlagLightMis (docs/LIGHT_MIS.md, with kGiFlagLightNee): instead of that
//                   exclusive rule, the NEE and such a list-light hit are weighted by the balance
//                   heuristic over a density proxy of the NEE and the cosine density of the
//                   diffuse bounce (LightNeeMisWeight); the primary vertex keeps the exclusive
//                   rule in ReSTIR GI mode (other pixels reuse its BSDF sample).
//                   kGiFlagRestirDi (docs/RESTIR_DI.md, render/caelix_restir_di.hlsl; with
//                   kGiFlagLightNee): the primary vertex's light NEE is replaced by the pixel's
//                   initial DI reservoir (fc.di_initial, di_candidates candidates, no shadow ray),
//                   written for every opaque primary surface (the diffuse-lobe bit tells the DI
//                   shade pass whether to add its contribution); the first segment after a
//                   diffuse primary vertex still skips list-light emission that the NEE can
//                   sample, or with kGiFlagLightMis (not in ReSTIR GI mode) counts it with the
//                   BSDF's balance heuristic weight from the proxy context of the primary
//                   vertex's unjittered light cell, the context that DiInitialReservoir stores
//                   in the reservoir and the DI passes weight their NEE share with
//                   (docs/LIGHT_MIS.md section 5). The bounce vertices keep their light NEE.
//
// Inputs: the push constant is the device address of the frame constants; the
// scene's tables (page table, slot rows, group infos, entity infos) are read by
// caelix_scene_trace.hlsl. The descriptors are the trace set (render/caelix_trace_set.hlsl):
// the TLAS (binding 0), the six host targets (bindings 1..6, format Unknown), the internal
// targets of the lighting modes (bindings 7..18, fixed formats), the sky cubemap (binding 19,
// sampled image) and its sampler (binding 20). Every binding is PARTIALLY_BOUND:
// the library writes only the TLAS (when the scene has one), the present targets
// and, in the lighting modes, the internal targets, and the kernel touches a
// binding only then (has_tlas, target_mask, mode). Bindings 19 and 20 are written
// every frame (the host's cubemap, or a 1 x 1 fallback cube that is never sampled
// because sky_cubemap is 0).
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "render/caelix_scene_trace.hlsl"
#include "common/caelix_material.hlsl"
#include "common/caelix_light.hlsl"
// The push constant, the targets and the sky bindings (before caelix_shading.hlsl).
#include "render/caelix_trace_set.hlsl"
#include "render/caelix_shading.hlsl"
// NRD's front-end packing (normal + roughness, radiance + normalized hit distance);
// NRDConfig.hlsli next to it fixes the encodings NRD was built with.
#include "NRD.hlsli"
#include "render/caelix_restir_gi.hlsl"
#include "render/caelix_light_sample.hlsl"
#include "render/caelix_restir_di.hlsl"

static const float kRayTMax = 1e30f;
static const uint kStepsHeatMax = 64;
// Internal view_z of a miss (beyond any denoising range).
static const float kInternalMissViewZ = 1e30f;

// Blue -> cyan -> green -> yellow -> red for t in [0, 1].
float3 HeatMap(float t)
{
    t = saturate(t);
    return saturate(float3(1.5f - abs(4.0f * t - 3.0f), 1.5f - abs(4.0f * t - 2.0f),
                           1.5f - abs(4.0f * t - 1.0f)));
}

// The pixel (current target size) where the previous render call's camera saw
// the world point `p`, with the same pinhole mapping as the ray generation, and
// the point's view depth for that camera. Returns false when the point was not in
// front of that camera.
bool PreviousPixel(CaelixFrameConstants fc, float3 p, out float2 pixel, out float prev_view_z)
{
    const float3 rel = p - fc.prev_position.xyz;
    const float z = dot(rel, fc.prev_forward.xyz);
    pixel = float2(0.0f, 0.0f);
    prev_view_z = z;
    if (!(z > 1e-6f))
    {
        return false;
    }
    const float2 ndc = float2(dot(rel, fc.prev_right.xyz) / (z * fc.prev_tan_half_fov_y * fc.prev_aspect),
                              dot(rel, fc.prev_up.xyz) / (z * fc.prev_tan_half_fov_y));
    pixel = float2((ndc.x + 1.0f) * 0.5f * float(fc.size.x), (1.0f - ndc.y) * 0.5f * float(fc.size.y));
    return all(abs(pixel) < 1e30f);
}

// Sun next-event estimation at a diffuse path vertex with world normal `normal`
// and pushed-off position `origin`: one direction L uniform in the sun cone (pdf
// 1 / omega) and a shadow ray. Returns the demodulated incident radiance of the
// Lambert lobe, sun_radiance * omega * dot(N, L) / pi when L is unoccluded, else 0
// (the albedo of the vertex is in the caller's throughput, as for the bounces).
float3 SunNextEvent(CaelixFrameConstants fc, float3 origin, float3 normal, inout uint rng)
{
    const float3 L = SampleSunDirection(fc, rng);
    const float cos_l = dot(normal, L);
    if (!(cos_l > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    RayDesc shadow;
    shadow.Origin = origin;
    shadow.Direction = L;
    shadow.TMin = 0.0f;
    shadow.TMax = kRayTMax;
    if (TraceShadow(fc, shadow))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return fc.sun_radiance.xyz * (fc.sun_solid_angle * cos_l * (1.0f / K_PI));
}

// NRD lobe rule of a path that went through a delta interface (primary surface
// replacement). 0: every pixel whose camera ray crossed an interface goes to the
// specular target with roughness 0 (the transparent lobes are delta). 1: only a
// path whose FIRST interface reflected goes to the specular target (roughness 0);
// a refracted path keeps the lobe and roughness of the replaced surface (Unity's
// BOUNCEIDX_IS_SPECULAR rule, the usual PSR practice). Compile-time switch for the
// 6d measurement; the build uses 0.
#ifndef CAELIX_PSR_LOBE
#define CAELIX_PSR_LOBE 2
#endif

// The delta chain of the primary ray (Unity's primary loop + AdvanceDeltaRay):
// starting from the primary hit, every smooth transparent interface (smoothness >=
// 0.99, not opaque) is crossed by reflection or refraction (CrossInterface), up to
// `budget` interfaces, and every segment inside a medium is attenuated by its
// extinction. The first hit that is not crossed is the primary surface for
// everything after it (primary surface replacement).
struct DeltaChain
{
    TraceResult surface;   // the hit that ended the chain; hit = false: the chain ended in the sky
    TraceResult geometry;  // the G-buffer's hit: `surface`, or the last interface on a sky end
                           // (the primary miss when the camera ray hit nothing)
    RayDesc ray;           // the ray that found `surface`
    float3 throughput;     // transmission of the chain (extinction; k = 1 at every interface)
    float3 inscatter;      // fog emission gathered along the chain's air segments (radiance)
    float t_virtual;       // chain length up to `geometry` (PSR depth along the camera ray)
    uint medium;           // the medium `ray` travels in (face-free transparent id, 0 = air)
    uint interfaces;       // interfaces crossed
    bool first_reflected;  // the first interface reflected
    bool split;            // the delta checkerboard decided the first interface (cross resolve flag)
};

// `checkerboard`: the first interface takes the branch of the pixel's parity (even px + py
// refracts, odd reflects) with the weight 2 (1 - F) / 2 F (CrossInterface split mode), as
// Unity's AdvanceDeltaRay does; later interfaces stay random.
DeltaChain RunDeltaChain(CaelixFrameConstants fc, RayDesc ray, TraceResult primary, uint budget,
                         bool checkerboard, uint2 pixel, inout uint rng)
{
    DeltaChain c;
    c.surface = primary;
    c.geometry = primary;
    c.ray = ray;
    c.throughput = float3(1.0f, 1.0f, 1.0f);
    c.inscatter = float3(0.0f, 0.0f, 0.0f);
    c.t_virtual = 0.0f;
    c.medium = 0u;
    c.interfaces = 0u;
    c.first_reflected = false;
    c.split = false;

    CAELIX_LOOP
    for (;;)
    {
        if (!c.surface.hit)
        {
            break;
        }
        // The segment that found this hit ran inside c.medium (the air: fog).
        TraverseSegment(fc, c.medium, c.surface.T, c.throughput, c.inscatter);
        c.t_virtual += c.surface.T;
        c.geometry = c.surface;
        if (c.interfaces >= budget)
        {
            break;
        }
        const uint block = (c.surface.attrib >> 16) & 0xFFFFu;
        // Opaque surfaces always end the chain (Unity: smooth metals are a TODO there; the
        // owner chose parity, the specular lobe of the stochastic shading reflects them).
        if (IsOpaque(int(block)))
        {
            break;
        }
        const CaelixMaterial material = GET_MATERIAL(block);
        if (material.smoothness < 0.99f)
        {
            break;
        }
        const float3 hit_position = c.ray.Origin + c.ray.Direction * c.surface.T;
        int split = kInterfaceRandom;
        if (checkerboard && c.interfaces == 0u)
        {
            split = ((pixel.x + pixel.y) & 1u) == 0u ? kInterfaceSplitRefract : kInterfaceSplitReflect;
        }
        const CaelixInterface it = CrossInterface(c.medium, CaelixTransparentId(block), material.ior,
                                                  c.ray.Direction, c.surface.normal, split, rng);
        if (c.interfaces == 0u && !it.refracted)
        {
            c.first_reflected = true;
        }
        if (it.split_applied)
        {
            c.split = true;
            c.throughput *= it.weight;
        }
        c.interfaces += 1u;
        c.medium = it.medium;
        c.ray.Origin =
            hit_position + (it.refracted ? -K_RAY_ORIGIN_PUSH_OFF : K_RAY_ORIGIN_PUSH_OFF) * it.normal;
        c.ray.Direction = it.direction;
        c.ray.TMin = 0.0f;
        c.ray.TMax = kRayTMax;
        c.surface = TraceScene(fc, c.ray);
    }
    return c;
}

// The path tracer at the primary surface (Unity CaelixPathTraceMain, "Further
// Bounces"): shades the surface `surface` found by `ray` (the last ray of the delta
// chain) in medium `medium`, then traces up to bounce_count stochastic segments.
// Every segment is attenuated by the medium it runs in and gains the fog's emission
// in the air (TraverseSegment; a miss has T = 0). A segment that misses receives the
// sky of its lobe: SkyDiffuseMiss (the irradiance SH of a cubemap sky) after a
// diffuse vertex, SkySpecularMiss (the cubemap mip of the lobe's roughness) after a
// specular one. Returns the surface's shade result; `stochastic` receives (incident
// radiance, hit distance of the first segment). The incident radiance does not
// include the delta chain's throughput: the caller applies it to the albedo.
//
// Sun NEE (kRenderFlagSunNee with a nonzero sun radiance): every vertex that
// sampled the diffuse lobe AND has a next segment (the primary when
// bounce_count >= 1, a bounce hit while bounce_index <= bounce_count) adds
// SunNextEvent * throughput (throughput including that vertex's albedo, before
// Russian roulette). So NEE replaces exactly the sun contribution that the next
// BSDF segment would have found, and the expected value does not change. The
// segment that leaves such a vertex sees the sky without the sun disk; a segment
// that leaves a specular-lobe vertex (no NEE there, transparent vertices included)
// still sees the disk, so the sun is counted once on every path. Shadow rays treat
// glass as opaque. NEE draws from its own random stream, so the BSDF path consumes
// the same random numbers with NEE on or off.
//
// ReSTIR GI (`restir`): sun NEE is off and the segments see no fog (the GI path ignores
// both); the first segment's end is returned in `gi` as the initial sample, and its
// throughput restarts at 1 there, so the returned radiance is L_o(x_s) (the emission of the
// first bounce hit plus the continuation from it), or the sky of a first segment that misses.
// Without `restir` the path is the same as before (no extra random number, same order).
//
// Light NEE (kGiFlagLightNee with a light table): at the same vertices as the sun NEE
// (diffuse lobe, a next segment exists) LightNextEvent * throughput is added, from its own
// random stream; the segment that leaves such a vertex skips the emission of a list light it
// hits (luminance of the emission >= fc.light_min_emission, at least the near-field distance
// away) when the NEE can sample that point (LightListCanSample: the brick has a table entry and
// the voxel's mask bit is set), so NEE replaces exactly that emission. A point the NEE cannot
// sample (a brick dropped beyond the table's capacity, a voxel whose material changed) keeps its
// emission: before step 3 (docs/LIGHT_MIS.md) such light was lost. A segment that leaves a vertex
// without light NEE (a specular lobe) still counts every emission.
// Light MIS (kGiFlagLightMis with light NEE): the vertex's NEE is multiplied by the balance
// heuristic weight w_nee and a list-light hit of the segment that leaves it (one the NEE can
// sample, beyond the near field) counts its emission times w_bsdf = 1 - w_nee of that point; both
// weights come from LightNeeMisWeight with the vertex's proxy context (kept in segment_ctx with
// segment_x_r and segment_n_r), so the two strategies sum to the full emission at every point. The
// primary vertex keeps the exclusive rule (segment_mis false) with `restir` (ReSTIR GI): its BSDF
// sample is reused by other pixels, so a pixel-dependent weight would bias the reuse.
// `primary_light` receives the primary vertex's light NEE; with `restir` it is not in the
// returned radiance (the caller writes it to the diffuse target, the shade pass adds it).
// `restir_di` (ReSTIR DI): the primary vertex draws no light NEE (primary_light stays 0, the DI
// passes add the direct light of the list lights there), but its first segment still skips the
// emission of a list light it hits (one the NEE can sample) when the vertex took the diffuse lobe.
// With light MIS (and without `restir`) that segment counts such a hit with w_bsdf instead: the
// DI is the NEE strategy there (docs/LIGHT_MIS.md section 5). Its proxy context comes from
// LightNeeProxyAt at the unjittered cell of the vertex; DiInitialReservoir (in main, later in the
// same dispatch) looks the same cell up, gets the same slot and C_prev and stores that context in
// the reservoir, so both strategies use one context. segment_n_r is then the axis normal of the
// visible face (GiFaceNormal), the n_v of the DI passes, so that both sides also evaluate p_bsdf
// with the same normal (it differs from surface.normal only on rotated entities).
struct GiInitialSample
{
    bool valid;            // the path had a first segment (bounce_count >= 1)
    bool sky;              // the first segment missed: sample_pos is its direction
    float3 sample_pos;     // x_s: the first bounce hit, or the direction of a sky sample
    float3 sample_normal;  // n_s (a hit sample)
    float3 radiance;       // L_o(x_s -> x_v)
    float cos_theta_v;     // max(0, dot(n_v, first bounce direction))
};

// The entity-local point of a hit for its light cell: a quarter voxel into the air along the face
// normal, so that the cell is the one of the space the vertex sees.
float3 LightCellPoint(TraceResult hit)
{
    return hit.object_hit + 0.25f * float3(UnpackObjectNormal(hit.attrib));
}

CaelixShadeResult TracePath(CaelixFrameConstants fc, uint2 pixel, RayDesc ray, TraceResult surface,
                            uint medium, bool restir, bool restir_di, inout uint rng, out float4 stochastic,
                            out GiInitialSample gi, out float3 primary_light)
{
    gi.valid = false;
    gi.sky = false;
    gi.sample_pos = float3(0.0f, 0.0f, 0.0f);
    gi.sample_normal = float3(0.0f, 0.0f, 0.0f);
    gi.radiance = float3(0.0f, 0.0f, 0.0f);
    gi.cos_theta_v = 0.0f;
    uint nee_rng = InitRandom(pixel, fc.frame_seed ^ 0x68E31DA4u);
    const bool nee = !restir && (fc.flags & kRenderFlagSunNee) != 0u && any(fc.sun_radiance.xyz > 0.0f);
    uint light_rng = InitRandom(pixel, fc.frame_seed ^ 0x7F4A7C15u);
    const bool light_nee = (fc.gi_flags & kGiFlagLightNee) != 0u && fc.light_table != 0u;
    const bool light_mis = light_nee && (fc.gi_flags & kGiFlagLightMis) != 0u;
    primary_light = float3(0.0f, 0.0f, 0.0f);
    const uint surface_block = (surface.attrib >> 16) & 0xFFFFu;
    const CaelixShadeResult primary_shade =
        ShadeVoxelHit(surface_block, ray.Origin + ray.Direction * surface.T, ray.Direction,
                      surface.normal, rng, medium);

    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    float3 throughput = 1.0f / max(0.001f, primary_shade.k);
    // Hit distance of the first stochastic segment (primary surface -> first bounce
    // hit) in world units; t_max when it reaches the sky, 0 when there is no segment
    // (bounce_count 0).
    float first_segment_t = fc.bounce_count >= 1u ? fc.t_max : 0.0f;
    bool first_segment = true;

    // The lobe of the current segment (the sky it receives on a miss).
    float3 segment_normal = surface.normal;
    bool segment_diffuse = !primary_shade.specular_lobe;
    float segment_roughness = primary_shade.roughness;

    // NEE at the primary vertex (demodulated: throughput 1 / k).
    bool segment_sees_sun = true;
    CAELIX_BRANCH
    if (nee && !primary_shade.specular_lobe && fc.bounce_count >= 1u)
    {
        radiance += SunNextEvent(fc, primary_shade.bounce_origin, surface.normal, nee_rng) * throughput;
        segment_sees_sun = false;
    }
    // Light NEE at the primary vertex. The state of the segment that leaves a light NEE vertex:
    // segment_mis = the MIS of kGiFlagLightMis applies to its list-light hits, with the vertex's
    // receiver (segment_x_r, segment_n_r) and proxy context (segment_ctx).
    bool segment_light_nee = false;
    bool segment_mis = false;
    float3 segment_x_r = float3(0.0f, 0.0f, 0.0f);
    float3 segment_n_r = float3(0.0f, 0.0f, 0.0f);
    LightNeeProxyContext segment_ctx = LightNeeProxyOfSlot(fc, -1, 0u, 0u, int3(0, 0, 0), 0.0f);
    CAELIX_BRANCH
    if (light_nee && !primary_shade.specular_lobe && fc.bounce_count >= 1u)
    {
        if (!restir_di)
        {
            // ReSTIR GI reuses the primary vertex's BSDF sample: the exclusive rule there.
            segment_mis = light_mis && !restir;
            primary_light = LightNextEvent(fc, primary_shade.bounce_origin, surface.normal, surface.group,
                                           LightCellPoint(surface), segment_mis, light_rng, segment_ctx) *
                            throughput;
            if (!restir)
            {
                radiance += primary_light;
            }
        }
        else if (light_mis && !restir)
        {
            // The DI's receiver: the pushed-off primary point, its face's axis normal and the
            // unjittered cell's context (DiInitialReservoir stores the same one).
            segment_mis = true;
            segment_ctx = LightNeeProxyAt(fc, primary_shade.bounce_origin, surface.group, LightCellPoint(surface));
        }
        segment_light_nee = true;
        segment_x_r = primary_shade.bounce_origin;
        segment_n_r = segment_mis && restir_di ? GiFaceNormal(GiFaceIndex(surface.normal)) : surface.normal;
    }

    RayDesc bounce;
    bounce.Origin = primary_shade.bounce_origin;
    bounce.Direction = primary_shade.bounce_direction;
    bounce.TMin = 0.0f;
    bounce.TMax = fc.t_max;
    if (restir)
    {
        gi.cos_theta_v = max(0.0f, dot(surface.normal, bounce.Direction));
    }

    // Unity's stochastic bounce counter: 1 after the primary shade, +1 per shaded
    // bounce hit; the loop runs while it is <= bounce_count, so bounce_count
    // segments at most.
    uint bounce_index = 1;
    CAELIX_LOOP
    for (; bounce_index <= fc.bounce_count;)
    {
        const TraceResult hit = TraceScene(fc, bounce);
        // The segment ran inside `medium` (a miss has T = 0: the sky is not attenuated and
        // the fog adds nothing). The GI path ignores the fog.
        if (!restir)
        {
            TraverseSegment(fc, medium, hit.T, throughput, radiance);
        }
        if (!hit.hit)
        {
            if (restir && first_segment)
            {
                // A sky sample: the radiance below with throughput 1.
                gi.sky = true;
                gi.sample_pos = bounce.Direction;
                throughput = float3(1.0f, 1.0f, 1.0f);
            }
            radiance += (segment_diffuse
                             ? SkyDiffuseMiss(fc, segment_normal, bounce.Direction, segment_sees_sun)
                             : SkySpecularMiss(fc, bounce.Direction, segment_roughness, segment_sees_sun)) *
                        throughput;
            break;
        }
        if (first_segment)
        {
            first_segment_t = hit.T;
            first_segment = false;
            if (restir)
            {
                // The sample point: L_o(x_s) starts here with throughput 1.
                gi.sample_pos = bounce.Origin + bounce.Direction * hit.T;
                gi.sample_normal = hit.normal;
                throughput = float3(1.0f, 1.0f, 1.0f);
            }
        }
        const CaelixShadeResult shade =
            ShadeVoxelHit((hit.attrib >> 16) & 0xFFFFu, bounce.Origin + bounce.Direction * hit.T,
                          bounce.Direction, hit.normal, rng, medium);
        bounce_index += 1;

        // A list light's emission belongs to the light NEE of the vertex the segment left when the
        // NEE can sample the point (LightListCanSample), except when the hit is closer than the
        // near-field split (the near field is the BSDF path's). With MIS at that vertex the
        // emission counts with the BSDF's balance heuristic weight instead of 0.
        float emission_weight = 1.0f;
        CAELIX_BRANCH
        if (segment_light_nee && shade.emission_luminance >= fc.light_min_emission &&
            hit.T >= g_lightNeeMinDistance)
        {
            LightEntry light_entry;
            uint3 light_voxel;
            uint light_face;
            if (LightListCanSample(fc, hit, light_entry, light_voxel, light_face))
            {
                emission_weight = 0.0f;
                if (segment_mis)
                {
                    const LightMisDesc desc = LightMisDescOf(fc, light_entry, light_voxel);
                    const float cos_l = -dot(hit.normal, bounce.Direction);
                    emission_weight = LightNeeMisWeight(segment_ctx, desc, segment_x_r, segment_n_r,
                                                        bounce.Direction, hit.T, cos_l, false);
                }
            }
        }
        if (emission_weight > 0.0f)
        {
            radiance += shade.emission * (throughput * emission_weight);
        }
        throughput *= shade.albedo / max(0.001f, shade.k);

        segment_sees_sun = true;
        CAELIX_BRANCH
        if (nee && !shade.specular_lobe && bounce_index <= fc.bounce_count)
        {
            radiance += SunNextEvent(fc, shade.bounce_origin, hit.normal, nee_rng) * throughput;
            segment_sees_sun = false;
        }
        segment_light_nee = false;
        segment_mis = false;
        CAELIX_BRANCH
        if (light_nee && !shade.specular_lobe && bounce_index <= fc.bounce_count)
        {
            radiance += LightNextEvent(fc, shade.bounce_origin, hit.normal, hit.group, LightCellPoint(hit),
                                       light_mis, light_rng, segment_ctx) *
                        throughput;
            segment_light_nee = true;
            segment_mis = light_mis;
            segment_x_r = shade.bounce_origin;
            segment_n_r = hit.normal;
        }

        // Russian roulette: dark paths stop early. A zero throughput must stop here,
        // or 0 * (1 / 0) below gives NaN.
        const float continue_probability = max(throughput.r, max(throughput.g, throughput.b));
        if (continue_probability <= 0.0f || continue_probability < RandomFloat01(rng))
        {
            break;
        }
        throughput *= 1.0f / continue_probability;

        bounce.Origin = shade.bounce_origin;
        bounce.Direction = shade.bounce_direction;
        segment_normal = hit.normal;
        segment_diffuse = !shade.specular_lobe;
        segment_roughness = shade.roughness;
    }
    stochastic = float4(SanitizeRadiance(radiance), first_segment_t);
    if (restir)
    {
        gi.radiance = radiance;
        gi.valid = fc.bounce_count >= 1u;
    }
    return primary_shade;
}

// The material debug view (kViewMaterial): the table's albedo, mixed with blue for
// a transparent code and with white for an emissive entry; air is magenta.
float3 MaterialViewColor(uint block)
{
    const CaelixMaterial m = GET_MATERIAL(block);
    if (!IsOpaque(int(block)))
    {
        return CaelixTransparentId(block) == 0u ? kAirColor : lerp(m.albedo, float3(0.2f, 0.45f, 1.0f), 0.5f);
    }
    return any(m.emission > 0.0f) ? lerp(m.albedo, float3(1.0f, 1.0f, 1.0f), 0.5f) : m.albedo;
}

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
    if (any(thread_id.xy >= fc.size))
    {
        return;
    }
    g_blockEncoding = fc.encoding;
    g_materialTable = fc.materials;
    g_emissionScale = fc.emission_scale;
    g_lightNeeMinDistance = fc.light_nee_min_distance;
    const bool restir = fc.gi_method == kGiRestir;
    g_forceDiffuse = (fc.gi_flags & kGiFlagDiffuseOnly) != 0u || restir;
    // ReSTIR DI: the renderer sets the flag only with light NEE, a light table and the DI buffers.
    const bool restir_di = (fc.gi_flags & kGiFlagRestirDi) != 0u && fc.di_initial != 0u;

    // Pixel -> ray. Row 0 is the top row of the image.
    RayDesc ray;
    ray.Origin = fc.position.xyz;
    ray.Direction = CaelixPrimaryDirection(fc, thread_id.xy);
    ray.TMin = 0.0f;
    ray.TMax = kRayTMax;

    const TraceResult primary = TraceScene(fc, ray);
    const bool hit = primary.hit;
    const uint targets = fc.target_mask;
    const bool lighting = fc.mode != kModePrimary;
    // The host targets' pixel (kRenderFlagFlipY mirrors the rows).
    const uint2 out_pixel = CaelixOutputPixel(fc, thread_id.xy);
    const bool flip_y = (fc.flags & kRenderFlagFlipY) != 0u;

    // The delta chain (lighting modes; the primary mode and budget 0 leave the primary
    // hit as the surface without drawing a random number, so an opaque scene consumes
    // the random stream exactly as before the chain existed).
    uint rng = InitRandom(thread_id.xy, fc.frame_seed);
    // The checkerboard runs in the path trace mode only: the reference mode needs an unbiased
    // estimate per pixel, which a fixed per-pixel branch is not.
    const bool checkerboard = fc.mode == kModePathTrace && (fc.flags & kRenderFlagDeltaCheckerboard) != 0u;
    const DeltaChain chain = RunDeltaChain(fc, ray, primary, lighting ? fc.bounce_count_transparent : 0u,
                                           checkerboard, thread_id.xy, rng);
    const float split_flag = chain.split ? 1.0f : 0.0f;
    // The G-buffer describes chain.geometry at the chain length t_virtual along the
    // camera ray (PSR). Without an interface it is the primary hit at primary.T.
    const TraceResult geometry = chain.geometry;
    const float T = chain.t_virtual;
    const float3 normal = geometry.normal;
    const uint block = (geometry.attrib >> 16) & 0xFFFFu;
    const float3 albedo = hit ? BlockAlbedo(block) : float3(0.0f, 0.0f, 0.0f);

    CAELIX_BRANCH
    if (!lighting && (targets & kTargetColor) != 0u)
    {
        float3 color = fc.sky_color.xyz;
        if (fc.debug_view == kViewSteps)
        {
            color = primary.candidates == 0 ? fc.sky_color.xyz
                                            : HeatMap(float(primary.candidates) / float(kStepsHeatMax));
        }
        else if (hit)
        {
            if (fc.debug_view == kViewNormal)
            {
                color = normal * 0.5f + 0.5f;
            }
            else if (fc.debug_view == kViewDepth)
            {
                color = saturate((T - fc.near_z) / max(fc.far_z - fc.near_z, 1e-6f)).xxx;
            }
            else if (fc.debug_view == kViewGroup)
            {
                color = Pastel(HashU32(primary.group + 0x632BE5ABu));
            }
            else if (fc.debug_view == kViewBrick)
            {
                color = Pastel(HashU32(HashU32(primary.group + 0x632BE5ABu) ^ (primary.id + 0x2545F491u)));
            }
            else
            {
                color = fc.debug_view == kViewMaterial ? MaterialViewColor(block) : albedo;
                if ((fc.flags & kRenderFlagShading) != 0u)
                {
                    color *= 0.25f + 0.75f * saturate(dot(normal, fc.sun_dir.xyz));
                }
            }
        }
        // Opaque everywhere, so that a host can present the image directly.
        g_color[out_pixel] = float4(color, 1.0f);
    }

    const float coverage = hit ? 1.0f : 0.0f;
    CAELIX_BRANCH
    if ((targets & kTargetAlbedo) != 0u)
    {
        g_albedo[out_pixel] = float4(albedo, coverage);
    }
    CAELIX_BRANCH
    if ((targets & kTargetNormal) != 0u)
    {
        const float3 stored = (hit && (targets & kTargetNormalUnorm) != 0u) ? normal * 0.5f + 0.5f : normal;
        g_normal[out_pixel] = float4(stored, coverage);
    }

    // View space of the current camera: the (virtual) hit is position + direction * T.
    const float3 rel = ray.Direction * T;
    const float view_z = dot(rel, fc.forward.xyz);
    // Device depth of the host projection: the depth target and the internal depth (upscaler).
    float device_depth = fc.miss_depth;
    if (hit)
    {
        const float4 view = float4(dot(rel, fc.right.xyz), dot(rel, fc.up.xyz), view_z, 1.0f);
        device_depth = dot(fc.projection[2], view) / dot(fc.projection[3], view);
    }
    CAELIX_BRANCH
    if ((targets & kTargetDepth) != 0u)
    {
        g_depth[out_pixel] = float4(device_depth, 0.0f, 0.0f, 0.0f);
    }
    CAELIX_BRANCH
    if ((targets & kTargetViewZ) != 0u)
    {
        g_view_z[out_pixel] = float4(hit ? view_z : fc.miss_view_z, 0.0f, 0.0f, 0.0f);
    }

    // Motion: the host target and the internal one of the lighting modes.
    float2 motion = float2(0.0f, 0.0f);
    float motion_view_z = 0.0f;  // previous view z - current view z
    CAELIX_BRANCH
    if (((targets & kTargetMotion) != 0u || lighting) && hit && fc.has_prev_camera != 0u)
    {
        // The hit point under the entity's previous transform.
        float3 prev_world = EntityPrevWorld(fc, geometry.group, geometry.object_hit);
        CAELIX_BRANCH
        if (chain.interfaces > 0u)
        {
            // PSR (Unity): the virtual point on the camera ray, moved by the replaced
            // surface's own motion (its previous minus its current world position).
            const float3 cur_world = EntityCurWorld(fc, geometry.group, geometry.object_hit);
            prev_world = fc.position.xyz + rel + (prev_world - cur_world);
        }
        float2 prev_pixel;
        float prev_view_z;
        if (PreviousPixel(fc, prev_world, prev_pixel, prev_view_z))
        {
            // The current pixel is the jittered sample position, where the current camera's
            // unjittered pinhole sees the hit; the previous pixel uses the previous camera's
            // unjittered pinhole. So the motion is the surface's own motion, free of jitter.
            motion = prev_pixel - CaelixSamplePosition(fc, thread_id.xy);
            motion_view_z = prev_view_z - view_z;
        }
    }
    CAELIX_BRANCH
    if ((targets & kTargetMotion) != 0u)
    {
        // w: the delta checkerboard flag (RGBA formats only; RG formats drop z and w).
        // kRenderFlagFlipY: y grows upward in the flipped image, so its motion y is negated.
        g_motion[out_pixel] = float4(motion.x, flip_y ? -motion.y : motion.y, 0.0f, split_flag);
    }

    CAELIX_BRANCH
    if (lighting)
    {
        const uint2 pixel = thread_id.xy;
        // A chain that ended in the sky has no surface: NRD sees a miss (view z 1e30, no
        // radiance) and the composite shows the deterministic sky radiance.
        const bool surface = chain.surface.hit;
        g_int_view_z[pixel] = float4(surface ? view_z : kInternalMissViewZ, 0.0f, 0.0f, 0.0f);
        // w = the checkerboard flag: the composite (albedo scale) and the cross resolve read it.
        g_int_motion[pixel] = float4(motion, motion_view_z, split_flag);

        float3 material_albedo = float3(0.0f, 0.0f, 0.0f);
        float3 deterministic = float3(0.0f, 0.0f, 0.0f);
        // NRD packing. The lobe that the path did not sample gets 0 radiance and 0 hit
        // distance (NRD's "invalid" sample, reconstructed by HitDistanceReconstructionMode);
        // a miss gets 0 in both (the composite draws the sky; view z 1e30 is outside the
        // denoising range). The hit distance is normalized with REBLUR's A, B, C at the
        // (virtual) view z: roughness 1 for the diffuse lobe, the NRD roughness for the
        // specular one.
        float4 diffuse = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 specular = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 normal_roughness = float4(0.0f, 0.0f, 0.0f, 0.0f);
        // DLSS RR guides (fc.upscaler = kUpscalerDlssRr), with the sky values of the RR guide.
        const bool rr_guides = fc.upscaler == kUpscalerDlssRr;
        float3 rr_diffuse_albedo = float3(0.5f, 0.5f, 0.5f);
        float3 rr_specular_albedo = float3(0.0f, 0.0f, 0.0f);
        float4 rr_normal_roughness = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float rr_hit_distance = kHalfMax;

        // ReSTIR GI: the pixel's initial reservoir (empty without a surface or a usable sample).
        GiReservoir gi_reservoir = GiEmptyReservoir();
        // ReSTIR DI: the pixel's initial DI reservoir (empty without an opaque surface or bounces).
        DiReservoir di_reservoir = DiEmptyReservoir();

        CAELIX_BRANCH
        if (surface)
        {
            float4 stochastic;
            GiInitialSample gi_sample;
            float3 primary_light;
            const CaelixShadeResult shade = TracePath(fc, pixel, chain.ray, chain.surface, chain.medium, restir,
                                                      restir_di, rng, stochastic, gi_sample, primary_light);
            CAELIX_BRANCH
            if (restir_di && fc.bounce_count >= 1u && IsOpaque(int((chain.surface.attrib >> 16) & 0xFFFFu)))
            {
                // x_v and the face as for the step 1 NEE of TracePath (the pushed-off primary point).
                di_reservoir = DiInitialReservoir(fc, pixel, shade.bounce_origin, GiFaceIndex(chain.surface.normal),
                                                  !shade.specular_lobe, chain.surface.group,
                                                  LightCellPoint(chain.surface));
            }
            // The chain's transmission modulates everything the surface sends (Unity:
            // AlbedoTarget = throughput * albedo, deterministic = throughput * emission).
            material_albedo = chain.throughput * shade.albedo;
            // A split pixel's throughput holds the checkerboard weight (up to 2); the RGBA8 albedo
            // target stores half of it and the composite doubles flagged pixels again.
            if (chain.split)
            {
                material_albedo *= 0.5f;
            }
            // Primary emission is deterministic: nothing was sampled to obtain it; so is the
            // fog's emission along the chain.
            deterministic = chain.throughput * shade.emission + chain.inscatter;
            bool specular_target = shade.specular_lobe;
            float nrd_roughness = shade.roughness;
#if CAELIX_PSR_LOBE == 0
            if (chain.interfaces > 0u)
            {
                specular_target = true;
                nrd_roughness = 0.0f;
            }
#elif CAELIX_PSR_LOBE == 1
            if (chain.first_reflected)
            {
                specular_target = true;
                nrd_roughness = 0.0f;
            }
#endif
            normal_roughness = NRD_FrontEnd_PackNormalAndRoughness(normal, nrd_roughness, 0.0f);
            const float lobe_roughness = specular_target ? nrd_roughness : 1.0f;
            const float norm_hit_dist =
                REBLUR_FrontEnd_GetNormHitDist(stochastic.a, view_z, fc.hit_dist_params.xyz, lobe_roughness);
            const float4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(stochastic.rgb, norm_hit_dist, true);
            if (specular_target)
            {
                specular = packed;
            }
            else
            {
                diffuse = packed;
            }

            CAELIX_BRANCH
            if (restir)
            {
                // The first bounce was cosine-sampled (pdf cos / pi), so W = pi / cos and
                // L_o * cos / pi * W = L_o, the path tracer's estimate. A sample with cos 0
                // (a refracted first bounce of rough glass, a grazing direction) is worthless
                // and stays invalid. The shade pass writes the diffuse radiance.
                const float W = gi_sample.cos_theta_v > 0.0f ? K_PI / gi_sample.cos_theta_v : 0.0f;
                if (gi_sample.valid && W > 0.0f)
                {
                    gi_reservoir.sample_pos = gi_sample.sample_pos;
                    gi_reservoir.sample_flags =
                        (gi_sample.sky ? kGiSampleSky : GiFaceIndex(gi_sample.sample_normal)) | kGiSampleValid;
                    gi_reservoir.radiance = SanitizeRadiance(gi_sample.radiance);
                    gi_reservoir.W = W;
                    gi_reservoir.visible_pos = shade.bounce_origin;
                    gi_reservoir.visible_flags_m = GiFaceIndex(normal) | (1u << kGiMShift);
                }
                // The primary vertex's light NEE (0 without the flag), unpacked: the shade pass adds
                // the reservoir's contribution to it and packs the sum.
                diffuse = float4(SanitizeRadiance(primary_light), 0.0f);
                specular = float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            CAELIX_BRANCH
            if (rr_guides)
            {
                const uint surface_block = (chain.surface.attrib >> 16) & 0xFFFFu;
                const CaelixMaterial material = GET_MATERIAL(surface_block);
                // The RGBA8 guides hold [0, 1]; the chain's transmission is at most 1 without the
                // checkerboard (RR forces it off).
                const float3 transmission = saturate(chain.throughput);
                const float3 surface_normal = chain.surface.normal;
                if (IsOpaque(int(surface_block)))
                {
                    const float roughness = 1.0f - material.smoothness;
                    const float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), material.albedo, material.metallic);
                    const float NoV = dot(surface_normal, -chain.ray.Direction);
                    rr_diffuse_albedo = transmission * material.albedo * (1.0f - material.metallic);
                    rr_specular_albedo = transmission * EnvBRDFApprox2(f0, roughness * roughness, NoV);
                    rr_normal_roughness = float4(surface_normal, roughness);
                }
                else
                {
                    // Rough glass (shaded stochastically): all of its light is specular.
                    rr_diffuse_albedo = float3(0.0f, 0.0f, 0.0f);
                    rr_specular_albedo = transmission;
                    rr_normal_roughness = float4(surface_normal, 1.0f - material.smoothness);
                }
                rr_hit_distance = (stochastic.a > 0.0f && stochastic.a < fc.t_max) ? stochastic.a : kHalfMax;
            }
        }
        else if (hit)
        {
            // The chain went through at least one interface and left the scene: the sky in
            // the last direction times the transmission (Unity: delta miss).
            deterministic = chain.throughput * SkyRadiance(fc, chain.ray.Direction) + chain.inscatter;
        }
        CAELIX_BRANCH
        if (restir)
        {
            StoreGiReservoir(GiReservoirAddress(fc.gi_initial, fc, pixel), gi_reservoir);
        }
        CAELIX_BRANCH
        if (restir_di)
        {
            StoreDiReservoir(DiReservoirAddress(fc.di_initial, fc, pixel), di_reservoir);
        }
        g_int_normal_roughness[pixel] = normal_roughness;
        // Coverage is albedo.a.
        g_int_albedo[pixel] = float4(material_albedo, coverage);
        g_int_deterministic[pixel] = float4(SanitizeRadiance(deterministic), coverage);
        g_int_diff[pixel] = diffuse;
        g_int_spec[pixel] = specular;
        g_int_depth[pixel] = float4(device_depth, 0.0f, 0.0f, 0.0f);
        CAELIX_BRANCH
        if (rr_guides)
        {
            g_int_rr_diffuse_albedo[pixel] = float4(saturate(rr_diffuse_albedo), 1.0f);
            g_int_rr_specular_albedo[pixel] = float4(saturate(rr_specular_albedo), 1.0f);
            g_int_rr_normal_roughness[pixel] = rr_normal_roughness;
            g_int_rr_hit_dist[pixel] = float4(rr_hit_distance, 0.0f, 0.0f, 0.0f);
        }
    }
}
