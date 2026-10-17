// Composite kernel of the lighting modes (kModePathTrace, kModeReference): turns
// the trace kernel's internal targets into the host's colour target.
//
//   diffuse, specular = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(binding 3, 4).rgb
//     (the renderer binds NRD's outputs when the frame was denoised, else the trace
//     kernel's raw REBLUR-packed radiance; both are YCoCg)
//   raw = deterministic + albedo * (diffuse + specular)      (incident radiance, demodulated)
//   kModeReference: accum = n == 0 ? raw : lerp(accum, raw, 1 / (n + 1)), output accum
//                   (n = accum_count, frames accumulated since the last reset)
//   kModePathTrace: output raw; debug view kViewNrdDiff outputs diffuse instead,
//                   kViewNrdSpec specular
//   colour = output * exposure; a miss is SkyRadiance(primary direction) * exposure
//   (the host's cubemap at mip 0 when it gave one, else the procedural sky; plus the
//   sun disk) with kRenderFlagSkyOnMiss, else 0; alpha = coverage; linear -> sRGB when the
//   colour target is a UNORM format (output_srgb).
//   kModePathTrace, debug view kViewNrdValidation, denoised frame: NRD's validation
//   layer (binding 6, RGBA8, a = opacity) is blended over the encoded colour.
//   kModePathTrace, debug view kViewMotion: the motion of the primary hit (binding 7,
//   pixels) as rgb = (saturate(motion.xy / 32 * 0.5 + 0.5), 0.5 * coverage), alpha =
//   coverage, without exposure and without the sRGB encode: a hit without motion is
//   grey (0.5, 0.5, 0.5), a miss (0.5, 0.5, 0).
//
//   kRenderFlagFlipY: the host colour target is written at the mirrored row
//   (CaelixOutputPixel), and the motion view shows the motion with y negated (the host
//   motion target's convention); every internal target is read at the unflipped pixel.
//
//   Delta checkerboard (motion.w = 1): the trace stored half the albedo (the split
//   weight can reach 2); the composite doubles it.
//   fc.cross_resolve (path trace mode with kRenderFlagDeltaCheckerboard): the output
//   (after exposure, before the sRGB encode and the validation overlay; the motion view
//   as it is) goes to the resolve image, and caelix_cross_resolve.hlsl writes the host
//   target.
//
// Descriptors: render/caelix_composite_set.hlsl.
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_block.hlsl"
#include "common/caelix_material.hlsl"
// Before the shading helpers: the set defines the sky bindings (9, 10) they declare.
#include "render/caelix_composite_set.hlsl"
#include "render/caelix_shading.hlsl"
#include "NRD.hlsli"

// kViewMotion: motion (pixels) that maps to full red or green.
static const float kMotionViewRange = 32.0f;

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
    if (any(thread_id.xy >= fc.size))
    {
        return;
    }
    g_emissionScale = fc.emission_scale;  // not used here; keeps the shading globals set
    const uint2 pixel = thread_id.xy;
    float4 albedo = g_int_albedo[pixel];
    const bool hit = albedo.a > 0.5f;
    const float4 motion_flag = g_int_motion[pixel];
    if (motion_flag.w > 0.5f)
    {
        albedo.rgb *= 2.0f;  // the trace stored half of a checkerboard pixel's albedo
    }
    const bool path_trace = fc.mode == kModePathTrace;
    CAELIX_BRANCH
    if (path_trace && fc.debug_view == kViewMotion)
    {
        const float2 motion =
            (fc.flags & kRenderFlagFlipY) != 0u ? float2(motion_flag.x, -motion_flag.y) : motion_flag.xy;
        const float coverage = hit ? 1.0f : 0.0f;
        const float2 shown = saturate(motion / kMotionViewRange * 0.5f + 0.5f);
        const float4 value = float4(shown, 0.5f * coverage, coverage);
        if (fc.cross_resolve != 0u)
        {
            g_int_resolve[pixel] = value;  // passed through by the cross resolve
        }
        else
        {
            g_color[CaelixOutputPixel(fc, pixel)] = value;
        }
        return;
    }

    float3 color = float3(0.0f, 0.0f, 0.0f);
    if (hit)
    {
        const float3 diffuse = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(g_int_diff[pixel]).rgb;
        const float3 specular = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(g_int_spec[pixel]).rgb;
        const float3 raw = g_int_deterministic[pixel].rgb + albedo.rgb * (diffuse + specular);
        color = raw;
        if (path_trace && fc.debug_view == kViewNrdDiff)
        {
            color = diffuse;
        }
        else if (path_trace && fc.debug_view == kViewNrdSpec)
        {
            color = specular;
        }
        CAELIX_BRANCH
        if (fc.mode == kModeReference)
        {
            // The first frame after a reset is exactly the raw frame.
            const float3 mean = fc.accum_count == 0u
                                    ? raw
                                    : lerp(g_int_accum[pixel].rgb, raw, 1.0f / float(fc.accum_count + 1u));
            g_int_accum[pixel] = float4(mean, 1.0f);
            color = mean;
        }
    }
    else
    {
        CAELIX_BRANCH
        if (fc.mode == kModeReference)
        {
            g_int_accum[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if ((fc.flags & kRenderFlagSkyOnMiss) != 0u)
        {
            color = SkyRadiance(fc, CaelixPrimaryDirection(fc, pixel));
        }
    }
    color *= fc.exposure;
    CAELIX_BRANCH
    if (fc.cross_resolve != 0u)
    {
        // Linear: the cross resolve blends, then encodes.
        g_int_resolve[pixel] = float4(color, hit ? 1.0f : 0.0f);
        return;
    }
    g_color[CaelixOutputPixel(fc, pixel)] = float4(CaelixOutputColor(fc, pixel, color), hit ? 1.0f : 0.0f);
}
