// Composite kernel of the lighting modes (kModePathTrace, kModeReference): turns
// the trace kernel's internal targets into the host's colour target.
//
//   diffuse, specular = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(binding 3, 4).rgb
//     (the renderer binds NRD's outputs when the frame was denoised, else the trace
//     kernel's raw REBLUR-packed radiance; both are YCoCg)
//   raw = deterministic + albedo * (diffuse + specular)      (incident radiance, demodulated)
//   kModeReference: accum = n == 0 ? raw : lerp(accum, raw, 1 / (n + 1)), output accum
//                   (n = accum_count, frames accumulated since the last reset)
//   kModePathTrace: output raw; debug view kViewNrdDiff outputs diffuse instead
//   colour = output * exposure; a miss is SkyRadiance(primary direction) * exposure
//   with kRenderFlagSkyOnMiss, else 0; alpha = coverage; linear -> sRGB when the
//   colour target is a UNORM format (output_srgb).
//   kModePathTrace, debug view kViewNrdValidation, denoised frame: NRD's validation
//   layer (binding 6, RGBA8, a = opacity) is blended over the encoded colour.
//
// Descriptors (set 0): binding 0 the host colour target (no format qualifier,
// write only), bindings 1..6 the internal targets albedo (a = coverage),
// deterministic, diffuse, specular, the reference accumulation and NRD's
// validation layer. The push constant is the device address of the frame
// constants (same block as the trace).
#include "caelix_platform.hlsl"
#include "caelix_record.hlsl"
#include "caelix_frame.hlsl"
#include "caelix_block.hlsl"
#include "caelix_shading.hlsl"
#include "NRD.hlsli"

CAELIX_PUSH_CONSTANTS(CaelixTracePush, g_push);
CAELIX_BINDING(0, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_color;
CAELIX_BINDING(1, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_albedo;
CAELIX_BINDING(2, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_deterministic;
CAELIX_BINDING(3, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_diff;
CAELIX_BINDING(4, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_spec;
CAELIX_BINDING(5, 0) CAELIX_FORMAT("rgba32f") RWTexture2D<float4> g_int_accum;
CAELIX_BINDING(6, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_validation;

float3 LinearToSrgb(float3 c) {
  c = saturate(c);
  const float3 low = c * 12.92f;
  const float3 high = 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
  return float3(c.x <= 0.0031308f ? low.x : high.x, c.y <= 0.0031308f ? low.y : high.y,
                c.z <= 0.0031308f ? low.z : high.z);
}

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
  const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
  if (any(thread_id.xy >= fc.size)) {
    return;
  }
  const uint2 pixel = thread_id.xy;
  const float4 albedo = g_int_albedo[pixel];
  const bool hit = albedo.a > 0.5f;
  const bool path_trace = fc.mode == kModePathTrace;

  float3 color = float3(0.0f, 0.0f, 0.0f);
  if (hit) {
    const float3 diffuse = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(g_int_diff[pixel]).rgb;
    const float3 specular = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(g_int_spec[pixel]).rgb;
    const float3 raw = g_int_deterministic[pixel].rgb + albedo.rgb * (diffuse + specular);
    color = (path_trace && fc.debug_view == kViewNrdDiff) ? diffuse : raw;
    CAELIX_BRANCH
    if (fc.mode == kModeReference) {
      // The first frame after a reset is exactly the raw frame.
      const float3 mean = fc.accum_count == 0u
                              ? raw
                              : lerp(g_int_accum[pixel].rgb, raw, 1.0f / float(fc.accum_count + 1u));
      g_int_accum[pixel] = float4(mean, 1.0f);
      color = mean;
    }
  } else {
    CAELIX_BRANCH
    if (fc.mode == kModeReference) {
      g_int_accum[pixel] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if ((fc.flags & kRenderFlagSkyOnMiss) != 0u) {
      color = SkyRadiance(fc, CaelixPrimaryDirection(fc, pixel));
    }
  }
  color *= fc.exposure;
  if (fc.output_srgb != 0u) {
    color = LinearToSrgb(color);
  }
  CAELIX_BRANCH
  if (path_trace && fc.debug_view == kViewNrdValidation && fc.denoised != 0u) {
    // NRD's debug layer is display-ready RGBA8: blended after the encode.
    const float4 layer = g_int_validation[pixel];
    color = lerp(color, layer.rgb, layer.a);
  }
  g_color[pixel] = float4(color, hit ? 1.0f : 0.0f);
}
