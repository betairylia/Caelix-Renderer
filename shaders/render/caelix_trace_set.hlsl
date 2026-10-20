// The trace descriptor set, shared by caelix_path_trace.hlsl and the ReSTIR GI kernels
// (caelix_restir_gi_temporal.hlsl, caelix_restir_gi_spatial.hlsl, caelix_restir_gi_shade.hlsl):
// one set layout, one set per frame slot.
//
// Set 0: binding 0 the TLAS (declared by render/caelix_scene_trace.hlsl), bindings 1..6 the
// host targets (no format qualifier: format Unknown in SPIR-V, needs
// shaderStorageImageWriteWithoutFormat, so every write is a float4 and the image keeps the
// components its format has), bindings 7..18 the internal targets of the lighting modes
// (lib/src/render/renderer.h, InternalTargets; fixed formats, so the kernels can read them),
// binding 19 the sky cubemap (sampled image) and binding 20 its sampler, declared by
// render/caelix_shading.hlsl through CAELIX_SKY_BINDING_IMAGE / _SAMPLER, which this file
// defines: include it BEFORE caelix_shading.hlsl. Every binding is PARTIALLY_BOUND. The push
// constant is CaelixTracePush (the device address of the frame constants and the spatial pass
// index of the GI spatial kernel).
#ifndef CAELIX_TRACE_SET_HLSL
#define CAELIX_TRACE_SET_HLSL

#define CAELIX_SKY_BINDING_IMAGE 19
#define CAELIX_SKY_BINDING_SAMPLER 20

#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"

CAELIX_PUSH_CONSTANTS(CaelixTracePush, g_push);
CAELIX_BINDING(1, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_color;
CAELIX_BINDING(2, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_albedo;
CAELIX_BINDING(3, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_normal;
CAELIX_BINDING(4, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_depth;
CAELIX_BINDING(5, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_motion;
CAELIX_BINDING(6, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_view_z;
// Internal targets of the lighting modes (lib/src/render/renderer.h, InternalTargets).
CAELIX_BINDING(7, 0) CAELIX_FORMAT("r32f") RWTexture2D<float4> g_int_view_z;
CAELIX_BINDING(8, 0) CAELIX_FORMAT("rgb10a2") RWTexture2D<float4> g_int_normal_roughness;
CAELIX_BINDING(9, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_motion;
CAELIX_BINDING(10, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_albedo;
CAELIX_BINDING(11, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_deterministic;
CAELIX_BINDING(12, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_diff;
CAELIX_BINDING(13, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_spec;
CAELIX_BINDING(14, 0) CAELIX_FORMAT("r32f") RWTexture2D<float4> g_int_depth;
CAELIX_BINDING(15, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_rr_diffuse_albedo;
CAELIX_BINDING(16, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_rr_specular_albedo;
CAELIX_BINDING(17, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_rr_normal_roughness;
CAELIX_BINDING(18, 0) CAELIX_FORMAT("r16f") RWTexture2D<float4> g_int_rr_hit_dist;

#endif  // CAELIX_TRACE_SET_HLSL
