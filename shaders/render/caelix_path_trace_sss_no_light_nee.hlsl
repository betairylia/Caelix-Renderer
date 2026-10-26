// CAELIX_KERNEL_VARIANT of render/caelix_path_trace.hlsl (cmake/Shaders.cmake compiles it as the kernel
// caelix_path_trace_sss_no_light_nee): the subsurface kernel (render/caelix_path_trace_sss.hlsl) without
// the light NEE (render/caelix_path_trace_no_light_nee.hlsl), for subsurface frames that run no light
// NEE. Same bindings, same frame constants.
#define CAELIX_SUBSURFACE 1
#define CAELIX_SPEC_NO_LIGHT_NEE 1
#include "render/caelix_path_trace.hlsl"
