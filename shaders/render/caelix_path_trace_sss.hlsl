// CAELIX_KERNEL_VARIANT of render/caelix_path_trace.hlsl (cmake/Shaders.cmake compiles it as the kernel
// caelix_path_trace_sss): the trace kernel with the subsurface walk of render/caelix_subsurface.hlsl
// (experiment branch sss-expr, docs/SSS_EXPERIMENT.md). The renderer binds it for lighting frames with
// light NEE and CAELIX_RENDER_FLAG_SUBSURFACE whose scene has an SSS material; every other frame binds the
// kernels without the walk code. Same bindings, same frame constants.
#define CAELIX_SUBSURFACE 1
#include "render/caelix_path_trace.hlsl"
