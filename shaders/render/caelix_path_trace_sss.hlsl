// CAELIX_KERNEL_VARIANT of render/caelix_path_trace.hlsl (cmake/Shaders.cmake compiles it as the kernel
// caelix_path_trace_sss): the trace kernel with the subsurface code (docs/SUBSURFACE.md section 4): the
// Henyey-Greenstein walk at the path vertices (kRenderFlagSubsurface) and the straight SSS walk of the
// transmissive shadow ray (docs/SHADOW_TRANSMISSION.md). The renderer binds it for the lighting frames
// with light NEE of a scene whose material table has an SSS entry (Scene::has_subsurface); every other
// scene keeps caelix_path_trace, which compiles none of that code (CAELIX_SUBSURFACE 0: the SSS code
// holds registers and costs occupancy even when no voxel uses it). Same bindings, same frame constants.
#define CAELIX_SUBSURFACE 1
#include "render/caelix_path_trace.hlsl"
