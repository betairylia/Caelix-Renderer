// CAELIX_KERNEL_VARIANT of render/caelix_path_trace.hlsl (cmake/Shaders.cmake compiles it as the kernel
// caelix_path_trace_light_tree_nee): the trace kernel whose light NEE has no light cells (experimental,
// CAELIX_LIGHT_NEE_FLAG_NO_CELLS, docs/LIGHT_NEE.md section 11.6). The renderer binds it for every frame
// that runs light NEE with that flag (light_nee_flags = kLightNeeDirect | kLightNeeContribute, with or
// without kLightNeeMis). CAELIX_SPEC_LIGHT_NEE_DIRECT replaces the cell lookup, the reservoir draw, the
// sun hint and the cell proxy of the MIS by a light tree draw for the vertex's own point and normal
// (LightTreeSample) and the exact pdf on the BSDF side (LightTreeBrickPdf x in_brick); the full kernel
// caelix_path_trace keeps only the cell path. Same bindings, same frame constants.
#define CAELIX_SPEC_LIGHT_NEE_DIRECT 1
#include "render/caelix_path_trace.hlsl"
