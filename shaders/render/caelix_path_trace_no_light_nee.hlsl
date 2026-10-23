// CAELIX_KERNEL_VARIANT of render/caelix_path_trace.hlsl (cmake/Shaders.cmake compiles it as the kernel
// caelix_path_trace_no_light_nee): the trace kernel without the light NEE (docs/LIGHT_NEE.md section 7).
// The renderer binds it for every frame that runs no light NEE (light NEE off, no light tree, the primary
// mode). CAELIX_SPEC_NO_LIGHT_NEE makes CaelixFrameConstants::light_nee_flags 0 at compile time, so the
// compiler removes the light cell lookups, the lamp NEE and the MIS: the kernel is the trace kernel of
// before the light NEE (96 registers instead of 128; with the light NEE code in, a frame without light
// NEE traced 20 % slower at 2560 x 1440 on rootfall). Same bindings, same frame constants.
#define CAELIX_SPEC_NO_LIGHT_NEE 1
#include "render/caelix_path_trace.hlsl"
