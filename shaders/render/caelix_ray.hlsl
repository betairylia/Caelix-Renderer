// Ray constants shared by the shading (render/caelix_shading.hlsl) and the scene traversal
// (render/caelix_scene_trace.hlsl: the transmissive shadow ray starts new rays on voxel faces).
#ifndef CAELIX_RAY_HLSL
#define CAELIX_RAY_HLSL

// Unity Utils.hlsl: K_RAY_ORIGIN_PUSH_OFF 0.002 (world units along the normal): a ray that starts on
// a voxel face starts this far off the face, on the side it travels to.
static const float K_RAY_ORIGIN_PUSH_OFF = 0.002f;

#endif  // CAELIX_RAY_HLSL
