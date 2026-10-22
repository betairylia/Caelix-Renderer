// The frame set of the five frame-constant kernels (caelix_path_trace, caelix_composite,
// caelix_cross_resolve, caelix_sky_sh, caelix_upscale_encode): set 1 (kFrameSetIndex in
// lib/src/render/frame_constants.h), binding 0 = the frame slot's CaelixFrameConstants as a uniform
// buffer. Each kernel's main reads `CaelixFrameConstants fc = g_frame;`. Set 0 is the kernel's own
// set; the kernels have no push constants.
//
// A uniform buffer is read-only, so the compiler may re-read a field where it is used instead of
// keeping the whole struct live in registers across the traces (a struct loaded through a buffer
// device address stays live). The block has DXC's default constant buffer layout: it matches the C
// offsets of lib/src/render/frame_constants.h only under the layout rule in common/caelix_frame.hlsl,
// which the test caelix_frame_layout_test checks on the compiled kernels.
#ifndef CAELIX_FRAME_UBO_HLSL
#define CAELIX_FRAME_UBO_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"

CAELIX_BINDING(0, 1) ConstantBuffer<CaelixFrameConstants> g_frame;

#endif  // CAELIX_FRAME_UBO_HLSL
