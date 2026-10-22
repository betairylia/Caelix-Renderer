// Cross resolve of the delta checkerboard (kRenderFlagDeltaCheckerboard, path trace
// mode): a port of Unity's CaelixCrossResolve (Shaders/Denoise/CaelixCrossResolve.hlsl).
//
// At the first transparent interface the trace kernel sent even pixels (px + py) down
// the refraction and odd pixels down the reflection, each with the weight 2 (1 - F) or
// 2 F, and flagged them (motion target w = 1). This kernel averages the two parities
// back together on the COMPOSITED linear colour (the split reaches every G-buffer, so
// only the final colour can be blended): a flagged pixel becomes
//   (0.5 * centre + 0.125 * each FLAGGED orthogonal neighbour) / (sum of the weights).
// 0.5 + 4 * 0.125 puts half the weight on each parity, which is what the x2 weight is
// scaled to cancel. Diagonals have the centre's parity and are left out. The blend is
// deliberately not edge-aware: the neighbour IS a different surface. Renormalizing keeps
// the 50/50 mix when neighbours are not flagged and turns an isolated flagged pixel into
// a pass-through. Unflagged pixels pass through; alpha is carried through.
// NATIVE: neighbours outside the image are clamped to the edge (Unity's CaelixClampCoord);
// the kernel also does the output stage that the composite skips when it runs (sRGB
// encode, NRD validation overlay); the motion debug view passes through untouched.
// kRenderFlagFlipY: the host colour target is written at the mirrored row; the taps read the
// internal images at unflipped pixels (they are symmetric, so the flip does not change them).
// fc.upscale (an upscaler runs): the result goes, linear and unflipped, to the upscale input
// (binding 9) instead of the host target (no sRGB encode, no validation overlay).
//
// Inputs: the composite set (render/caelix_composite_set.hlsl): binding 8 the
// composite's linear output, binding 7 the motion (w = flag), binding 0 the host colour.
#include "common/caelix_platform.hlsl"
#include "common/caelix_frame.hlsl"
#include "render/caelix_composite_set.hlsl"

bool Flagged(uint2 pixel)
{
    return g_int_motion[pixel].w > 0.5f;
}

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    const CaelixFrameConstants fc = g_frame;
    if (any(thread_id.xy >= fc.size))
    {
        return;
    }
    const uint2 pixel = thread_id.xy;
    const float4 centre = g_int_resolve[pixel];
    CAELIX_BRANCH
    if (fc.mode == kModePathTrace && fc.debug_view == kViewMotion)
    {
        if (fc.upscale != 0u)
        {
            g_int_upscale_in[pixel] = centre;
        }
        else
        {
            g_color[CaelixOutputPixel(fc, pixel)] = centre;
        }
        return;
    }

    float3 color = centre.rgb;
    CAELIX_BRANCH
    if (Flagged(pixel))
    {
        float3 sum = centre.rgb * 0.5f;
        float weight = 0.5f;
        const int2 offsets[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
        const int2 last = int2(fc.size) - 1;
        CAELIX_UNROLL
        for (int tap = 0; tap < 4; ++tap)
        {
            const uint2 neighbour = uint2(clamp(int2(pixel) + offsets[tap], int2(0, 0), last));
            if (!Flagged(neighbour))
            {
                continue;
            }
            sum += g_int_resolve[neighbour].rgb * 0.125f;
            weight += 0.125f;
        }
        color = sum / weight;
    }
    CAELIX_BRANCH
    if (fc.upscale != 0u)
    {
        g_int_upscale_in[pixel] = CaelixUpscaleInput(color, centre.a);
        return;
    }
    g_color[CaelixOutputPixel(fc, pixel)] = float4(CaelixOutputColor(fc, pixel, color), centre.a);
}
