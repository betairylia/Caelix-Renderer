// Voxel accumulate kernel of the per-voxel colour experiment (CAELIX_RENDER_FLAG_VOXEL_COLOR, fc.voxel_color;
// experiment branch sss-voxel-color-expr, docs/VOXEL_COLOR_EXPERIMENT.md): one thread per OUTPUT pixel
// (output_size; without an upscaler output_size = size). Every pixel adds its final linear colour to the
// hash slot of the voxel that its camera ray hits first; caelix_upscale_encode.hlsl then writes each pixel
// with its slot's mean (one colour per voxel, every face of the voxel alike).
//
//   source   = g_src[p * upscale_src_size / output_size]   (the image the encode kernel reads: the
//              upscale input without an upscaler or on the fallback, the upscaler's output otherwise)
//   coverage = g_int_albedo[p * size / output_size].a      (as the encode kernel)
//   ray      = without an upscaler CaelixPrimaryDirection(fc, p), the trace kernel's jittered primary
//              ray, so the key is the first hit that the frame traced; with an upscaler
//              CaelixOutputCentreDirection(fc, p), the unjittered ray through the output pixel centre
//              (the view of the upscaler's output). Origin fc.position, TMin 0, TMax kPrimaryTMax.
//   key      = (entity, voxel) of TraceScene's committed hit: the FIRST hit, transparent voxels included
//              (a water or glass surface voxel collects every pixel that sees it, whatever the delta
//              chain found behind it). voxel = HitVoxel (the rule of MediumCursorFromHit in
//              render/caelix_subsurface.hlsl).
//   slot     = linear probe from VoxelKeyHash(key, kVoxelSlotSeed): a compare-exchange of the fingerprint
//              word from 0; the slot is ours when the word was 0 or already our fingerprint. Then count
//              += 1 and the fixed-point colour sums (common/caelix_voxel_color.hlsl). A miss, coverage 0 or
//              kVoxelProbes failed probes store kVoxelSlotInvalid (the encode keeps the pixel's colour).
//
// Descriptors (the voxel set, set 0): binding 0 the TLAS (g_tlas, render/caelix_scene_trace.hlsl),
// binding 1 the source (RGBA16F), binding 2 the internal albedo (RGBA8, a = coverage), binding 3 the slot
// image (R32_UINT, output size, written for every pixel), binding 4 the hash table (cleared by the
// renderer before this dispatch). The frame constants are the uniform buffer g_frame (set 1, binding 0).
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_frame.hlsl"
#include "common/caelix_frame_ubo.hlsl"
#include "common/caelix_block.hlsl"
#include "common/caelix_material.hlsl"
#include "common/caelix_voxel_color.hlsl"
#include "render/caelix_scene_trace.hlsl"

CAELIX_BINDING(1, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_src;
CAELIX_BINDING(2, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_albedo;
CAELIX_BINDING(3, 0) CAELIX_FORMAT("r32ui") RWTexture2D<uint> g_voxel_slots;
CAELIX_BINDING(4, 0) RWStructuredBuffer<uint> g_voxel_hash;

// The primary ray length of the trace kernel (kRayTMax of render/caelix_path_trace.hlsl).
static const float kPrimaryTMax = 1e30f;

// The entity-local voxel that the face hit `hit` entered: half a voxel behind the hit point along the
// object-space face normal, clamped into the brick that the hit's candidate traced (the rule of
// MediumCursorFromHit, render/caelix_subsurface.hlsl). w = the entity that owns the hit's group.
int4 HitVoxel(CaelixFrameConstants fc, TraceResult hit)
{
    const uint64_t info = fc.group_infos + uint64_t(hit.group) * kGroupInfoBytes;
    const uint4 w0 = CAELIX_LOAD_U32x4(info);  // coord, entity (kGroupInfoCoord = 0, kGroupInfoEntity = 12)
    const uint slot_row_base = CAELIX_LOAD_U32(info + kGroupInfoSlotRowBase);
    const uint4 row = CAELIX_LOAD_U32x4(fc.slot_tables + (uint64_t(slot_row_base) + hit.id) * kSlotRowBytes);
    // Entity-local block coordinate of the brick's corner (as in TraceCandidate).
    const uint shift = fc.group_shift;
    const int mask = int((1u << shift) - 1u);
    const uint pos = SlotRowPos(row.x);
    const int3 brick = int3(int(pos) & mask, int(pos >> shift) & mask, int(pos >> (2 * shift)) & mask);
    const int3 origin = (int3(w0.xyz) * int(1u << shift) + brick) * int(kBrickEdge);
    const float3 n = float3(UnpackObjectNormal(hit.attrib));
    const int3 voxel = origin + clamp(int3(floor(hit.object_hit - 0.5f * n)) - origin, 0, int(kBrickEdge) - 1);
    return int4(voxel, int(w0.w));
}

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    const CaelixFrameConstants fc = g_frame;
    if (any(thread_id.xy >= fc.output_size))
    {
        return;
    }
    g_blockEncoding = fc.encoding;
    g_materialTable = fc.materials;
    const uint2 pixel = thread_id.xy;
    const uint2 src_pixel = min(pixel * fc.upscale_src_size / fc.output_size, fc.upscale_src_size - 1u);
    const uint2 render_pixel = min(pixel * fc.size / fc.output_size, fc.size - 1u);
    uint slot = kVoxelSlotInvalid;
    CAELIX_BRANCH
    if (g_int_albedo[render_pixel].a > 0.5f)
    {
        RayDesc ray;
        ray.Origin = fc.position.xyz;
        ray.Direction = fc.upscale != 0u ? CaelixOutputCentreDirection(fc, pixel) : CaelixPrimaryDirection(fc, pixel);
        ray.TMin = 0.0f;
        ray.TMax = kPrimaryTMax;
        const TraceResult hit = TraceScene(fc, ray);
        CAELIX_BRANCH
        if (hit.hit)
        {
            const int4 voxel = HitVoxel(fc, hit);
            const uint4 key = uint4(uint(voxel.w), asuint(voxel.xyz));
            const uint hashed = VoxelKeyHash(key, kVoxelFingerprintSeed);
            const uint fingerprint = hashed != 0u ? hashed : 1u;
            const uint start = VoxelKeyHash(key, kVoxelSlotSeed);
            CAELIX_LOOP
            for (uint probe = 0; probe < kVoxelProbes; ++probe)
            {
                const uint candidate = (start + probe) & fc.voxel_slot_mask;
                uint original;
                InterlockedCompareExchange(g_voxel_hash[candidate * kVoxelSlotWords + kVoxelSlotFingerprint], 0u,
                                           fingerprint, original);
                if (original == 0u || original == fingerprint)
                {
                    slot = candidate;
                    break;
                }
            }
            CAELIX_BRANCH
            if (slot != kVoxelSlotInvalid)
            {
                const uint base = slot * kVoxelSlotWords;
                InterlockedAdd(g_voxel_hash[base + kVoxelSlotCount], 1u);
                const float3 color = g_src[src_pixel].rgb;
                CAELIX_UNROLL
                for (uint c = 0; c < 3u; ++c)
                {
                    // 64-bit add: the carry of the low word is detected from the value it held before.
                    const uint value = VoxelFixed(color[c]);
                    uint before;
                    InterlockedAdd(g_voxel_hash[base + kVoxelSlotSumLo + c], value, before);
                    if (before + value < before)
                    {
                        InterlockedAdd(g_voxel_hash[base + kVoxelSlotSumHi + c], 1u);
                    }
                }
            }
        }
    }
    g_voxel_slots[pixel] = slot;
}
