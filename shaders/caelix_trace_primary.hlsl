// Primary ray kernel: one ray per pixel through the scene's TLAS with an inline
// ray query; the brick DDA (caelix_brick_trace.hlsl) runs on every procedural
// candidate, as in Unity's CaelixRayQueryTrace.hlsl. Writes the albedo of the
// hit or a debug view (CaelixFrameConstants::debug_view) into the colour target.
//
// Inputs: the push constant is the device address of the frame constants; the
// scene's tables (page table, slot rows, group infos) are read through the
// addresses in them. The TLAS and the colour target are the only descriptors.
// The colour target has no format qualifier (format Unknown in SPIR-V, needs
// shaderStorageImageWriteWithoutFormat), so any storage format the host picks
// (RGBA8, BGRA8, RGBA16F) works.
//
// Instances: instanceCustomIndex (CandidateInstanceID()) is the scene-wide group
// index, the primitive index is the dense brick id inside the group, and the
// object space is the entity-local block space.
#include "caelix_platform.hlsl"
#include "caelix_record.hlsl"
#include "caelix_frame.hlsl"
#include "caelix_block.hlsl"
#include "caelix_brick_trace.hlsl"

CAELIX_PUSH_CONSTANTS(CaelixTracePush, g_push);
CAELIX_BINDING(0, 0) RaytracingAccelerationStructure g_tlas;
CAELIX_BINDING(1, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_color;

static const float kRayTMax = 1e30f;
static const uint kStepsHeatMax = 64;
static const float3 kAirColor = float3(1.0f, 0.0f, 1.0f);     // air hit: should not happen
static const float3 kGlassTint = float3(0.6f, 0.8f, 1.0f);

uint64_t RecordAddress(CaelixFrameConstants fc, uint record_index) {
  const uint page = record_index >> fc.records_per_page_shift;
  const uint slot = record_index & ((1u << fc.records_per_page_shift) - 1u);
  const uint64_t page_address = CAELIX_LOAD_U64(fc.page_table + uint64_t(page) * 8);
  return page_address + uint64_t(slot) * fc.record_bytes;
}

uint HashU32(uint value) {
  value ^= value >> 16;
  value *= 0x7FEB352Du;
  value ^= value >> 15;
  value *= 0x846CA68Bu;
  value ^= value >> 16;
  return value;
}

// A light, saturated-but-soft colour per hash value.
float3 Pastel(uint hash) {
  const float3 rgb = float3(hash & 0xFFu, (hash >> 8) & 0xFFu, (hash >> 16) & 0xFFu) / 255.0f;
  return 0.35f + 0.6f * rgb;
}

// Blue -> cyan -> green -> yellow -> red for t in [0, 1].
float3 HeatMap(float t) {
  t = saturate(t);
  return saturate(float3(1.5f - abs(4.0f * t - 3.0f), 1.5f - abs(4.0f * t - 2.0f),
                         1.5f - abs(4.0f * t - 1.0f)));
}

// The albedo of a block value (face bits included). Packed encoding: opaque
// rgb5 (bit 15 set), emissive rgb4 (bits 15..14 = 01), glass with a fixed tint;
// block-ID encoding: a pastel hash of the id. Air (transparent id 0) is magenta,
// so a primary ray that reports an air boundary is visible.
float3 BlockAlbedo(uint v) {
  if (g_blockEncoding == kEncodingPacked) {
    if ((v & 0xC000u) == 0u) {
      return CaelixTransparentId(v) == 0u ? kAirColor : kGlassTint;
    }
    if ((v & 0x8000u) != 0u) {
      return float3((v >> 10) & 31u, (v >> 5) & 31u, v & 31u) / 31.0f;
    }
    return float3((v >> 10) & 15u, (v >> 6) & 15u, (v >> 2) & 15u) / 15.0f;
  }
  if ((v & 0x8000u) == 0u) {
    const uint id = CaelixTransparentId(v);
    return id == 0u ? kAirColor : Pastel(HashU32(id | 0x10000u));
  }
  return Pastel(HashU32(v & 0x7FFFu));
}

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
  const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
  if (any(thread_id.xy >= fc.size)) {
    return;
  }
  g_blockEncoding = fc.encoding;

  // Pixel -> ray. Row 0 is the top row of the image.
  const float2 ndc = float2((float(thread_id.x) + 0.5f) / float(fc.size.x) * 2.0f - 1.0f,
                            1.0f - (float(thread_id.y) + 0.5f) / float(fc.size.y) * 2.0f);
  RayDesc ray;
  ray.Origin = fc.position.xyz;
  ray.Direction = normalize(fc.right.xyz * (ndc.x * fc.tan_half_fov_y * fc.aspect) +
                            fc.up.xyz * (ndc.y * fc.tan_half_fov_y) + fc.forward.xyz);
  ray.TMin = 0.0f;
  ray.TMax = kRayTMax;

  RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
  q.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, ray);

  const uint shift = fc.group_shift;
  const int mask = int((1u << shift) - 1u);
  uint candidates = 0;
  uint committed_attrib = 0;
  uint committed_group = 0;
  uint committed_id = 0;
  while (q.Proceed()) {
    if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) {
      continue;
    }
    ++candidates;
    const uint group = q.CandidateInstanceID();
    const uint id = q.CandidatePrimitiveIndex();
    // GroupInfo: coord (words 0..2), entity, slot_row_base (word 4), ...
    const uint64_t info = fc.group_infos + uint64_t(group) * 32;
    const uint4 w0 = CAELIX_LOAD_U32x4(info);
    const uint slot_row_base = CAELIX_LOAD_U32(info + 16);
    const uint4 row = CAELIX_LOAD_U32x4(fc.slot_tables + (uint64_t(slot_row_base) + id) * kSlotRowBytes);
    if (SlotRowState(row.x) != kSlotStateResident || row.z == kInvalidRecord) {
      continue;
    }
    g_recordBase = RecordAddress(fc, row.z);
    // Entity-local block coordinate of the brick's corner (same formula as the
    // finalize kernel's AABB origin).
    const uint pos = SlotRowPos(row.x);
    const int3 brick = int3(int(pos) & mask, int(pos >> shift) & mask, int(pos >> (2 * shift)) & mask);
    const float3 brick_origin = float3((int3(w0.xyz) * int(1u << shift) + brick) * 8);

    AttributeData attrib;
    const float t = CaelixTraceBrickPrimitiveCore(0, q.CandidateObjectRayOrigin(),
                                                  q.CandidateObjectRayDirection(), q.CommittedRayT(),
                                                  brick_origin, attrib);
    // Nothing rejects a t outside [TMin, TCurrent] for us, so the test is done here.
    if (attrib.matID_faceNormal != 0u && t >= ray.TMin && t < q.CommittedRayT()) {
      q.CommitProceduralPrimitiveHit(t);
      committed_attrib = attrib.matID_faceNormal;
      committed_group = group;
      committed_id = id;
    }
  }

  float3 color = fc.sky_color.xyz;
  if (fc.debug_view == kViewSteps) {
    color = candidates == 0 ? fc.sky_color.xyz : HeatMap(float(candidates) / float(kStepsHeatMax));
  } else if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
    const float T = q.CommittedRayT();
    const float3x4 object_to_world = q.CommittedObjectToWorld3x4();
    const float3 normal =
        normalize(mul((float3x3)object_to_world, UnpackObjectNormal(committed_attrib)));
    const uint block = (committed_attrib >> 16) & 0xFFFFu;
    if (fc.debug_view == kViewNormal) {
      color = normal * 0.5f + 0.5f;
    } else if (fc.debug_view == kViewDepth) {
      color = saturate((T - fc.near_z) / max(fc.far_z - fc.near_z, 1e-6f)).xxx;
    } else if (fc.debug_view == kViewGroup) {
      color = Pastel(HashU32(committed_group + 0x632BE5ABu));
    } else if (fc.debug_view == kViewBrick) {
      color = Pastel(HashU32(HashU32(committed_group + 0x632BE5ABu) ^ (committed_id + 0x2545F491u)));
    } else {
      color = BlockAlbedo(block);
      if ((fc.flags & kRenderFlagShading) != 0u) {
        color *= 0.25f + 0.75f * saturate(dot(normal, fc.sun_dir.xyz));
      }
    }
  }
  g_color[thread_id.xy] = float4(color, 1.0f);
}
