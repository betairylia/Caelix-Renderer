// Primary ray kernel: one ray per pixel through the scene's TLAS with an inline
// ray query; the brick DDA (caelix_brick_trace.hlsl) runs on every procedural
// candidate, as in Unity's CaelixRayQueryTrace.hlsl. Writes the targets that the
// host passed (CaelixFrameConstants::target_mask): the colour target (albedo or a
// debug view) and the geometry targets albedo, normal, device depth, motion and
// linear view depth.
//
// Inputs: the push constant is the device address of the frame constants; the
// scene's tables (page table, slot rows, group infos, entity infos) are read
// through the addresses in them. The descriptors are the TLAS (binding 0) and the
// six targets (bindings 1..6). Every binding is PARTIALLY_BOUND: the library
// writes only the TLAS (when the scene has one) and the present targets, and the
// kernel touches a binding only when its bit is set (has_tlas, target_mask).
// The targets have no format qualifier (format Unknown in SPIR-V, needs
// shaderStorageImageWriteWithoutFormat), so every write is a float4 and the
// image keeps the components its format has.
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
CAELIX_BINDING(2, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_albedo;
CAELIX_BINDING(3, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_normal;
CAELIX_BINDING(4, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_depth;
CAELIX_BINDING(5, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_motion;
CAELIX_BINDING(6, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_view_z;

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

// The pixel (current target size) where the previous render call's camera saw
// the world point `p`, with the same pinhole mapping as the ray generation.
// Returns false when the point was not in front of that camera.
bool PreviousPixel(CaelixFrameConstants fc, float3 p, out float2 pixel) {
  const float3 rel = p - fc.prev_position.xyz;
  const float z = dot(rel, fc.prev_forward.xyz);
  pixel = float2(0.0f, 0.0f);
  if (!(z > 1e-6f)) {
    return false;
  }
  const float2 ndc = float2(dot(rel, fc.prev_right.xyz) / (z * fc.prev_tan_half_fov_y * fc.prev_aspect),
                            dot(rel, fc.prev_up.xyz) / (z * fc.prev_tan_half_fov_y));
  pixel = float2((ndc.x + 1.0f) * 0.5f * float(fc.size.x), (1.0f - ndc.y) * 0.5f * float(fc.size.y));
  return all(abs(pixel) < 1e30f);
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

  uint candidates = 0;
  bool hit = false;
  float T = 0.0f;
  float3 normal = float3(0.0f, 0.0f, 0.0f);
  float3 object_hit = float3(0.0f, 0.0f, 0.0f);  // entity-local block coordinates
  uint committed_attrib = 0;
  uint committed_group = 0;
  uint committed_id = 0;
  // Before the scene's first TLAS build binding 0 holds no valid descriptor
  // (PARTIALLY_BOUND): the ray query does not start and every pixel misses.
  CAELIX_BRANCH
  if (fc.has_tlas != 0u) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, ray);

    const uint shift = fc.group_shift;
    const int mask = int((1u << shift) - 1u);
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
    if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
      hit = true;
      T = q.CommittedRayT();
      const float3x4 object_to_world = q.CommittedObjectToWorld3x4();
      normal = normalize(mul((float3x3)object_to_world, UnpackObjectNormal(committed_attrib)));
      // The object ray direction is the world direction in object space (not
      // normalized), so the same T applies.
      object_hit = q.CommittedObjectRayOrigin() + q.CommittedObjectRayDirection() * T;
    }
  }

  const uint block = (committed_attrib >> 16) & 0xFFFFu;
  const float3 albedo = hit ? BlockAlbedo(block) : float3(0.0f, 0.0f, 0.0f);
  const uint targets = fc.target_mask;

  CAELIX_BRANCH
  if ((targets & kTargetColor) != 0u) {
    float3 color = fc.sky_color.xyz;
    if (fc.debug_view == kViewSteps) {
      color = candidates == 0 ? fc.sky_color.xyz : HeatMap(float(candidates) / float(kStepsHeatMax));
    } else if (hit) {
      if (fc.debug_view == kViewNormal) {
        color = normal * 0.5f + 0.5f;
      } else if (fc.debug_view == kViewDepth) {
        color = saturate((T - fc.near_z) / max(fc.far_z - fc.near_z, 1e-6f)).xxx;
      } else if (fc.debug_view == kViewGroup) {
        color = Pastel(HashU32(committed_group + 0x632BE5ABu));
      } else if (fc.debug_view == kViewBrick) {
        color = Pastel(HashU32(HashU32(committed_group + 0x632BE5ABu) ^ (committed_id + 0x2545F491u)));
      } else {
        color = albedo;
        if ((fc.flags & kRenderFlagShading) != 0u) {
          color *= 0.25f + 0.75f * saturate(dot(normal, fc.sun_dir.xyz));
        }
      }
    }
    // Opaque everywhere, so that a host can present the image directly.
    g_color[thread_id.xy] = float4(color, 1.0f);
  }

  const float coverage = hit ? 1.0f : 0.0f;
  CAELIX_BRANCH
  if ((targets & kTargetAlbedo) != 0u) {
    g_albedo[thread_id.xy] = float4(albedo, coverage);
  }
  CAELIX_BRANCH
  if ((targets & kTargetNormal) != 0u) {
    const float3 stored = (hit && (targets & kTargetNormalUnorm) != 0u) ? normal * 0.5f + 0.5f : normal;
    g_normal[thread_id.xy] = float4(stored, coverage);
  }

  // View space of the current camera: the hit is position + direction * T.
  const float3 rel = ray.Direction * T;
  const float view_z = dot(rel, fc.forward.xyz);
  CAELIX_BRANCH
  if ((targets & kTargetDepth) != 0u) {
    float depth = fc.miss_depth;
    if (hit) {
      const float4 view = float4(dot(rel, fc.right.xyz), dot(rel, fc.up.xyz), view_z, 1.0f);
      depth = dot(fc.projection[2], view) / dot(fc.projection[3], view);
    }
    g_depth[thread_id.xy] = float4(depth, 0.0f, 0.0f, 0.0f);
  }
  CAELIX_BRANCH
  if ((targets & kTargetViewZ) != 0u) {
    g_view_z[thread_id.xy] = float4(hit ? view_z : fc.miss_view_z, 0.0f, 0.0f, 0.0f);
  }
  CAELIX_BRANCH
  if ((targets & kTargetMotion) != 0u) {
    float2 motion = float2(0.0f, 0.0f);
    if (hit && fc.has_prev_camera != 0u) {
      // The hit point under the entity's previous transform: EntityInfo prev rows,
      // bytes 80..127 of the entity's 128-byte row (GroupInfo word 3 = entity).
      const uint entity = CAELIX_LOAD_U32(fc.group_infos + uint64_t(committed_group) * 32 + 12);
      const uint64_t entity_info = fc.entity_infos + uint64_t(entity) * 128;
      const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + 80));
      const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + 96));
      const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + 112));
      const float4 h = float4(object_hit, 1.0f);
      const float3 prev_world = float3(dot(r0, h), dot(r1, h), dot(r2, h));
      float2 prev_pixel;
      if (PreviousPixel(fc, prev_world, prev_pixel)) {
        motion = prev_pixel - (float2(thread_id.xy) + 0.5f);
      }
    }
    g_motion[thread_id.xy] = float4(motion, 0.0f, 0.0f);
  }
}
