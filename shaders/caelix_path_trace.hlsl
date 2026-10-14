// Trace kernel: one primary ray per pixel through the scene's TLAS with an inline
// ray query; the brick DDA (caelix_brick_trace.hlsl) runs on every procedural
// candidate, as in Unity's CaelixRayQueryTrace.hlsl.
//
// Every mode writes the host's geometry targets that are present
// (CaelixFrameConstants::target_mask): albedo, normal, device depth, motion and
// linear view depth.
//   kModePrimary:   the colour target gets a debug view of the primary hit.
//   kModePathTrace, kModeReference: the path tracer of Unity's
//                   CaelixPathTraceCommon.hlsl (opaque surfaces): the primary hit
//                   is shaded (CaelixShadeVoxelHit), its emission goes to the
//                   deterministic target, and up to bounce_count stochastic
//                   segments gather the incident radiance into the diffuse OR
//                   the specular target (by the lobe chosen at the primary hit),
//                   packed for NRD REBLUR (REBLUR_FrontEnd_PackRadianceAndNormHitDist:
//                   YCoCg radiance, normalized hit distance of the first segment in
//                   alpha). The internal G-buffer (view_z, normal_roughness packed by
//                   NRD_FrontEnd_PackNormalAndRoughness, motion, albedo) is written
//                   for the composite and for NRD. The colour target is written by
//                   caelix_composite.hlsl.
//                   With kRenderFlagSunNee every diffuse vertex also samples the
//                   sun cone and traces a shadow ray (TracePath, SunNextEvent).
//
// Inputs: the push constant is the device address of the frame constants; the
// scene's tables (page table, slot rows, group infos, entity infos) are read
// through the addresses in them. The descriptors are the TLAS (binding 0), the
// six host targets (bindings 1..6, no format qualifier: format Unknown in SPIR-V,
// needs shaderStorageImageWriteWithoutFormat, so every write is a float4 and the
// image keeps the components its format has) and the internal targets of the
// lighting modes (bindings 7..13, fixed formats). Every binding is
// PARTIALLY_BOUND: the library writes only the TLAS (when the scene has one), the
// present targets and, in the lighting modes, the internal targets, and the
// kernel touches a binding only then (has_tlas, target_mask, mode).
//
// Instances: instanceCustomIndex (CandidateInstanceID()) is the scene-wide group
// index, the primitive index is the dense brick id inside the group, and the
// object space is the entity-local block space.
#include "caelix_platform.hlsl"
#include "caelix_record.hlsl"
#include "caelix_frame.hlsl"
#include "caelix_block.hlsl"
#include "caelix_brick_trace.hlsl"
#include "caelix_shading.hlsl"
// NRD's front-end packing (normal + roughness, radiance + normalized hit distance);
// NRDConfig.hlsli next to it fixes the encodings NRD was built with.
#include "NRD.hlsli"

CAELIX_PUSH_CONSTANTS(CaelixTracePush, g_push);
CAELIX_BINDING(0, 0) RaytracingAccelerationStructure g_tlas;
CAELIX_BINDING(1, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_color;
CAELIX_BINDING(2, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_albedo;
CAELIX_BINDING(3, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_normal;
CAELIX_BINDING(4, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_depth;
CAELIX_BINDING(5, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_motion;
CAELIX_BINDING(6, 0) CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_view_z;
// Internal targets of the lighting modes (lib/src/renderer.h, InternalTargets).
CAELIX_BINDING(7, 0) CAELIX_FORMAT("r32f") RWTexture2D<float4> g_int_view_z;
CAELIX_BINDING(8, 0) CAELIX_FORMAT("rgb10a2") RWTexture2D<float4> g_int_normal_roughness;
CAELIX_BINDING(9, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_motion;
CAELIX_BINDING(10, 0) CAELIX_FORMAT("rgba8") RWTexture2D<float4> g_int_albedo;
CAELIX_BINDING(11, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_deterministic;
CAELIX_BINDING(12, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_diff;
CAELIX_BINDING(13, 0) CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_int_spec;

static const float kRayTMax = 1e30f;
static const uint kStepsHeatMax = 64;
// Internal view_z of a miss (beyond any denoising range).
static const float kInternalMissViewZ = 1e30f;

uint64_t RecordAddress(CaelixFrameConstants fc, uint record_index) {
  const uint page = record_index >> fc.records_per_page_shift;
  const uint slot = record_index & ((1u << fc.records_per_page_shift) - 1u);
  const uint64_t page_address = CAELIX_LOAD_U64(fc.page_table + uint64_t(page) * 8);
  return page_address + uint64_t(slot) * fc.record_bytes;
}

// Blue -> cyan -> green -> yellow -> red for t in [0, 1].
float3 HeatMap(float t) {
  t = saturate(t);
  return saturate(float3(1.5f - abs(4.0f * t - 3.0f), 1.5f - abs(4.0f * t - 2.0f),
                         1.5f - abs(4.0f * t - 1.0f)));
}

// The pixel (current target size) where the previous render call's camera saw
// the world point `p`, with the same pinhole mapping as the ray generation, and
// the point's view depth for that camera. Returns false when the point was not in
// front of that camera.
bool PreviousPixel(CaelixFrameConstants fc, float3 p, out float2 pixel, out float prev_view_z) {
  const float3 rel = p - fc.prev_position.xyz;
  const float z = dot(rel, fc.prev_forward.xyz);
  pixel = float2(0.0f, 0.0f);
  prev_view_z = z;
  if (!(z > 1e-6f)) {
    return false;
  }
  const float2 ndc = float2(dot(rel, fc.prev_right.xyz) / (z * fc.prev_tan_half_fov_y * fc.prev_aspect),
                            dot(rel, fc.prev_up.xyz) / (z * fc.prev_tan_half_fov_y));
  pixel = float2((ndc.x + 1.0f) * 0.5f * float(fc.size.x), (1.0f - ndc.y) * 0.5f * float(fc.size.y));
  return all(abs(pixel) < 1e30f);
}

// The committed hit of one ray through the scene.
struct TraceResult {
  bool hit;
  float T;               // world distance (the ray direction is unit length)
  uint attrib;           // AttributeData::matID_faceNormal: block << 16 | face flags
  uint group;            // scene-wide group index
  uint id;               // dense brick id in the group
  uint candidates;       // procedural candidates visited
  float3 normal;         // world-space face normal
  float3 object_hit;     // entity-local block coordinates of the hit
};

// The brick DDA on one procedural candidate (group, dense id) of a ray query:
// returns the DDA's t (object space = world distance, scale 1) and its attribute;
// attrib.matID_faceNormal == 0 means no hit (or the brick is not resident).
float TraceCandidate(CaelixFrameConstants fc, uint group, uint id, float3 object_origin,
                     float3 object_direction, float t_current, out AttributeData attrib) {
  attrib.matID_faceNormal = 0u;
  // GroupInfo: coord (words 0..2), entity, slot_row_base (word 4), ...
  const uint64_t info = fc.group_infos + uint64_t(group) * 32;
  const uint4 w0 = CAELIX_LOAD_U32x4(info);
  const uint slot_row_base = CAELIX_LOAD_U32(info + 16);
  const uint4 row = CAELIX_LOAD_U32x4(fc.slot_tables + (uint64_t(slot_row_base) + id) * kSlotRowBytes);
  if (SlotRowState(row.x) != kSlotStateResident || row.z == kInvalidRecord) {
    return t_current;
  }
  g_recordBase = RecordAddress(fc, row.z);
  // Entity-local block coordinate of the brick's corner (same formula as the
  // finalize kernel's AABB origin).
  const uint shift = fc.group_shift;
  const int mask = int((1u << shift) - 1u);
  const uint pos = SlotRowPos(row.x);
  const int3 brick = int3(int(pos) & mask, int(pos >> shift) & mask, int(pos >> (2 * shift)) & mask);
  const float3 brick_origin = float3((int3(w0.xyz) * int(1u << shift) + brick) * 8);
  return CaelixTraceBrickPrimitiveCore(0, object_origin, object_direction, t_current, brick_origin,
                                       attrib);
}

// Traces `ray` through the TLAS: every procedural candidate runs the brick DDA on
// its record, and a DDA hit inside [TMin, CommittedRayT()) is committed. Before
// the scene's first TLAS build binding 0 holds no valid descriptor
// (PARTIALLY_BOUND): the ray query does not start and the ray misses.
TraceResult TraceScene(CaelixFrameConstants fc, RayDesc ray) {
  TraceResult r;
  r.hit = false;
  r.T = 0.0f;
  r.attrib = 0;
  r.group = 0;
  r.id = 0;
  r.candidates = 0;
  r.normal = float3(0.0f, 0.0f, 0.0f);
  r.object_hit = float3(0.0f, 0.0f, 0.0f);
  CAELIX_BRANCH
  if (fc.has_tlas != 0u) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed()) {
      if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) {
        continue;
      }
      ++r.candidates;
      const uint group = q.CandidateInstanceID();
      const uint id = q.CandidatePrimitiveIndex();
      AttributeData attrib;
      const float t = TraceCandidate(fc, group, id, q.CandidateObjectRayOrigin(),
                                     q.CandidateObjectRayDirection(), q.CommittedRayT(), attrib);
      // Nothing rejects a t outside [TMin, TCurrent] for us, so the test is done here.
      if (attrib.matID_faceNormal != 0u && t >= ray.TMin && t < q.CommittedRayT()) {
        q.CommitProceduralPrimitiveHit(t);
        r.attrib = attrib.matID_faceNormal;
        r.group = group;
        r.id = id;
      }
    }
    if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
      r.hit = true;
      r.T = q.CommittedRayT();
      const float3x4 object_to_world = q.CommittedObjectToWorld3x4();
      r.normal = normalize(mul((float3x3)object_to_world, UnpackObjectNormal(r.attrib)));
      // The object ray direction is the world direction in object space (not
      // normalized), so the same T applies.
      r.object_hit = q.CommittedObjectRayOrigin() + q.CommittedObjectRayDirection() * r.T;
    }
  }
  return r;
}

// Shadow ray: true when a voxel lies on `ray` inside [TMin, TMax). The first
// candidate whose DDA reports a hit in range is committed and ends the search
// (RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH); no closest-hit data is needed.
bool TraceShadow(CaelixFrameConstants fc, RayDesc ray) {
  bool occluded = false;
  CAELIX_BRANCH
  if (fc.has_tlas != 0u) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(g_tlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {
      if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) {
        continue;
      }
      AttributeData attrib;
      const float t = TraceCandidate(fc, q.CandidateInstanceID(), q.CandidatePrimitiveIndex(),
                                     q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(),
                                     q.CommittedRayT(), attrib);
      if (attrib.matID_faceNormal != 0u && t >= ray.TMin && t < q.CommittedRayT()) {
        q.CommitProceduralPrimitiveHit(t);
        occluded = true;
        break;
      }
    }
  }
  return occluded;
}

// Sun next-event estimation at a diffuse path vertex with world normal `normal`
// and pushed-off position `origin`: one direction L uniform in the sun cone (pdf
// 1 / omega) and a shadow ray. Returns the demodulated incident radiance of the
// Lambert lobe, sun_radiance * omega * dot(N, L) / pi when L is unoccluded, else 0
// (the albedo of the vertex is in the caller's throughput, as for the bounces).
float3 SunNextEvent(CaelixFrameConstants fc, float3 origin, float3 normal, inout uint rng) {
  const float3 L = SampleSunDirection(fc, rng);
  const float cos_l = dot(normal, L);
  if (!(cos_l > 0.0f)) {
    return float3(0.0f, 0.0f, 0.0f);
  }
  RayDesc shadow;
  shadow.Origin = origin;
  shadow.Direction = L;
  shadow.TMin = 0.0f;
  shadow.TMax = kRayTMax;
  if (TraceShadow(fc, shadow)) {
    return float3(0.0f, 0.0f, 0.0f);
  }
  return fc.sun_radiance.xyz * (fc.sun_solid_angle * cos_l * (1.0f / K_PI));
}

// The path tracer at one primary hit (Unity CaelixPathTraceMain, "Further
// Bounces", opaque surfaces only): shades the primary hit, then traces up to
// bounce_count stochastic segments. Returns the primary shade result; `stochastic`
// receives (incident radiance, hit distance of the first segment).
//
// Sun NEE (kRenderFlagSunNee with a nonzero sun radiance): every vertex that
// sampled the diffuse lobe AND has a next segment (the primary when
// bounce_count >= 1, a bounce hit while bounce_index <= bounce_count) adds
// SunNextEvent * throughput (throughput including that vertex's albedo, before
// Russian roulette). So NEE replaces exactly the sun contribution that the next
// BSDF segment would have found, and the expected value does not change. The
// segment that leaves such a vertex sees the sky without the sun disk; a segment
// that leaves a specular-lobe vertex (no NEE there) still sees the disk, so the sun
// is counted once on every path. NEE draws from its own random stream, so the BSDF
// path consumes the same random numbers with NEE on or off.
CaelixShadeResult TracePath(CaelixFrameConstants fc, uint2 pixel, RayDesc ray, TraceResult primary,
                            out float4 stochastic) {
  uint rng = InitRandom(pixel, fc.frame_seed);
  uint nee_rng = InitRandom(pixel, fc.frame_seed ^ 0x68E31DA4u);
  const bool nee = (fc.flags & kRenderFlagSunNee) != 0u && any(fc.sun_radiance.xyz > 0.0f);
  const uint primary_block = (primary.attrib >> 16) & 0xFFFFu;
  const CaelixShadeResult primary_shade =
      ShadeVoxelHit(primary_block, ray.Origin + ray.Direction * primary.T, ray.Direction,
                    primary.normal, rng);

  float3 radiance = float3(0.0f, 0.0f, 0.0f);
  float3 throughput = 1.0f / max(0.001f, primary_shade.k);
  // Hit distance of the first stochastic segment (primary surface -> first bounce
  // hit) in world units; t_max when it reaches the sky, 0 when there is no segment
  // (bounce_count 0).
  float first_segment_t = fc.bounce_count >= 1u ? fc.t_max : 0.0f;
  bool first_segment = true;

  // NEE at the primary vertex (demodulated: throughput 1 / k).
  bool segment_sees_sun = true;
  CAELIX_BRANCH
  if (nee && !primary_shade.specular_lobe && fc.bounce_count >= 1u) {
    radiance += SunNextEvent(fc, primary_shade.bounce_origin, primary.normal, nee_rng) * throughput;
    segment_sees_sun = false;
  }

  RayDesc bounce;
  bounce.Origin = primary_shade.bounce_origin;
  bounce.Direction = primary_shade.bounce_direction;
  bounce.TMin = 0.0f;
  bounce.TMax = fc.t_max;

  // Unity's stochastic bounce counter: 1 after the primary shade, +1 per shaded
  // bounce hit; the loop runs while it is <= bounce_count, so bounce_count
  // segments at most.
  uint bounce_index = 1;
  CAELIX_LOOP
  for (; bounce_index <= fc.bounce_count;) {
    const TraceResult hit = TraceScene(fc, bounce);
    if (!hit.hit) {
      radiance += (segment_sees_sun ? SkyRadiance(fc, bounce.Direction)
                                    : SkyRadianceNoSun(fc, bounce.Direction)) * throughput;
      break;
    }
    if (first_segment) {
      first_segment_t = hit.T;
      first_segment = false;
    }
    const CaelixShadeResult shade =
        ShadeVoxelHit((hit.attrib >> 16) & 0xFFFFu, bounce.Origin + bounce.Direction * hit.T,
                      bounce.Direction, hit.normal, rng);
    bounce_index += 1;

    radiance += shade.emission * throughput;
    throughput *= shade.albedo / max(0.001f, shade.k);

    segment_sees_sun = true;
    CAELIX_BRANCH
    if (nee && !shade.specular_lobe && bounce_index <= fc.bounce_count) {
      radiance += SunNextEvent(fc, shade.bounce_origin, hit.normal, nee_rng) * throughput;
      segment_sees_sun = false;
    }

    // Russian roulette: dark paths stop early. A zero throughput must stop here,
    // or 0 * (1 / 0) below gives NaN.
    const float continue_probability = max(throughput.r, max(throughput.g, throughput.b));
    if (continue_probability <= 0.0f || continue_probability < RandomFloat01(rng)) {
      break;
    }
    throughput *= 1.0f / continue_probability;

    bounce.Origin = shade.bounce_origin;
    bounce.Direction = shade.bounce_direction;
  }
  stochastic = float4(SanitizeRadiance(radiance), first_segment_t);
  return primary_shade;
}

[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
  const CaelixFrameConstants fc = CAELIX_LOAD_STRUCT(CaelixFrameConstants, g_push.frame_constants, 16);
  if (any(thread_id.xy >= fc.size)) {
    return;
  }
  g_blockEncoding = fc.encoding;

  // Pixel -> ray. Row 0 is the top row of the image.
  RayDesc ray;
  ray.Origin = fc.position.xyz;
  ray.Direction = CaelixPrimaryDirection(fc, thread_id.xy);
  ray.TMin = 0.0f;
  ray.TMax = kRayTMax;

  const TraceResult primary = TraceScene(fc, ray);
  const bool hit = primary.hit;
  const float T = primary.T;
  const float3 normal = primary.normal;

  const uint block = (primary.attrib >> 16) & 0xFFFFu;
  const float3 albedo = hit ? BlockAlbedo(block) : float3(0.0f, 0.0f, 0.0f);
  const uint targets = fc.target_mask;
  const bool lighting = fc.mode != kModePrimary;

  CAELIX_BRANCH
  if (!lighting && (targets & kTargetColor) != 0u) {
    float3 color = fc.sky_color.xyz;
    if (fc.debug_view == kViewSteps) {
      color = primary.candidates == 0 ? fc.sky_color.xyz
                                      : HeatMap(float(primary.candidates) / float(kStepsHeatMax));
    } else if (hit) {
      if (fc.debug_view == kViewNormal) {
        color = normal * 0.5f + 0.5f;
      } else if (fc.debug_view == kViewDepth) {
        color = saturate((T - fc.near_z) / max(fc.far_z - fc.near_z, 1e-6f)).xxx;
      } else if (fc.debug_view == kViewGroup) {
        color = Pastel(HashU32(primary.group + 0x632BE5ABu));
      } else if (fc.debug_view == kViewBrick) {
        color = Pastel(HashU32(HashU32(primary.group + 0x632BE5ABu) ^ (primary.id + 0x2545F491u)));
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

  // Motion: the host target and the internal one of the lighting modes.
  float2 motion = float2(0.0f, 0.0f);
  float motion_view_z = 0.0f;  // previous view z - current view z
  CAELIX_BRANCH
  if (((targets & kTargetMotion) != 0u || lighting) && hit && fc.has_prev_camera != 0u) {
    // The hit point under the entity's previous transform: EntityInfo prev rows,
    // bytes 80..127 of the entity's 128-byte row (GroupInfo word 3 = entity).
    const uint entity = CAELIX_LOAD_U32(fc.group_infos + uint64_t(primary.group) * 32 + 12);
    const uint64_t entity_info = fc.entity_infos + uint64_t(entity) * 128;
    const float4 r0 = asfloat(CAELIX_LOAD_U32x4(entity_info + 80));
    const float4 r1 = asfloat(CAELIX_LOAD_U32x4(entity_info + 96));
    const float4 r2 = asfloat(CAELIX_LOAD_U32x4(entity_info + 112));
    const float4 h = float4(primary.object_hit, 1.0f);
    const float3 prev_world = float3(dot(r0, h), dot(r1, h), dot(r2, h));
    float2 prev_pixel;
    float prev_view_z;
    if (PreviousPixel(fc, prev_world, prev_pixel, prev_view_z)) {
      motion = prev_pixel - (float2(thread_id.xy) + 0.5f);
      motion_view_z = prev_view_z - view_z;
    }
  }
  CAELIX_BRANCH
  if ((targets & kTargetMotion) != 0u) {
    g_motion[thread_id.xy] = float4(motion, 0.0f, 0.0f);
  }

  CAELIX_BRANCH
  if (lighting) {
    const uint2 pixel = thread_id.xy;
    g_int_view_z[pixel] = float4(hit ? view_z : kInternalMissViewZ, 0.0f, 0.0f, 0.0f);
    g_int_motion[pixel] = float4(motion, motion_view_z, 0.0f);

    float3 material_albedo = float3(0.0f, 0.0f, 0.0f);
    float3 deterministic = float3(0.0f, 0.0f, 0.0f);
    // NRD packing. The lobe that the primary hit did not sample gets 0 radiance and
    // 0 hit distance (NRD's "invalid" sample, reconstructed by
    // HitDistanceReconstructionMode); a miss gets 0 in both (the composite draws the
    // sky; view z 1e30 is outside the denoising range). The hit distance is
    // normalized with REBLUR's A, B, C at the primary view z: roughness 1 for the
    // diffuse lobe, the material roughness for the specular one.
    float4 diffuse = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 specular = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 normal_roughness = float4(0.0f, 0.0f, 0.0f, 0.0f);
    CAELIX_BRANCH
    if (hit) {
      float4 stochastic;
      const CaelixShadeResult shade = TracePath(fc, pixel, ray, primary, stochastic);
      material_albedo = shade.albedo;
      // Primary emission is deterministic: nothing was sampled to obtain it.
      deterministic = shade.emission;
      normal_roughness = NRD_FrontEnd_PackNormalAndRoughness(normal, shade.roughness, 0.0f);
      const float lobe_roughness = shade.specular_lobe ? shade.roughness : 1.0f;
      const float norm_hit_dist =
          REBLUR_FrontEnd_GetNormHitDist(stochastic.a, view_z, fc.hit_dist_params.xyz, lobe_roughness);
      const float4 packed = REBLUR_FrontEnd_PackRadianceAndNormHitDist(stochastic.rgb, norm_hit_dist, true);
      if (shade.specular_lobe) {
        specular = packed;
      } else {
        diffuse = packed;
      }
    }
    g_int_normal_roughness[pixel] = normal_roughness;
    // Coverage is albedo.a.
    g_int_albedo[pixel] = float4(material_albedo, coverage);
    g_int_deterministic[pixel] = float4(SanitizeRadiance(deterministic), coverage);
    g_int_diff[pixel] = diffuse;
    g_int_spec[pixel] = specular;
  }
}
