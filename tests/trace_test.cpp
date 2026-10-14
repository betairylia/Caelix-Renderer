// The primary ray trace kernel (caelix_renderer_render): a one-brick scene and
// the random opaque scene are rendered into offscreen storage images (the colour
// target and the optional geometry targets albedo, normal, depth, motion and
// view_z), which are read back and checked pixel by pixel. Uses the public C ABI
// only.
#include "test_scenes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace test;

namespace {

constexpr uint32_t kSize = 256;
constexpr float kSky[3] = {0.25f, 0.5f, 0.75f};
constexpr float kNear = 0.1f;
constexpr float kFar = 1000.0f;
constexpr float kMissViewZ = 1e30f;

uint32_t FindMemoryType(Gpu& gpu, uint32_t type_bits, VkMemoryPropertyFlags wanted) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(gpu.device.physical_device.physical_device, &properties);
  uint32_t found = UINT32_MAX;
  for (uint32_t i = 0; i < properties.memoryTypeCount && found == UINT32_MAX; ++i) {
    if ((type_bits & (1u << i)) != 0 &&
        (properties.memoryTypes[i].propertyFlags & wanted) == wanted) {
      found = i;
    }
  }
  REQUIRE(found != UINT32_MAX);
  return found;
}

uint32_t BytesPerPixel(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R16G16_SFLOAT:
      return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
      return 8;
    default:
      return 0;  // not a format of this test
  }
}

float HalfToFloat(uint16_t h) {
  const uint32_t exponent = (h >> 10) & 31u;
  const uint32_t mantissa = h & 1023u;
  float value = 0.0f;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    value = mantissa != 0 ? NAN : INFINITY;
  } else {
    value = std::ldexp(static_cast<float>(mantissa | 1024u), static_cast<int>(exponent) - 25);
  }
  return (h & 0x8000u) != 0 ? -value : value;
}

// A storage image that a render call writes, and its last readback.
struct Target {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t bytes_per_pixel = 0;
  // UNDEFINED the first time, TRANSFER_SRC_OPTIMAL after a readback.
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  std::vector<uint8_t> bytes;  // width * height * bytes_per_pixel, row 0 = top

  const uint8_t* at(uint32_t x, uint32_t y) const {
    return bytes.data() + (static_cast<size_t>(y) * width + x) * bytes_per_pixel;
  }
  float F32(uint32_t x, uint32_t y, uint32_t c) const {
    float value = 0.0f;
    std::memcpy(&value, at(x, y) + 4 * c, 4);
    return value;
  }
  float F16(uint32_t x, uint32_t y, uint32_t c) const {
    uint16_t value = 0;
    std::memcpy(&value, at(x, y) + 2 * c, 2);
    return HalfToFloat(value);
  }
};

// An image with the usages a target needs (STORAGE only) plus TRANSFER_SRC for the
// readback. No TRANSFER_DST: the library never clears a target.
Target CreateTarget(Gpu& gpu, VkFormat format, uint32_t width = kSize, uint32_t height = kSize) {
  Target out;
  out.format = format;
  out.width = width;
  out.height = height;
  out.bytes_per_pixel = BytesPerPixel(format);
  REQUIRE(out.bytes_per_pixel != 0);
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  REQUIRE(vkCreateImage(gpu.device.device, &info, nullptr, &out.image) == VK_SUCCESS);
  VkMemoryRequirements requirements{};
  vkGetImageMemoryRequirements(gpu.device.device, out.image, &requirements);
  VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex =
      FindMemoryType(gpu, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  REQUIRE(vkAllocateMemory(gpu.device.device, &allocate, nullptr, &out.memory) == VK_SUCCESS);
  REQUIRE(vkBindImageMemory(gpu.device.device, out.image, out.memory, 0) == VK_SUCCESS);
  return out;
}

void DestroyTarget(Gpu& gpu, Target& target) {
  vkDestroyImage(gpu.device.device, target.image, nullptr);
  vkFreeMemory(gpu.device.device, target.memory, nullptr);
}

// The targets of one render call, in CaelixRenderTargets order; null = absent.
enum TargetIndex { kColor, kAlbedo, kNormal, kDepth, kMotion, kViewZ, kTargetCount };
struct TargetSet {
  Target* t[kTargetCount] = {};
};

CaelixTargetImage Abi(const Target* target) {
  CaelixTargetImage out{};
  if (target != nullptr) {
    out.image = reinterpret_cast<uint64_t>(target->image);
    out.format = static_cast<uint32_t>(target->format);
    out.width = target->width;
    out.height = target->height;
    out.layout = static_cast<uint32_t>(target->layout);
  }
  return out;
}

CaelixRenderTargets ToAbi(const TargetSet& set) {
  CaelixRenderTargets out{};
  out.color = Abi(set.t[kColor]);
  out.albedo = Abi(set.t[kAlbedo]);
  out.normal = Abi(set.t[kNormal]);
  out.depth = Abi(set.t[kDepth]);
  out.motion = Abi(set.t[kMotion]);
  out.view_z = Abi(set.t[kViewZ]);
  return out;
}

// Column-major projections for the camera's view space (+z forward), as a host
// builds them. Element (row r, column c) = m[c * 4 + r].
void NormalZProjection(float m[16], float tan_half_fov_y, float aspect) {
  std::memset(m, 0, 16 * sizeof(float));
  m[0] = 1.0f / (tan_half_fov_y * aspect);
  m[5] = 1.0f / tan_half_fov_y;
  m[10] = kFar / (kFar - kNear);          // clip.z = far / (far - near) * (z - near)
  m[14] = -kFar * kNear / (kFar - kNear);
  m[11] = 1.0f;                           // clip.w = z
}

void ReverseZInfiniteProjection(float m[16], float tan_half_fov_y, float aspect) {
  std::memset(m, 0, 16 * sizeof(float));
  m[0] = 1.0f / (tan_half_fov_y * aspect);
  m[5] = 1.0f / tan_half_fov_y;
  m[14] = kNear;  // clip.z = near
  m[11] = 1.0f;   // clip.w = z
}

// Device depth clip.z / clip.w of a point at view depth z on the view axis.
float ProjectDepth(const float m[16], float z) { return (m[10] * z + m[14]) / (m[11] * z + m[15]); }

// Camera at `position` looking along +Z, vertical field of view 60 degrees, with
// the normal-Z projection.
CaelixCamera LookPlusZ(float x, float y, float z) {
  CaelixCamera camera{};
  camera.position[0] = x;
  camera.position[1] = y;
  camera.position[2] = z;
  camera.right[0] = 1.0f;
  camera.up[1] = 1.0f;
  camera.forward[2] = 1.0f;
  camera.tan_half_fov_y = std::tan(30.0f * 3.14159265f / 180.0f);
  camera.aspect = 1.0f;
  camera.near_z = kNear;
  camera.far_z = kFar;
  NormalZProjection(camera.projection, camera.tan_half_fov_y, camera.aspect);
  return camera;
}

CaelixRenderOptions Options(uint32_t view) {
  CaelixRenderOptions options{};
  options.debug_view = view;
  options.flags = 0;  // unshaded: the checks compare exact albedo
  options.sun_dir[0] = 0.3f;
  options.sun_dir[1] = 1.0f;
  options.sun_dir[2] = -0.5f;
  options.sky_color[0] = kSky[0];
  options.sky_color[1] = kSky[1];
  options.sky_color[2] = kSky[2];
  options.miss_depth = 1.0f;
  options.miss_view_z = kMissViewZ;
  return options;
}

// One frame: flush, render into the present targets, copy each to a host buffer.
// Afterwards every present target holds its pixels and is in TRANSFER_SRC_OPTIMAL.
void Render(Rig& rig, CaelixRenderer* renderer, TargetSet& set, const CaelixCamera& camera,
            const CaelixRenderOptions& options) {
  Readback readbacks[kTargetCount];
  for (int i = 0; i < kTargetCount; ++i) {
    if (set.t[i] != nullptr) {
      const Target& target = *set.t[i];
      CreateReadback(rig.gpu, static_cast<VkDeviceSize>(target.width) * target.height *
                                  target.bytes_per_pixel,
                     readbacks[i]);
    }
  }
  const uint64_t frame = rig.frames.next++;
  SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
    CaelixFrameDesc desc{};
    desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
    desc.frame_number = frame;
    desc.safe_frame_number = frame - 1;
    REQUIRE(caelix_scene_flush(rig.scene, &desc) == CAELIX_OK);
    const CaelixRenderTargets targets = ToAbi(set);
    REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &options) == CAELIX_OK);

    for (int i = 0; i < kTargetCount; ++i) {
      if (set.t[i] == nullptr) {
        continue;
      }
      Target& target = *set.t[i];
      // GENERAL (render contract) -> TRANSFER_SRC, copy.
      VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
      barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = target.image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.imageMemoryBarrierCount = 1;
      dependency.pImageMemoryBarriers = &barrier;
      vkCmdPipelineBarrier2(cmd, &dependency);
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {target.width, target.height, 1};
      vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readbacks[i].buffer, 1, &region);
    }
    CopyToHostBarrier(cmd);
  });
  REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
  for (int i = 0; i < kTargetCount; ++i) {
    if (set.t[i] == nullptr) {
      continue;
    }
    Target& target = *set.t[i];
    target.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(readbacks[i].words);
    target.bytes.assign(bytes, bytes + static_cast<size_t>(target.width) * target.height *
                                           target.bytes_per_pixel);
    DestroyReadback(rig.gpu, readbacks[i]);
  }
}

// Render with the colour target only.
void RenderColor(Rig& rig, CaelixRenderer* renderer, Target& color, const CaelixCamera& camera,
                 const CaelixRenderOptions& options) {
  TargetSet set;
  set.t[kColor] = &color;
  Render(rig, renderer, set, camera, options);
}

// Calls caelix_renderer_render with a dummy command buffer and requires `expected`
// with one library error: argument checks run before anything is recorded.
void ExpectRenderError(Rig& rig, CaelixRenderer* renderer, const CaelixRenderTargets& targets,
                       const CaelixCamera& camera, CaelixResult expected) {
  ExpectLibraryErrors(1, [&] {
    CaelixFrameDesc desc{};
    desc.command_buffer = 1;
    desc.frame_number = rig.frames.next;
    const CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
    REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &options) == expected);
  });
}

// Flushes until every acceleration structure is built.
void DrainBuilds(Rig& rig) {
  int flushes = 0;
  while (Stats(rig.scene).as_builds_pending > 0) {
    FlushFrame(rig.gpu, rig.scene, rig.frames);
    REQUIRE(++flushes < 64);
  }
}

bool Near(const uint8_t* pixel, float r, float g, float b, int tolerance) {
  const float expected[3] = {r, g, b};
  for (int c = 0; c < 3; ++c) {
    if (std::abs(static_cast<int>(pixel[c]) - static_cast<int>(std::lround(expected[c] * 255.0f))) >
        tolerance) {
      return false;
    }
  }
  return true;
}

bool IsSky(const uint8_t* pixel) { return Near(pixel, kSky[0], kSky[1], kSky[2], 2); }

void PrintPixel(const char* name, const uint8_t* pixel) {
  std::printf("%s: %u %u %u %u\n", name, pixel[0], pixel[1], pixel[2], pixel[3]);
}

// One opaque pure-red brick (packed: bit 15 = opaque, red = 31 << 10) at key (0, 0, 0):
// blocks [0, 8)^3 of the entity.
void UploadRedBrick(Rig& rig) {
  World world;
  Brick brick{};
  brick.fill(static_cast<uint16_t>(0x8000u | (31u << 10)));
  world.bricks[{0, 0, 0}] = brick;
  Upload(rig, world, KeysOf(world));
  DrainBuilds(rig);
}

}  // namespace

TEST_CASE("one red brick: albedo, normal and brick views; sky before the first TLAS") {
  Rig rig;
  OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
  CaelixRenderer* renderer = nullptr;
  REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
  Target target = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
  const CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);

  // No bricks, no TLAS: the kernel traces nothing and writes the miss values (the
  // TLAS binding is not written; it is PARTIALLY_BOUND).
  {
    Target depth = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT);
    Target albedo = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
    TargetSet set;
    set.t[kColor] = &target;
    set.t[kDepth] = &depth;
    set.t[kAlbedo] = &albedo;
    Render(rig, renderer, set, camera, Options(CAELIX_VIEW_ALBEDO));
    REQUIRE(IsSky(target.at(kSize / 2, kSize / 2)));
    REQUIRE(IsSky(target.at(0, 0)));
    REQUIRE(target.at(kSize / 2, kSize / 2)[3] == 255);
    REQUIRE(depth.F32(kSize / 2, kSize / 2, 0) == 1.0f);
    REQUIRE(albedo.at(kSize / 2, kSize / 2)[3] == 0);
    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    DestroyTarget(rig.gpu, depth);
    DestroyTarget(rig.gpu, albedo);
  }

  UploadRedBrick(rig);

  RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_ALBEDO));
  const uint8_t* centre = target.at(kSize / 2, kSize / 2);
  PrintPixel("albedo centre", centre);
  PrintPixel("albedo corner", target.at(0, 0));
  CHECK(centre[0] > 200);
  CHECK(centre[1] < 30);
  CHECK(centre[2] < 30);
  CHECK(IsSky(target.at(0, 0)));
  CHECK(IsSky(target.at(kSize - 1, kSize - 1)));

  // The camera sees the -Z face: normal (0, 0, -1) -> (0.5, 0.5, 0).
  RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_NORMAL));
  PrintPixel("normal centre", target.at(kSize / 2, kSize / 2));
  CHECK(Near(target.at(kSize / 2, kSize / 2), 0.5f, 0.5f, 0.0f, 2));

  RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_BRICK));
  const uint8_t* brick_centre = target.at(kSize / 2, kSize / 2);
  PrintPixel("brick centre", brick_centre);
  CHECK((brick_centre[0] | brick_centre[1] | brick_centre[2]) != 0);
  CHECK(!IsSky(brick_centre));

  // The other views run without validation messages.
  RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_DEPTH));
  RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_GROUP));
  RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_STEPS));
  CHECK(!IsSky(target.at(kSize / 2, kSize / 2)));  // one candidate
  CHECK(IsSky(target.at(0, 0)));                    // none

  // GPU timings: every frame was waited for, so the previous render's frame is resolved.
  CaelixGpuTimings timings{};
  caelix_renderer_get_gpu_timings(renderer, &timings);
  std::printf("timings: frame %llu, trace %.4f ms, flush %.4f ms\n",
              static_cast<unsigned long long>(timings.frame_number), timings.trace_ms,
              timings.flush_ms);
  CHECK(timings.frame_number > 0);
  CHECK(timings.frame_number < rig.frames.next);
  CHECK(timings.trace_ms > 0.0f);

  // Invalid arguments are rejected with an error log.
  ExpectLibraryErrors(1, [&] {
    CaelixRenderOptions bad = Options(99);
    CaelixFrameDesc desc{};
    desc.command_buffer = 1;
    desc.frame_number = rig.frames.next;
    CaelixRenderTargets targets{};
    targets.color = Abi(&target);
    REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &bad) ==
            CAELIX_ERROR_INVALID_ARGUMENT);
  });

  REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
  caelix_renderer_release_target_views(renderer);
  caelix_renderer_destroy(renderer);
  DestroyTarget(rig.gpu, target);
  CloseRig(rig);
}

TEST_CASE("geometry targets: albedo, normal, depth, motion, view_z") {
  Rig rig;
  OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
  UploadRedBrick(rig);
  CaelixRenderer* renderer = nullptr;
  REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);

  Target color = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
  Target albedo = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
  Target normal = CreateTarget(rig.gpu, VK_FORMAT_R16G16B16A16_SFLOAT);
  Target depth = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT);
  Target motion = CreateTarget(rig.gpu, VK_FORMAT_R32G32_SFLOAT);
  Target view_z = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT);
  TargetSet all;
  all.t[kColor] = &color;
  all.t[kAlbedo] = &albedo;
  all.t[kNormal] = &normal;
  all.t[kDepth] = &depth;
  all.t[kMotion] = &motion;
  all.t[kViewZ] = &view_z;

  constexpr uint32_t cx = kSize / 2;
  constexpr uint32_t cy = kSize / 2;
  const auto is_hit = [&](uint32_t x, uint32_t y) { return albedo.at(x, y)[3] == 255; };

  // 1. First render of the renderer, normal-Z projection. The brick covers blocks
  //    [0, 8)^3, so its -Z face lies in the plane z = 0; the camera sits at z = -20
  //    looking +Z, and the centre pixel's ray hits that face near (4, 4, 0):
  //    view_z = 0 - (-20) = 20.
  const CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);
  Render(rig, renderer, all, camera, Options(CAELIX_VIEW_ALBEDO));
  const float view_z0 = view_z.F32(cx, cy, 0);
  std::printf("centre: albedo %u %u %u %u, normal %.4f %.4f %.4f %.4f, view_z %.6f, depth %.8f\n",
              albedo.at(cx, cy)[0], albedo.at(cx, cy)[1], albedo.at(cx, cy)[2],
              albedo.at(cx, cy)[3], normal.F16(cx, cy, 0), normal.F16(cx, cy, 1),
              normal.F16(cx, cy, 2), normal.F16(cx, cy, 3), view_z0, depth.F32(cx, cy, 0));
  CHECK(albedo.at(cx, cy)[0] == 255);
  CHECK(albedo.at(cx, cy)[1] == 0);
  CHECK(albedo.at(cx, cy)[2] == 0);
  CHECK(albedo.at(cx, cy)[3] == 255);
  CHECK(std::abs(normal.F16(cx, cy, 0)) < 1e-3f);
  CHECK(std::abs(normal.F16(cx, cy, 1)) < 1e-3f);
  CHECK(std::abs(normal.F16(cx, cy, 2) + 1.0f) < 1e-3f);
  CHECK(normal.F16(cx, cy, 3) == 1.0f);
  CHECK(std::abs(view_z0 - 20.0f) < 1e-3f);
  CHECK(std::abs(depth.F32(cx, cy, 0) - ProjectDepth(camera.projection, view_z0)) < 1e-5f);
  CHECK(color.at(cx, cy)[0] == 255);
  CHECK(color.at(cx, cy)[3] == 255);
  // The corner sees the sky: miss values everywhere.
  CHECK(albedo.at(0, 0)[0] == 0);
  CHECK(albedo.at(0, 0)[3] == 0);
  CHECK(normal.F16(0, 0, 3) == 0.0f);
  CHECK(depth.F32(0, 0, 0) == 1.0f);
  CHECK(view_z.F32(0, 0, 0) == kMissViewZ);
  CHECK(IsSky(color.at(0, 0)));
  CHECK(color.at(0, 0)[3] == 255);  // the colour target is opaque everywhere
  // First render: no previous camera, so motion is exactly 0 everywhere. Coverage
  // agrees across the targets.
  uint32_t hits = 0;
  uint32_t first_motion_nonzero = 0;
  uint32_t coverage_mismatch = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const bool hit = is_hit(x, y);
      hits += hit ? 1 : 0;
      if (motion.F32(x, y, 0) != 0.0f || motion.F32(x, y, 1) != 0.0f) {
        ++first_motion_nonzero;
      }
      const bool normal_hit = normal.F16(x, y, 3) == 1.0f;
      const bool view_hit = view_z.F32(x, y, 0) != kMissViewZ;
      const bool depth_hit = depth.F32(x, y, 0) != 1.0f;
      if (normal_hit != hit || view_hit != hit || depth_hit != hit ||
          (!hit && albedo.at(x, y)[0] != 0)) {
        ++coverage_mismatch;
      }
    }
  }
  std::printf("first render: %u hit pixels, %u nonzero motion, %u coverage mismatches\n", hits,
              first_motion_nonzero, coverage_mismatch);
  CHECK(hits > 1000);
  CHECK(first_motion_nonzero == 0);
  CHECK(coverage_mismatch == 0);

  // 2. Same camera with a reverse-Z infinite projection (depth = near / z, miss 0).
  //    The previous camera now exists and equals the current one: motion ~0 on hits.
  {
    CaelixCamera reverse = camera;
    ReverseZInfiniteProjection(reverse.projection, reverse.tan_half_fov_y, reverse.aspect);
    CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
    options.miss_depth = 0.0f;
    Render(rig, renderer, all, reverse, options);
    const float z = view_z.F32(cx, cy, 0);
    std::printf("reverse Z: centre depth %.8f, expected %.8f\n", depth.F32(cx, cy, 0),
                ProjectDepth(reverse.projection, z));
    CHECK(std::abs(depth.F32(cx, cy, 0) - ProjectDepth(reverse.projection, z)) < 1e-5f);
    CHECK(std::abs(depth.F32(cx, cy, 0) - kNear / 20.0f) < 1e-5f);
    CHECK(depth.F32(0, 0, 0) == 0.0f);
    float max_motion = 0.0f;
    uint32_t miss_motion_nonzero = 0;
    for (uint32_t y = 0; y < kSize; ++y) {
      for (uint32_t x = 0; x < kSize; ++x) {
        const float m = std::max(std::abs(motion.F32(x, y, 0)), std::abs(motion.F32(x, y, 1)));
        if (is_hit(x, y)) {
          max_motion = std::max(max_motion, m);
        } else if (m != 0.0f) {
          ++miss_motion_nonzero;
        }
      }
    }
    std::printf("same camera: max |motion| on hits %.6f px\n", max_motion);
    CHECK(max_motion < 1e-3f);
    CHECK(miss_motion_nonzero == 0);
  }

  // 3. The camera moves +1 unit along `right` (+X). A static point at view depth z
  //    then sits 1 unit further LEFT in the new view than in the previous one, i.e.
  //    the previous camera saw it 1 unit to the right: previous view x = current
  //    view x + 1. One unit at depth z spans width / (2 * tan_half_fov_y * aspect * z)
  //    pixels, so motion.x = previous_pixel.x - current_pixel.x = +256 / (2 * 0.57735
  //    * 1 * 20) = +11.085 px, motion.y = 0.
  const CaelixCamera moved = LookPlusZ(5.0f, 4.0f, -20.0f);
  Render(rig, renderer, all, moved, Options(CAELIX_VIEW_ALBEDO));
  const float view_z1 = view_z.F32(cx, cy, 0);
  const float pixels_per_unit =
      static_cast<float>(kSize) / (2.0f * moved.tan_half_fov_y * moved.aspect * view_z1);
  std::printf("camera +1 right: centre motion %.5f %.5f px, expected %.5f 0, view_z %.6f\n",
              motion.F32(cx, cy, 0), motion.F32(cx, cy, 1), pixels_per_unit, view_z1);
  REQUIRE(is_hit(cx, cy));
  CHECK(std::abs(view_z1 - 20.0f) < 1e-3f);
  CHECK(std::abs(motion.F32(cx, cy, 0) - pixels_per_unit) < 1e-2f);
  CHECK(std::abs(motion.F32(cx, cy, 0) - 11.085f) < 2e-2f);
  CHECK(std::abs(motion.F32(cx, cy, 1)) < 1e-3f);

  // 4. Only the depth target (no colour target): G-buffer style call.
  {
    TargetSet depth_only;
    depth_only.t[kDepth] = &depth;
    Render(rig, renderer, depth_only, moved, Options(CAELIX_VIEW_ALBEDO));
    CHECK(std::abs(depth.F32(cx, cy, 0) - ProjectDepth(moved.projection, view_z1)) < 1e-5f);
    CHECK(depth.F32(0, 0, 0) == 1.0f);
  }
  // Argument errors: no target, sizes differ, one image twice, a wrong format.
  {
    ExpectRenderError(rig, renderer, CaelixRenderTargets{}, moved, CAELIX_ERROR_INVALID_ARGUMENT);
    Target small = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT, kSize / 2, kSize / 2);
    CaelixRenderTargets targets{};
    targets.depth = Abi(&depth);
    targets.view_z = Abi(&small);
    ExpectRenderError(rig, renderer, targets, moved, CAELIX_ERROR_INVALID_ARGUMENT);
    targets.view_z = Abi(&depth);
    ExpectRenderError(rig, renderer, targets, moved, CAELIX_ERROR_INVALID_ARGUMENT);
    targets = CaelixRenderTargets{};
    targets.depth = Abi(&color);  // RGBA8 is not an accepted depth format
    ExpectRenderError(rig, renderer, targets, moved, CAELIX_ERROR_INVALID_ARGUMENT);
    DestroyTarget(rig.gpu, small);
  }

  // 5. The entity moves +2 along X; the camera stays (`moved`, also the previous
  //    camera of this call). The flush of this frame writes the new transform and
  //    keeps the old one as the previous rows. The centre ray now hits world x ~ 5 =
  //    entity-local x ~ 3; under the previous transform that point was at world
  //    x ~ 3, i.e. 2 units LEFT of where it is now, so motion.x = -2 * 11.085 =
  //    -22.17 px.
  {
    const float position[3] = {2.0f, 0.0f, 0.0f};
    const float rotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    REQUIRE(caelix_scene_set_entity_transform(rig.scene, rig.entity, position, rotation) ==
            CAELIX_OK);
  }
  Render(rig, renderer, all, moved, Options(CAELIX_VIEW_ALBEDO));
  CHECK(Stats(rig.scene).as_builds_pending == 0);
  const float view_z2 = view_z.F32(cx, cy, 0);
  const float entity_expected =
      -2.0f * static_cast<float>(kSize) / (2.0f * moved.tan_half_fov_y * moved.aspect * view_z2);
  std::printf("entity +2 x: centre motion %.5f %.5f px, expected %.5f 0\n", motion.F32(cx, cy, 0),
              motion.F32(cx, cy, 1), entity_expected);
  REQUIRE(is_hit(cx, cy));
  CHECK(std::abs(motion.F32(cx, cy, 0) - entity_expected) < 2e-2f);
  CHECK(std::abs(motion.F32(cx, cy, 0) + 22.17f) < 4e-2f);
  CHECK(std::abs(motion.F32(cx, cy, 1)) < 1e-3f);

  // 6. Nothing moves: the next flush makes prev = cur again, motion returns to ~0.
  Render(rig, renderer, all, moved, Options(CAELIX_VIEW_ALBEDO));
  std::printf("at rest again: centre motion %.6f %.6f px\n", motion.F32(cx, cy, 0),
              motion.F32(cx, cy, 1));
  CHECK(std::abs(motion.F32(cx, cy, 0)) < 1e-3f);
  CHECK(std::abs(motion.F32(cx, cy, 1)) < 1e-3f);

  REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
  caelix_renderer_release_target_views(renderer);
  caelix_renderer_destroy(renderer);
  for (int i = 0; i < kTargetCount; ++i) {
    DestroyTarget(rig.gpu, *all.t[i]);
  }
  CloseRig(rig);
}

TEST_CASE("random opaque scene: hits, no air boundary") {
  Rig rig;
  OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
  CaelixRenderer* renderer = nullptr;
  REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
  Target target = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);

  std::mt19937 rng(1234);
  World world = RandomOpaqueWorld(rng, 600);
  // Pure magenta (r = b = 31, g = 0) is the air marker: no opaque block may have it.
  for (auto& [key, brick] : world.bricks) {
    for (uint16_t& block : brick) {
      if (block == 0xFC1Fu) {
        block = 0xFC1Eu;
      }
    }
  }
  Upload(rig, world, KeysOf(world));
  DrainBuilds(rig);

  // Keys span about [-40, 40] x [-20, 20] x [-40, 40] bricks = +-320 blocks.
  const CaelixCamera camera = LookPlusZ(0.0f, 0.0f, -900.0f);
  // Unshaded, so an air hit is exactly magenta.
  RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_ALBEDO));
  uint32_t hits = 0;
  uint32_t magenta = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const uint8_t* p = target.at(x, y);
      if (!IsSky(p)) {
        ++hits;
      }
      if (p[0] > 250 && p[1] < 5 && p[2] > 250) {
        ++magenta;
      }
    }
  }
  std::printf("random scene: %u of %u pixels hit, %u magenta\n", hits, kSize * kSize, magenta);
  CHECK(hits > 0);
  CHECK(magenta == 0);

  CaelixRenderOptions shaded = Options(CAELIX_VIEW_ALBEDO);
  shaded.flags = CAELIX_RENDER_FLAG_SHADING;
  RenderColor(rig, renderer, target, camera, shaded);
  for (uint32_t view = CAELIX_VIEW_NORMAL; view <= CAELIX_VIEW_STEPS; ++view) {
    RenderColor(rig, renderer, target, camera, Options(view));
  }

  REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
  caelix_renderer_destroy(renderer);
  DestroyTarget(rig.gpu, target);
  CloseRig(rig);
}
