// The primary ray trace kernel (caelix_renderer_render): a one-brick scene and
// the random opaque scene are rendered into an offscreen RGBA8 storage image,
// which is read back and checked pixel by pixel. Uses the public C ABI only.
#include "test_scenes.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace test;

namespace {

constexpr uint32_t kSize = 256;
constexpr float kSky[3] = {0.25f, 0.5f, 0.75f};

struct Image {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};

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

// A kSize x kSize RGBA8 image with the usages a colour target needs, plus TRANSFER_SRC for the
// readback.
Image CreateTarget(Gpu& gpu) {
  Image out;
  VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = VK_FORMAT_R8G8B8A8_UNORM;
  info.extent = {kSize, kSize, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

void DestroyTarget(Gpu& gpu, Image& image) {
  vkDestroyImage(gpu.device.device, image.image, nullptr);
  vkFreeMemory(gpu.device.device, image.memory, nullptr);
}

// Camera at `position` looking along +Z, vertical field of view 60 degrees.
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
  camera.near_z = 0.1f;
  camera.far_z = 1000.0f;
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
  return options;
}

struct Pixels {
  std::vector<uint8_t> rgba;  // kSize * kSize * 4, row 0 = top
  const uint8_t* at(uint32_t x, uint32_t y) const { return rgba.data() + (y * kSize + x) * 4; }
};

// One frame: flush, render, copy the image to a host buffer; returns the pixels.
// `layout` is the image's layout before the render (UNDEFINED the first time,
// TRANSFER_SRC_OPTIMAL after a readback).
Pixels RenderFrame(Rig& rig, CaelixRenderer* renderer, Image& target, VkImageLayout& layout,
                   const CaelixCamera& camera, const CaelixRenderOptions& options) {
  Readback readback;
  CreateReadback(rig.gpu, kSize * kSize * 4, readback);
  const uint64_t frame = rig.frames.next++;
  SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
    CaelixFrameDesc desc{};
    desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
    desc.frame_number = frame;
    desc.safe_frame_number = frame - 1;
    REQUIRE(caelix_scene_flush(rig.scene, &desc) == CAELIX_OK);
    CaelixRenderTargets targets{};
    targets.color.image = reinterpret_cast<uint64_t>(target.image);
    targets.color.format = VK_FORMAT_R8G8B8A8_UNORM;
    targets.color.width = kSize;
    targets.color.height = kSize;
    targets.color.layout = static_cast<uint32_t>(layout);
    REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &options) == CAELIX_OK);

    // GENERAL (render contract) -> TRANSFER_SRC, copy, then make it visible to the host.
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
    region.imageExtent = {kSize, kSize, 1};
    vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, 1, &region);
    CopyToHostBarrier(cmd);
  });
  REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
  layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  Pixels pixels;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(readback.words);
  pixels.rgba.assign(bytes, bytes + kSize * kSize * 4);
  DestroyReadback(rig.gpu, readback);
  return pixels;
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

}  // namespace

TEST_CASE("one red brick: albedo, normal and brick views; sky before the first TLAS") {
  Rig rig;
  OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
  CaelixRenderer* renderer = nullptr;
  REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
  Image target = CreateTarget(rig.gpu);
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  const CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);

  // No bricks, no TLAS: the render clears to the sky colour.
  {
    const Pixels empty = RenderFrame(rig, renderer, target, layout, camera, Options(CAELIX_VIEW_ALBEDO));
    REQUIRE(IsSky(empty.at(kSize / 2, kSize / 2)));
    REQUIRE(IsSky(empty.at(0, 0)));
  }

  // One opaque pure-red brick (packed: bit 15 = opaque, red = 31 << 10) at key (0, 0, 0).
  World world;
  Brick brick{};
  brick.fill(static_cast<uint16_t>(0x8000u | (31u << 10)));
  world.bricks[{0, 0, 0}] = brick;
  Upload(rig, world, KeysOf(world));
  DrainBuilds(rig);

  const Pixels albedo = RenderFrame(rig, renderer, target, layout, camera, Options(CAELIX_VIEW_ALBEDO));
  const uint8_t* centre = albedo.at(kSize / 2, kSize / 2);
  PrintPixel("albedo centre", centre);
  PrintPixel("albedo corner", albedo.at(0, 0));
  CHECK(centre[0] > 200);
  CHECK(centre[1] < 30);
  CHECK(centre[2] < 30);
  CHECK(IsSky(albedo.at(0, 0)));
  CHECK(IsSky(albedo.at(kSize - 1, kSize - 1)));

  // The camera sees the -Z face: normal (0, 0, -1) -> (0.5, 0.5, 0).
  const Pixels normal = RenderFrame(rig, renderer, target, layout, camera, Options(CAELIX_VIEW_NORMAL));
  PrintPixel("normal centre", normal.at(kSize / 2, kSize / 2));
  CHECK(Near(normal.at(kSize / 2, kSize / 2), 0.5f, 0.5f, 0.0f, 2));

  const Pixels bricks = RenderFrame(rig, renderer, target, layout, camera, Options(CAELIX_VIEW_BRICK));
  const uint8_t* brick_centre = bricks.at(kSize / 2, kSize / 2);
  PrintPixel("brick centre", brick_centre);
  CHECK((brick_centre[0] | brick_centre[1] | brick_centre[2]) != 0);
  CHECK(!IsSky(brick_centre));

  // The other views run without validation messages.
  RenderFrame(rig, renderer, target, layout, camera, Options(CAELIX_VIEW_DEPTH));
  RenderFrame(rig, renderer, target, layout, camera, Options(CAELIX_VIEW_GROUP));
  const Pixels steps = RenderFrame(rig, renderer, target, layout, camera, Options(CAELIX_VIEW_STEPS));
  CHECK(!IsSky(steps.at(kSize / 2, kSize / 2)));  // one candidate
  CHECK(IsSky(steps.at(0, 0)));                    // none

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
    targets.color.image = reinterpret_cast<uint64_t>(target.image);
    targets.color.format = VK_FORMAT_R8G8B8A8_UNORM;
    targets.color.width = kSize;
    targets.color.height = kSize;
    REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &bad) ==
            CAELIX_ERROR_INVALID_ARGUMENT);
  });

  REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
  caelix_renderer_release_target_views(renderer);
  caelix_renderer_destroy(renderer);
  DestroyTarget(rig.gpu, target);
  CloseRig(rig);
}

TEST_CASE("random opaque scene: hits, no air boundary") {
  Rig rig;
  OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
  CaelixRenderer* renderer = nullptr;
  REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
  Image target = CreateTarget(rig.gpu);
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;

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
  const Pixels pixels = RenderFrame(rig, renderer, target, layout, camera, Options(CAELIX_VIEW_ALBEDO));
  uint32_t hits = 0;
  uint32_t magenta = 0;
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const uint8_t* p = pixels.at(x, y);
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
  RenderFrame(rig, renderer, target, layout, camera, shaded);
  for (uint32_t view = CAELIX_VIEW_NORMAL; view <= CAELIX_VIEW_STEPS; ++view) {
    RenderFrame(rig, renderer, target, layout, camera, Options(view));
  }

  REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
  caelix_renderer_destroy(renderer);
  DestroyTarget(rig.gpu, target);
  CloseRig(rig);
}
