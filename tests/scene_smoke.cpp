// Headless smoke test of the scene path: upload -> flush -> GPU pools, checked by
// reading records back from the GPU. Uses the PUBLIC C ABI only; vk-bootstrap
// creates the instance and the device the way a host would.
#include <volk.h>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <VkBootstrap.h>
#include <catch2/catch_test_macros.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "caelix/caelix.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Message counters

std::atomic<int> g_validation_errors{0};
std::atomic<int> g_validation_warnings{0};
std::atomic<int> g_validation_ignored{0};
std::atomic<int> g_library_errors{0};
std::atomic<int> g_library_warnings{0};

VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void* /*user_data*/) {
  const char* message = data != nullptr && data->pMessage != nullptr ? data->pMessage : "";
  const char* id = data != nullptr && data->pMessageIdName != nullptr ? data->pMessageIdName : "";
  // Implicit layers of other software (Overwolf) and the loader's notes about them
  // are not this project's problem.
  if (std::strstr(message, "VK_LAYER_OW_") != nullptr || std::strstr(id, "VK_LAYER_OW_") != nullptr ||
      std::strstr(message, "uses API version 1.2") != nullptr) {
    ++g_validation_ignored;
    std::printf("[vulkan ignored] %s\n", message);
    return VK_FALSE;
  }
  if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
    ++g_validation_errors;
    std::printf("[vulkan ERROR] %s: %s\n", id, message);
  } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
    ++g_validation_warnings;
    std::printf("[vulkan WARNING] %s: %s\n", id, message);
  }
  std::fflush(stdout);
  return VK_FALSE;
}

void LibraryLog(CaelixLogLevel level, const char* message, void* /*user_data*/) {
  static const char* const kNames[] = {"debug", "info", "warning", "ERROR"};
  if (level == CAELIX_LOG_ERROR) {
    ++g_library_errors;
  } else if (level == CAELIX_LOG_WARNING) {
    ++g_library_warnings;
  }
  if (level != CAELIX_LOG_DEBUG) {
    std::printf("[caelix %s] %s\n", kNames[level], message);
    std::fflush(stdout);
  }
}

// Runs `body` and requires that it logs exactly `expected` library errors.
template <typename Body>
void ExpectLibraryErrors(int expected, Body body) {
  const int before = g_library_errors.load();
  body();
  REQUIRE(g_library_errors.load() - before == expected);
  g_library_errors -= expected;
}

// ---------------------------------------------------------------------------
// Vulkan setup

struct Gpu {
  vkb::Instance instance;
  vkb::Device device;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
};

void CreateGpu(Gpu& gpu) {
  REQUIRE(volkInitialize() == VK_SUCCESS);
  const vkb::SystemInfo system = vkb::SystemInfo::get_system_info(vkGetInstanceProcAddr).value();
  std::printf("validation layer available: %s, debug utils available: %s\n",
              system.validation_layers_available ? "yes" : "no",
              system.debug_utils_available ? "yes" : "no");

  vkb::InstanceBuilder instance_builder(vkGetInstanceProcAddr);
  instance_builder.set_app_name("caelix_scene_smoke")
      .require_api_version(1, 3, 0)
      .set_headless(true)
      .request_validation_layers(true)
      .set_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
      .set_debug_callback(DebugCallback);
  auto instance = instance_builder.build();
  if (!instance) {
    FAIL("vkb instance: " << instance.error().message());
  }
  gpu.instance = instance.value();
  volkLoadInstanceOnly(gpu.instance.instance);

  CaelixDeviceRequirements requirements{};
  caelix_get_device_requirements(&requirements);
  REQUIRE(requirements.api_version >= VK_MAKE_API_VERSION(0, 1, 3, 0));

  // Translate the library's feature bits to Vulkan feature structs.
  const uint32_t flags = requirements.feature_flags;
  const uint32_t known = CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS | CAELIX_FEATURE_DESCRIPTOR_INDEXING |
                         CAELIX_FEATURE_SYNCHRONIZATION2 | CAELIX_FEATURE_DYNAMIC_RENDERING |
                         CAELIX_FEATURE_TIMELINE_SEMAPHORE | CAELIX_FEATURE_ACCELERATION_STRUCTURE |
                         CAELIX_FEATURE_RAY_QUERY | CAELIX_FEATURE_SHADER_INT64 |
                         CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT;
  REQUIRE((flags & ~known) == 0);  // a new bit needs a translation here
  VkPhysicalDeviceFeatures f10{};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR ray_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  if (flags & CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS) f12.bufferDeviceAddress = VK_TRUE;
  if (flags & CAELIX_FEATURE_DESCRIPTOR_INDEXING) {
    f12.runtimeDescriptorArray = VK_TRUE;
    f12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    f12.descriptorBindingPartiallyBound = VK_TRUE;
    f12.descriptorBindingVariableDescriptorCount = VK_TRUE;
  }
  if (flags & CAELIX_FEATURE_SYNCHRONIZATION2) f13.synchronization2 = VK_TRUE;
  if (flags & CAELIX_FEATURE_DYNAMIC_RENDERING) f13.dynamicRendering = VK_TRUE;
  if (flags & CAELIX_FEATURE_TIMELINE_SEMAPHORE) f12.timelineSemaphore = VK_TRUE;
  if (flags & CAELIX_FEATURE_ACCELERATION_STRUCTURE) acceleration.accelerationStructure = VK_TRUE;
  if (flags & CAELIX_FEATURE_RAY_QUERY) ray_query.rayQuery = VK_TRUE;
  if (flags & CAELIX_FEATURE_SHADER_INT64) f10.shaderInt64 = VK_TRUE;
  if (flags & CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT) f12.scalarBlockLayout = VK_TRUE;

  vkb::PhysicalDeviceSelector selector(gpu.instance);
  selector.set_minimum_version(1, 3)
      .require_present(false)
      .set_required_features(f10)
      .set_required_features_12(f12)
      .set_required_features_13(f13);
  for (uint32_t i = 0; i < requirements.device_extension_count; ++i) {
    selector.add_required_extension(requirements.device_extensions[i]);
  }
  if (flags & CAELIX_FEATURE_ACCELERATION_STRUCTURE) selector.add_required_extension_features(acceleration);
  if (flags & CAELIX_FEATURE_RAY_QUERY) selector.add_required_extension_features(ray_query);
  auto physical = selector.select();
  if (!physical) {
    FAIL("vkb physical device: " << physical.error().message());
  }
  std::printf("physical device: %s\n", physical.value().name.c_str());

  auto device = vkb::DeviceBuilder(physical.value()).build();
  if (!device) {
    FAIL("vkb device: " << device.error().message());
  }
  gpu.device = device.value();
  volkLoadDevice(gpu.device.device);
  gpu.queue = gpu.device.get_queue(vkb::QueueType::graphics).value();
  gpu.queue_family = gpu.device.get_queue_index(vkb::QueueType::graphics).value();

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  pool_info.queueFamilyIndex = gpu.queue_family;
  REQUIRE(vkCreateCommandPool(gpu.device.device, &pool_info, nullptr, &gpu.pool) == VK_SUCCESS);
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  REQUIRE(vkCreateFence(gpu.device.device, &fence_info, nullptr, &gpu.fence) == VK_SUCCESS);
}

void DestroyGpu(Gpu& gpu) {
  vkDestroyFence(gpu.device.device, gpu.fence, nullptr);
  vkDestroyCommandPool(gpu.device.device, gpu.pool, nullptr);
  vkb::destroy_device(gpu.device);
  vkb::destroy_instance(gpu.instance);
}

// Records `body` into a fresh command buffer, submits it and waits for it.
template <typename Body>
void SubmitAndWait(Gpu& gpu, Body body) {
  VkCommandBufferAllocateInfo allocate_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate_info.commandPool = gpu.pool;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  REQUIRE(vkAllocateCommandBuffers(gpu.device.device, &allocate_info, &cmd) == VK_SUCCESS);
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  REQUIRE(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
  body(cmd);
  REQUIRE(vkEndCommandBuffer(cmd) == VK_SUCCESS);
  VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
  cmd_info.commandBuffer = cmd;
  VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  submit.commandBufferInfoCount = 1;
  submit.pCommandBufferInfos = &cmd_info;
  REQUIRE(vkQueueSubmit2(gpu.queue, 1, &submit, gpu.fence) == VK_SUCCESS);
  REQUIRE(vkWaitForFences(gpu.device.device, 1, &gpu.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);
  REQUIRE(vkResetFences(gpu.device.device, 1, &gpu.fence) == VK_SUCCESS);
  REQUIRE(vkResetCommandPool(gpu.device.device, gpu.pool, 0) == VK_SUCCESS);
  vkFreeCommandBuffers(gpu.device.device, gpu.pool, 1, &cmd);
}

// A host-visible, coherent buffer for reading records back.
struct Readback {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  const uint32_t* words = nullptr;
};

void CreateReadback(Gpu& gpu, VkDeviceSize size, Readback& out) {
  const VkDevice device = gpu.device.device;
  VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  REQUIRE(vkCreateBuffer(device, &buffer_info, nullptr, &out.buffer) == VK_SUCCESS);
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
  VkPhysicalDeviceMemoryProperties memory_properties{};
  vkGetPhysicalDeviceMemoryProperties(gpu.device.physical_device.physical_device, &memory_properties);
  const VkMemoryPropertyFlags wanted =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
        (memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted) {
      type = i;
      break;
    }
  }
  REQUIRE(type != UINT32_MAX);
  VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = type;
  REQUIRE(vkAllocateMemory(device, &allocate_info, nullptr, &out.memory) == VK_SUCCESS);
  REQUIRE(vkBindBufferMemory(device, out.buffer, out.memory, 0) == VK_SUCCESS);
  void* mapped = nullptr;
  REQUIRE(vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS);
  out.words = static_cast<const uint32_t*>(mapped);
}

void DestroyReadback(Gpu& gpu, Readback& readback) {
  vkUnmapMemory(gpu.device.device, readback.memory);
  vkDestroyBuffer(gpu.device.device, readback.buffer, nullptr);
  vkFreeMemory(gpu.device.device, readback.memory, nullptr);
}

// ---------------------------------------------------------------------------
// Synthetic bricks and the host-side model

constexpr uint32_t kBlocksPerBrick = 512;
constexpr uint32_t kRecordWords = 274;
constexpr uint32_t kRecordBytes = kRecordWords * 4;
constexpr int32_t kGroupShift = 4;  // the scene default

uint32_t Hash(const CaelixBrickKey& key, uint32_t i, uint32_t version) {
  uint32_t h = static_cast<uint32_t>(key.x) * 73856093u ^ static_cast<uint32_t>(key.y) * 19349663u ^
               static_cast<uint32_t>(key.z) * 83492791u ^ i * 2654435761u ^ version * 0x9E3779B9u;
  h ^= h >> 16;
  h *= 0x7FEB352Du;
  h ^= h >> 15;
  h *= 0x846CA68Bu;
  h ^= h >> 16;
  return h;
}

uint16_t BlockValue(const CaelixBrickKey& key, uint32_t i, uint32_t version) {
  return static_cast<uint16_t>(0x8000u | (Hash(key, i, version) & 0x7FFFu));
}

void FillBlocks(const std::vector<CaelixBrickKey>& keys, uint32_t version,
                std::vector<uint16_t>& blocks) {
  blocks.resize(keys.size() * kBlocksPerBrick);
  for (size_t b = 0; b < keys.size(); ++b) {
    for (uint32_t i = 0; i < kBlocksPerBrick; ++i) {
      blocks[b * kBlocksPerBrick + i] = BlockValue(keys[b], i, version);
    }
  }
}

uint32_t GroupLocalPos(const CaelixBrickKey& key) {
  const int32_t mask = (1 << kGroupShift) - 1;
  return static_cast<uint32_t>(key.x & mask) | (static_cast<uint32_t>(key.y & mask) << kGroupShift) |
         (static_cast<uint32_t>(key.z & mask) << (2 * kGroupShift));
}

using GroupId = std::tuple<CaelixEntityId, int32_t, int32_t, int32_t>;
using BrickId = std::tuple<CaelixEntityId, int32_t, int32_t, int32_t>;

GroupId GroupOf(CaelixEntityId entity, const CaelixBrickKey& key) {
  return {entity, key.x >> kGroupShift, key.y >> kGroupShift, key.z >> kGroupShift};
}

// What the scene should hold: brick -> block value version.
struct Model {
  std::map<BrickId, uint32_t> bricks;

  void Set(CaelixEntityId entity, const CaelixBrickKey& key, uint32_t version) {
    bricks[{entity, key.x, key.y, key.z}] = version;
  }
  void Erase(CaelixEntityId entity, const CaelixBrickKey& key) {
    bricks.erase({entity, key.x, key.y, key.z});
  }
  void EraseEntity(CaelixEntityId entity) {
    std::erase_if(bricks, [entity](const auto& item) { return std::get<0>(item.first) == entity; });
  }
  std::map<GroupId, uint32_t> GroupCounts() const {
    std::map<GroupId, uint32_t> counts;
    for (const auto& [brick, version] : bricks) {
      const CaelixBrickKey key{std::get<1>(brick), std::get<2>(brick), std::get<3>(brick)};
      ++counts[GroupOf(std::get<0>(brick), key)];
    }
    return counts;
  }
};

CaelixSceneStats Stats(const CaelixScene* scene) {
  CaelixSceneStats stats{};
  caelix_scene_get_stats(scene, &stats);
  return stats;
}

void PrintStats(const char* label, const CaelixSceneStats& s) {
  std::printf(
      "[stats %s] entities %u, groups %u, resident %llu, known %llu, record pages %llu (%llu bytes), "
      "slot table %llu bytes, staging %llu bytes, uploaded total %llu, pending %llu\n",
      label, s.entities, s.groups, static_cast<unsigned long long>(s.bricks_resident),
      static_cast<unsigned long long>(s.bricks_known), static_cast<unsigned long long>(s.record_pages),
      static_cast<unsigned long long>(s.record_bytes),
      static_cast<unsigned long long>(s.slot_table_bytes),
      static_cast<unsigned long long>(s.staging_bytes),
      static_cast<unsigned long long>(s.uploaded_bricks_total),
      static_cast<unsigned long long>(s.pending_bricks));
  std::fflush(stdout);
}

// Frame bookkeeping of the test host: every submit is waited for, so the frame
// before the current one is always complete.
struct Frames {
  uint64_t next = 1;
};

void FlushFrame(Gpu& gpu, CaelixScene* scene, Frames& frames) {
  const uint64_t frame = frames.next++;
  SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
    CaelixFrameDesc desc{};
    desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
    desc.frame_number = frame;
    desc.safe_frame_number = frame - 1;
    REQUIRE(caelix_scene_flush(scene, &desc) == CAELIX_OK);
  });
  // The frame has completed: its staging space can be reused right away.
  REQUIRE(caelix_scene_recycle(scene, frame) == CAELIX_OK);
}

// Copies the records of `bricks` from the pools to the host and checks every word.
void VerifyRecords(Gpu& gpu, const CaelixScene* scene, const Model& model,
                   const std::vector<BrickId>& bricks) {
  Readback readback;
  CreateReadback(gpu, static_cast<VkDeviceSize>(bricks.size()) * kRecordBytes, readback);
  std::vector<CaelixBrickInfo> infos(bricks.size());
  for (size_t i = 0; i < bricks.size(); ++i) {
    const CaelixBrickKey key{std::get<1>(bricks[i]), std::get<2>(bricks[i]), std::get<3>(bricks[i])};
    REQUIRE(caelix_scene_query_brick(scene, std::get<0>(bricks[i]), key, &infos[i]) == CAELIX_OK);
    REQUIRE(infos[i].resident == 1);
    REQUIRE(infos[i].record_page_buffer != 0);
    REQUIRE(infos[i].record_page_address != 0);
  }
  SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
    for (size_t i = 0; i < bricks.size(); ++i) {
      const VkBufferCopy region{infos[i].record_page_offset_bytes,
                                static_cast<VkDeviceSize>(i) * kRecordBytes, kRecordBytes};
      vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(infos[i].record_page_buffer),
                      readback.buffer, 1, &region);
    }
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
  });

  for (size_t b = 0; b < bricks.size(); ++b) {
    const CaelixBrickKey key{std::get<1>(bricks[b]), std::get<2>(bricks[b]), std::get<3>(bricks[b])};
    const uint32_t version = model.bricks.at(bricks[b]);
    const uint32_t* record = readback.words + b * kRecordWords;
    INFO("brick entity " << std::get<0>(bricks[b]) << " key (" << key.x << ", " << key.y << ", "
                         << key.z << ") version " << version);
    REQUIRE(record[0] == GroupLocalPos(key));  // pos in bits 0..15, coarse occupancy 0
    for (uint32_t w = 1; w < 18; ++w) {
      REQUIRE(record[w] == 0);
    }
    for (uint32_t z = 0; z < 8; ++z) {
      for (uint32_t y = 0; y < 8; ++y) {
        for (uint32_t x = 0; x < 8; x += 2) {
          const uint32_t i = x + 8 * y + 64 * z;
          const uint32_t expected = (static_cast<uint32_t>(BlockValue(key, i, version)) << 16) |
                                    BlockValue(key, i + 1, version);
          const uint32_t word = 18 + (x >> 1) + 4 * y + 32 * z;
          if (record[word] != expected) {
            FAIL("payload word " << word << " is " << record[word] << ", expected " << expected);
          }
        }
      }
    }
  }
  DestroyReadback(gpu, readback);
}

// Copies the slot table rows of `bricks` to the host and checks them against
// caelix_scene_query_brick: state resident, group-local position, record index.
void VerifySlotRows(Gpu& gpu, const CaelixScene* scene, const std::vector<BrickId>& bricks) {
  constexpr uint32_t kRowBytes = 16;
  Readback readback;
  CreateReadback(gpu, static_cast<VkDeviceSize>(bricks.size()) * kRowBytes, readback);
  std::vector<CaelixBrickInfo> infos(bricks.size());
  for (size_t i = 0; i < bricks.size(); ++i) {
    const CaelixBrickKey key{std::get<1>(bricks[i]), std::get<2>(bricks[i]), std::get<3>(bricks[i])};
    REQUIRE(caelix_scene_query_brick(scene, std::get<0>(bricks[i]), key, &infos[i]) == CAELIX_OK);
    REQUIRE(infos[i].resident == 1);
    REQUIRE(infos[i].slot_table_buffer != 0);
  }
  SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
    for (size_t i = 0; i < bricks.size(); ++i) {
      const VkBufferCopy region{infos[i].slot_row_offset_bytes,
                                static_cast<VkDeviceSize>(i) * kRowBytes, kRowBytes};
      vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(infos[i].slot_table_buffer), readback.buffer,
                      1, &region);
    }
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
  });
  for (size_t i = 0; i < bricks.size(); ++i) {
    const CaelixBrickKey key{std::get<1>(bricks[i]), std::get<2>(bricks[i]), std::get<3>(bricks[i])};
    const uint32_t* row = readback.words + i * 4;
    INFO("slot row of entity " << std::get<0>(bricks[i]) << " key (" << key.x << ", " << key.y
                               << ", " << key.z << ")");
    REQUIRE(row[0] == (GroupLocalPos(key) | (1u << 24)));
    REQUIRE(row[1] == 0);
    REQUIRE(row[2] == infos[i].record_index);
    REQUIRE(row[3] == 0);
  }
  DestroyReadback(gpu, readback);
}

std::vector<BrickId> Sample(const Model& model, size_t count, uint32_t seed) {
  std::vector<BrickId> all;
  for (const auto& [brick, brick_version] : model.bricks) {
    all.push_back(brick);
  }
  std::mt19937 rng(seed);
  std::shuffle(all.begin(), all.end(), rng);
  all.resize(std::min(count, all.size()));
  return all;
}

}  // namespace

TEST_CASE("scene upload, flush, removal and GPU readback", "[scene]") {
  caelix_set_log_callback(LibraryLog, nullptr);

  CaelixVersion version{};
  caelix_get_version(&version);
  REQUIRE(version.abi == CAELIX_ABI_VERSION);
  REQUIRE(version.abi == 2);

  Gpu gpu;
  CreateGpu(gpu);

  // 1. Adopt the device; create the scene.
  CaelixDeviceAdoptDesc adopt{};
  adopt.get_instance_proc_addr = reinterpret_cast<void*>(vkGetInstanceProcAddr);
  adopt.instance = reinterpret_cast<uint64_t>(gpu.instance.instance);
  adopt.physical_device = reinterpret_cast<uint64_t>(gpu.device.physical_device.physical_device);
  adopt.device = reinterpret_cast<uint64_t>(gpu.device.device);
  adopt.graphics_queue = reinterpret_cast<uint64_t>(gpu.queue);
  adopt.graphics_queue_family = gpu.queue_family;
  CaelixDevice* device = nullptr;
  REQUIRE(caelix_device_adopt(&adopt, &device) == CAELIX_OK);

  CaelixSceneDesc scene_desc{};
  scene_desc.staging_bytes = 8ull << 20;
  scene_desc.records_per_page = 1024;
  CaelixScene* scene = nullptr;
  REQUIRE(caelix_scene_create(device, &scene_desc, &scene) == CAELIX_OK);

  ExpectLibraryErrors(3, [&] {
    CaelixScene* bad = nullptr;
    CaelixSceneDesc d{};
    d.staging_bytes = 1 << 20;
    REQUIRE(caelix_scene_create(device, &d, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    d = CaelixSceneDesc{};
    d.block_bytes = 4;
    REQUIRE(caelix_scene_create(device, &d, &bad) == CAELIX_ERROR_UNSUPPORTED);
    d = CaelixSceneDesc{};
    d.group_shift = 6;
    REQUIRE(caelix_scene_create(device, &d, &bad) == CAELIX_ERROR_UNSUPPORTED);
    REQUIRE(bad == nullptr);
  });

  // 2. Two entities.
  CaelixEntityDesc entity_desc{};
  entity_desc.rotation[3] = 1.0f;
  CaelixEntityId e0 = CAELIX_INVALID_ENTITY;
  CaelixEntityId e1 = CAELIX_INVALID_ENTITY;
  REQUIRE(caelix_scene_add_entity(scene, &entity_desc, &e0) == CAELIX_OK);
  entity_desc.guid[0] = 1;
  entity_desc.position[0] = 100.0f;
  REQUIRE(caelix_scene_add_entity(scene, &entity_desc, &e1) == CAELIX_OK);
  REQUIRE(e0 != e1);
  const float position[3] = {1.0f, 2.0f, 3.0f};
  const float rotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  REQUIRE(caelix_scene_set_entity_transform(scene, e1, position, rotation) == CAELIX_OK);

  Model model;
  Frames frames;

  // Entity 0: a sparse lattice around the origin, 25 x 20 x 10 = 5000 bricks.
  std::vector<CaelixBrickKey> keys0;
  for (int32_t k = 0; k < 10; ++k) {
    for (int32_t j = 0; j < 20; ++j) {
      for (int32_t i = 0; i < 25; ++i) {
        keys0.push_back({-50 + 4 * i, -40 + 4 * j, -20 + 4 * k});
      }
    }
  }
  std::vector<uint16_t> blocks0;
  FillBlocks(keys0, 0, blocks0);
  uint32_t consumed = 0;
  REQUIRE(caelix_scene_upload_bricks(scene, e0, static_cast<uint32_t>(keys0.size()), keys0.data(),
                                     blocks0.data(), &consumed) == CAELIX_OK);
  REQUIRE(consumed == keys0.size());
  for (const CaelixBrickKey& key : keys0) {
    model.Set(e0, key, 0);
  }
  const size_t groups_e0 = model.GroupCounts().size();
  std::printf("entity 0: %zu bricks in %zu groups\n", keys0.size(), groups_e0);
  REQUIRE(groups_e0 >= 30);
  REQUIRE(Stats(scene).pending_bricks == 5000);

  // Entity 1: a dense 30 x 20 x 20 box = 12000 bricks, more than the ring holds.
  std::vector<CaelixBrickKey> keys1;
  for (int32_t z = -10; z < 10; ++z) {
    for (int32_t y = -10; y < 10; ++y) {
      for (int32_t x = 0; x < 30; ++x) {
        keys1.push_back({x, y, z});
      }
    }
  }
  std::vector<uint16_t> blocks1;
  FillBlocks(keys1, 0, blocks1);
  size_t done = 0;
  int iterations = 0;
  while (done < keys1.size()) {
    REQUIRE(caelix_scene_upload_bricks(scene, e1, static_cast<uint32_t>(keys1.size() - done),
                                       keys1.data() + done, blocks1.data() + done * kBlocksPerBrick,
                                       &consumed) == CAELIX_OK);
    std::printf("entity 1 upload %d: consumed %u of %zu\n", iterations, consumed,
                keys1.size() - done);
    REQUIRE(consumed > 0);
    done += consumed;
    FlushFrame(gpu, scene, frames);
    ++iterations;
    REQUIRE(iterations < 100);
  }
  REQUIRE(iterations > 1);  // the batch did not fit one flush
  for (const CaelixBrickKey& key : keys1) {
    model.Set(e1, key, 0);
  }
  CaelixSceneStats stats = Stats(scene);
  PrintStats("after upload", stats);
  REQUIRE(stats.bricks_resident == 17000);
  REQUIRE(stats.groups == model.GroupCounts().size());
  REQUIRE(stats.pending_bricks == 0);
  REQUIRE(stats.uploaded_bricks_total == 17000);

  // 3a. Replace 100 bricks of entity 0 (outside group (0,0,0), which is emptied below).
  const GroupId emptied_group{e0, 0, 0, 0};
  std::vector<CaelixBrickKey> replaced;
  for (size_t i = 7; i < keys0.size() && replaced.size() < 100; i += 41) {
    if (GroupOf(e0, keys0[i]) != emptied_group) {
      replaced.push_back(keys0[i]);
    }
  }
  REQUIRE(replaced.size() == 100);
  std::vector<uint16_t> replaced_blocks;
  FillBlocks(replaced, 1, replaced_blocks);
  REQUIRE(caelix_scene_upload_bricks(scene, e0, 100, replaced.data(), replaced_blocks.data(),
                                     &consumed) == CAELIX_OK);
  REQUIRE(consumed == 100);
  for (const CaelixBrickKey& key : replaced) {
    model.Set(e0, key, 1);
  }
  REQUIRE(Stats(scene).bricks_resident == 17000);
  REQUIRE(Stats(scene).pending_bricks == 100);

  // 3b. Remove 300 bricks of entity 0: all of group (0,0,0), then single bricks of
  //     other groups (never their last brick, never a replaced brick).
  const size_t groups_before = model.GroupCounts().size();
  std::set<BrickId> replaced_set;
  for (const CaelixBrickKey& key : replaced) {
    replaced_set.insert({e0, key.x, key.y, key.z});
  }
  std::vector<CaelixBrickKey> removed;
  for (const CaelixBrickKey& key : keys0) {
    if (GroupOf(e0, key) == emptied_group) {
      removed.push_back(key);
    }
  }
  const size_t emptied_count = removed.size();
  std::printf("group (0,0,0) of entity 0 holds %zu bricks\n", emptied_count);
  REQUIRE(emptied_count > 0);
  {
    std::map<GroupId, uint32_t> counts = model.GroupCounts();
    for (size_t i = 3; i < keys0.size() && removed.size() < 300; i += 13) {
      const CaelixBrickKey& key = keys0[i];
      const GroupId group = GroupOf(e0, key);
      if (group == emptied_group || replaced_set.count({e0, key.x, key.y, key.z}) != 0 ||
          counts[group] <= 1) {
        continue;
      }
      --counts[group];
      removed.push_back(key);
    }
  }
  REQUIRE(removed.size() == 300);
  // Two calls, plus keys the entity does not have (ignored).
  REQUIRE(caelix_scene_remove_bricks(scene, e0, 150, removed.data()) == CAELIX_OK);
  REQUIRE(caelix_scene_remove_bricks(scene, e0, 150, removed.data() + 150) == CAELIX_OK);
  const CaelixBrickKey absent[2] = {{1000, 1000, 1000}, {-49, -40, -20}};
  REQUIRE(caelix_scene_remove_bricks(scene, e0, 2, absent) == CAELIX_OK);
  for (const CaelixBrickKey& key : removed) {
    model.Erase(e0, key);
  }
  FlushFrame(gpu, scene, frames);

  stats = Stats(scene);
  PrintStats("after replace and remove", stats);
  REQUIRE(stats.bricks_resident == 5000 + 12000 - 300);
  REQUIRE(stats.bricks_known == 5000 + 12000 - 300);
  REQUIRE(stats.groups == groups_before - 1);
  REQUIRE(stats.groups == model.GroupCounts().size());
  REQUIRE(stats.record_pages >= 17);
  REQUIRE(stats.pending_bricks == 0);
  REQUIRE(stats.entities == 2);

  // 4. Read records back from the GPU.
  std::vector<BrickId> survivors;
  for (const auto& [brick, brick_version] : model.bricks) {
    survivors.push_back(brick);
  }
  std::mt19937 rng(12345);
  std::shuffle(survivors.begin(), survivors.end(), rng);
  std::vector<BrickId> to_check(survivors.begin(), survivors.begin() + 20);
  for (size_t i = 0; i < 3; ++i) {
    to_check.push_back({e0, replaced[i * 30].x, replaced[i * 30].y, replaced[i * 30].z});
  }
  VerifyRecords(gpu, scene, model, to_check);
  VerifySlotRows(gpu, scene, to_check);
  std::printf("verified %zu records and slot rows (3 of them replaced)\n", to_check.size());

  for (size_t i = 0; i < 5; ++i) {
    CaelixBrickInfo info{};
    info.resident = 7;
    REQUIRE(caelix_scene_query_brick(scene, e0, removed[i * 60], &info) == CAELIX_OK);
    REQUIRE(info.resident == 0);
    REQUIRE(info.record_index == 0);
    REQUIRE(info.record_page_buffer == 0);
  }

  // Invalid arguments.
  ExpectLibraryErrors(5, [&] {
    CaelixBrickInfo info{};
    REQUIRE(caelix_scene_query_brick(scene, 77, keys0[0], &info) == CAELIX_ERROR_INVALID_ARGUMENT);
    const CaelixBrickKey far_key{1 << 20, 0, 0};
    REQUIRE(caelix_scene_upload_bricks(scene, e0, 1, &far_key, blocks0.data(), &consumed) ==
            CAELIX_ERROR_INVALID_ARGUMENT);
    REQUIRE(consumed == 0);
    REQUIRE(caelix_scene_remove_bricks(scene, CAELIX_INVALID_ENTITY, 1, keys0.data()) ==
            CAELIX_ERROR_INVALID_ARGUMENT);
    SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
      CaelixFrameDesc desc{};
      desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
      desc.frame_number = frames.next - 1;  // not above the previous flush
      desc.safe_frame_number = 0;
      REQUIRE(caelix_scene_flush(scene, &desc) == CAELIX_ERROR_INVALID_ARGUMENT);
      desc.frame_number = frames.next;
      desc.safe_frame_number = frames.next;  // safe must be below frame
      REQUIRE(caelix_scene_flush(scene, &desc) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
  });

  // 5. Remove entity 1.
  const uint64_t pages_before = stats.record_pages;
  REQUIRE(caelix_scene_remove_entity(scene, e1) == CAELIX_OK);
  model.EraseEntity(e1);
  stats = Stats(scene);
  REQUIRE(stats.bricks_resident == 4700);
  REQUIRE(stats.entities == 1);
  REQUIRE(stats.groups == model.GroupCounts().size());
  ExpectLibraryErrors(1, [&] {
    REQUIRE(caelix_scene_upload_bricks(scene, e1, 1, keys1.data(), blocks1.data(), &consumed) ==
            CAELIX_ERROR_INVALID_ARGUMENT);
  });
  FlushFrame(gpu, scene, frames);
  PrintStats("after removing entity 1", Stats(scene));

  // A new entity reuses the freed id and the freed records: no new page.
  CaelixEntityId e2 = CAELIX_INVALID_ENTITY;
  REQUIRE(caelix_scene_add_entity(scene, &entity_desc, &e2) == CAELIX_OK);
  REQUIRE(e2 == e1);
  std::vector<CaelixBrickKey> keys2(keys1.begin(), keys1.begin() + 1000);
  std::vector<uint16_t> blocks2;
  FillBlocks(keys2, 2, blocks2);
  REQUIRE(caelix_scene_upload_bricks(scene, e2, 1000, keys2.data(), blocks2.data(), &consumed) ==
          CAELIX_OK);
  REQUIRE(consumed == 1000);
  for (const CaelixBrickKey& key : keys2) {
    model.Set(e2, key, 2);
  }
  FlushFrame(gpu, scene, frames);
  stats = Stats(scene);
  PrintStats("after entity 2", stats);
  REQUIRE(stats.bricks_resident == 5700);
  REQUIRE(stats.record_pages == pages_before);
  VerifyRecords(gpu, scene, model,
                {BrickId{e2, keys2[0].x, keys2[0].y, keys2[0].z},
                 BrickId{e2, keys2[499].x, keys2[499].y, keys2[499].z},
                 BrickId{e2, keys2[999].x, keys2[999].y, keys2[999].z}});
  VerifySlotRows(gpu, scene, Sample(model, 30, 4242));

  // 6. Slot table growth: 4200 bricks, each in its own group, need 4200 * 16 rows
  //    more than the initial 65536-row table holds. The flush copies the old table
  //    into the new one; rows written before the growth must survive it.
  const uint64_t table_bytes_before = stats.slot_table_bytes;
  CaelixEntityId e3 = CAELIX_INVALID_ENTITY;
  REQUIRE(caelix_scene_add_entity(scene, &entity_desc, &e3) == CAELIX_OK);
  std::vector<CaelixBrickKey> keys3;
  for (int32_t i = 0; i < 4200; ++i) {
    keys3.push_back({16 * (i - 2100), 160, -48});
  }
  std::vector<uint16_t> blocks3;
  FillBlocks(keys3, 3, blocks3);
  done = 0;
  iterations = 0;
  while (done < keys3.size()) {
    REQUIRE(caelix_scene_upload_bricks(scene, e3, static_cast<uint32_t>(keys3.size() - done),
                                       keys3.data() + done, blocks3.data() + done * kBlocksPerBrick,
                                       &consumed) == CAELIX_OK);
    REQUIRE(consumed > 0);
    done += consumed;
    if (iterations == 0) {
      REQUIRE(Stats(scene).slot_table_bytes > table_bytes_before);
    }
    FlushFrame(gpu, scene, frames);
    ++iterations;
    REQUIRE(iterations < 100);
  }
  for (const CaelixBrickKey& key : keys3) {
    model.Set(e3, key, 3);
  }
  stats = Stats(scene);
  PrintStats("after slot table growth", stats);
  REQUIRE(stats.slot_table_bytes > table_bytes_before);
  REQUIRE(stats.groups == model.GroupCounts().size());
  REQUIRE(stats.bricks_resident == 5700 + 4200);
  {
    const std::vector<BrickId> sample = Sample(model, 40, 777);
    VerifyRecords(gpu, scene, model, sample);
    VerifySlotRows(gpu, scene, sample);
    std::printf("verified %zu records and slot rows after the slot table growth\n", sample.size());
  }

  // Teardown: scene, adopted device, then the host's Vulkan objects.
  REQUIRE(vkDeviceWaitIdle(gpu.device.device) == VK_SUCCESS);
  caelix_scene_destroy(scene);
  caelix_device_destroy(device);
  DestroyGpu(gpu);
  caelix_set_log_callback(nullptr, nullptr);

  std::printf("validation messages: %d errors, %d warnings, %d ignored; library: %d unexpected "
              "errors, %d warnings\n",
              g_validation_errors.load(), g_validation_warnings.load(), g_validation_ignored.load(),
              g_library_errors.load(), g_library_warnings.load());
  REQUIRE(g_validation_errors.load() == 0);
  REQUIRE(g_validation_warnings.load() == 0);
  REQUIRE(g_library_errors.load() == 0);
  REQUIRE(g_library_warnings.load() == 0);
}
