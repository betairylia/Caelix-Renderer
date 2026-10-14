// Acceleration structures: the BLAS per render group and the scene TLAS that
// caelix_scene_flush builds after the finalize kernel. There is no trace kernel
// yet, so the test proves the builds through the validation layers, through
// acceleration structure property queries (serialization size) on the GPU, and
// through the scene stats. Uses the public C ABI only.
#include "test_scenes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <vector>

using namespace test;

namespace {

// Flushes until every acceleration structure is up to date; returns the number
// of flushes.
int Drain(Rig& rig, int limit = 64) {
  int flushes = 0;
  while (Stats(rig.scene).as_builds_pending > 0) {
    FlushFrame(rig.gpu, rig.scene, rig.frames);
    REQUIRE(++flushes < limit);
  }
  return flushes;
}

// One flush that is submitted and waited for, WITHOUT caelix_scene_recycle of its
// frame afterwards. Returns the frame number.
uint64_t FlushWithoutRecycle(Rig& rig) {
  const uint64_t frame = rig.frames.next++;
  SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
    CaelixFrameDesc desc{};
    desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
    desc.frame_number = frame;
    desc.safe_frame_number = frame - 1;
    REQUIRE(caelix_scene_flush(rig.scene, &desc) == CAELIX_OK);
  });
  return frame;
}

struct Tlas {
  uint64_t handle = 0;
  uint64_t address = 0;
};

Tlas GetTlas(const Rig& rig) {
  Tlas tlas;
  REQUIRE(caelix_scene_get_tlas(rig.scene, &tlas.handle, &tlas.address) == CAELIX_OK);
  return tlas;
}

CaelixBrickInfo Query(const Rig& rig, const Key& key) {
  CaelixBrickInfo info{};
  REQUIRE(caelix_scene_query_brick(rig.scene, rig.entity, CaelixBrickKey{key[0], key[1], key[2]},
                                   &info) == CAELIX_OK);
  return info;
}

// The BLAS handle of the first `count` distinct groups of `world` (in key order).
std::vector<uint64_t> GroupBlasHandles(const Rig& rig, const World& world, size_t count) {
  std::set<uint32_t> seen;
  std::vector<uint64_t> handles;
  for (const auto& [key, brick] : world.bricks) {
    const CaelixBrickInfo info = Query(rig, key);
    REQUIRE(info.resident == 1);
    if (seen.insert(info.group_index).second) {
      REQUIRE(info.blas_handle != 0);
      REQUIRE(info.blas_address != 0);
      handles.push_back(info.blas_handle);
      if (handles.size() == count) {
        break;
      }
    }
  }
  return handles;
}

// Every brick of `world` is resident and its group has a BLAS.
void RequireEveryGroupBuilt(const Rig& rig, const World& world) {
  for (const auto& [key, brick] : world.bricks) {
    const CaelixBrickInfo info = Query(rig, key);
    INFO("key (" << key[0] << ", " << key[1] << ", " << key[2] << ")");
    REQUIRE(info.resident == 1);
    REQUIRE(info.blas_handle != 0);
    REQUIRE(info.blas_address != 0);
  }
}

// The serialization size of each acceleration structure, written by
// vkCmdWriteAccelerationStructuresPropertiesKHR on the GPU.
std::vector<uint64_t> SerializationSizes(Gpu& gpu, const std::vector<uint64_t>& handles) {
  const VkDevice device = gpu.device.device;
  const uint32_t count = static_cast<uint32_t>(handles.size());
  VkQueryPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  pool_info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
  pool_info.queryCount = count;
  VkQueryPool pool = VK_NULL_HANDLE;
  REQUIRE(vkCreateQueryPool(device, &pool_info, nullptr, &pool) == VK_SUCCESS);
  std::vector<VkAccelerationStructureKHR> structures;
  for (const uint64_t handle : handles) {
    structures.push_back(reinterpret_cast<VkAccelerationStructureKHR>(handle));
  }
  SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
    vkCmdResetQueryPool(cmd, pool, 0, count);
    vkCmdWriteAccelerationStructuresPropertiesKHR(
        cmd, count, structures.data(),
        VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR, pool, 0);
  });
  std::vector<uint64_t> sizes(count);
  REQUIRE(vkGetQueryPoolResults(device, pool, 0, count, count * sizeof(uint64_t), sizes.data(),
                                sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS);
  vkDestroyQueryPool(device, pool, nullptr);
  return sizes;
}

// Property queries of the BLASes of up to `count` groups and of the TLAS: every
// size must be nonzero.
void RequireBuiltOnGpu(Rig& rig, const World& world, size_t count, const char* label) {
  std::vector<uint64_t> handles = GroupBlasHandles(rig, world, count);
  const Tlas tlas = GetTlas(rig);
  REQUIRE(tlas.handle != 0);
  handles.push_back(tlas.handle);
  const std::vector<uint64_t> sizes = SerializationSizes(rig.gpu, handles);
  std::printf("[%s] serialization sizes:", label);
  for (size_t i = 0; i < sizes.size(); ++i) {
    std::printf(" %s%llu", i + 1 == sizes.size() ? "TLAS " : "",
                static_cast<unsigned long long>(sizes[i]));
    REQUIRE(sizes[i] > 0);
  }
  std::printf("\n");
  std::fflush(stdout);
}

void PrintAsStats(const char* label, const CaelixSceneStats& s) {
  std::printf("[%s] groups %u, blas_count %llu, blas_bytes %llu, tlas_bytes %llu, scratch %llu, "
              "builds total %llu, pending %llu\n",
              label, s.groups, static_cast<unsigned long long>(s.blas_count),
              static_cast<unsigned long long>(s.blas_bytes),
              static_cast<unsigned long long>(s.tlas_bytes),
              static_cast<unsigned long long>(s.scratch_bytes),
              static_cast<unsigned long long>(s.as_builds_total),
              static_cast<unsigned long long>(s.as_builds_pending));
  std::fflush(stdout);
}

// The EntityInfo row of the rig's entity (32 words), read back from the GPU.
std::vector<float> ReadEntityRows(Rig& rig, const Key& any_key) {
  const CaelixBrickInfo info = Query(rig, any_key);
  REQUIRE(info.entity_info_buffer != 0);
  Readback readback;
  CreateReadback(rig.gpu, 128, readback);
  SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
    const VkBufferCopy region{info.entity_info_offset_bytes, 0, 128};
    vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(info.entity_info_buffer), readback.buffer, 1,
                    &region);
    CopyToHostBarrier(cmd);
  });
  // Words 8..19 = cur rows, 20..31 = prev rows.
  std::vector<float> rows(24);
  std::memcpy(rows.data(), readback.words + 8, 24 * sizeof(float));
  DestroyReadback(rig.gpu, readback);
  return rows;
}

// Object-to-world rows of a position and a unit quaternion (x, y, z, w).
std::vector<float> ExpectedRows(const float p[3], const float q[4]) {
  const float x = q[0], y = q[1], z = q[2], w = q[3];
  return {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),     p[0],
          2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),     p[1],
          2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y), p[2]};
}

void RequireRows(const std::vector<float>& rows, size_t first, const std::vector<float>& expected,
                 const char* what) {
  for (size_t i = 0; i < 12; ++i) {
    INFO(what << " element " << i << ": expected " << expected[i] << ", actual " << rows[first + i]);
    REQUIRE(std::fabs(rows[first + i] - expected[i]) < 1e-6f);
  }
}

}  // namespace

TEST_CASE("acceleration structures: random scene, updates and entity removal", "[as]") {
  Rig rig;
  // Group shift 3 and small record pages, like the finalize test: ~20 groups.
  OpenRig(rig, "caelix_as_test", RigDesc(0, 3, 128));
  std::mt19937 rng(20261014);
  World world = RandomOpaqueWorld(rng, 300);
  const size_t groups = GroupCount(KeysOf(world), rig.group_shift);
  std::printf("random scene: %zu bricks in %zu groups\n", world.bricks.size(), groups);
  REQUIRE(groups >= 15);

  // 1. Every group gets a BLAS, the scene a TLAS.
  REQUIRE(GetTlas(rig).handle == 0);
  Upload(rig, world, KeysOf(world));
  std::printf("random scene: %d extra flushes until no build is pending\n", Drain(rig));
  CaelixSceneStats stats = Stats(rig.scene);
  PrintAsStats("random scene", stats);
  REQUIRE(stats.groups == groups);
  REQUIRE(stats.blas_count == groups);
  REQUIRE(stats.blas_bytes > 0);
  REQUIRE(stats.tlas_bytes > 0);
  REQUIRE(stats.scratch_bytes == (128ull << 20));
  REQUIRE(stats.as_builds_total >= groups + 1);
  const Tlas tlas = GetTlas(rig);
  REQUIRE(tlas.handle != 0);
  REQUIRE(tlas.address != 0);
  RequireEveryGroupBuilt(rig, world);

  // 2. The structures exist on the GPU: property queries of 5 BLASes and the TLAS.
  RequireBuiltOnGpu(rig, world, 5, "random scene");

  // 3. Updates: remove 20 bricks, replace 20, add 30 in new groups.
  std::vector<Key> shuffled = KeysOf(world);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  const std::vector<Key> removed(shuffled.begin(), shuffled.begin() + 20);
  const std::vector<Key> replaced(shuffled.begin() + 20, shuffled.begin() + 40);
  std::vector<Key> added;
  for (int32_t i = 0; i < 30; ++i) {
    // One brick per new group (group edge 8 bricks), far from the blobs.
    added.push_back(Key{200 + 8 * (i % 6), 100, 200 + 8 * (i / 6)});
  }
  for (const Key& key : removed) {
    world.bricks.erase(key);
  }
  std::vector<Key> uploads = replaced;
  uploads.insert(uploads.end(), added.begin(), added.end());
  for (const Key& key : uploads) {
    world.bricks[key] = RandomBrick(rng);
  }
  const uint64_t builds_before = stats.as_builds_total;
  Remove(rig, removed);
  Upload(rig, world, uploads);
  Drain(rig);
  stats = Stats(rig.scene);
  PrintAsStats("after updates", stats);
  const size_t groups_after = GroupCount(KeysOf(world), rig.group_shift);
  REQUIRE(groups_after > groups);  // 30 new one-brick groups, a few emptied by the removal
  REQUIRE(stats.groups == groups_after);
  REQUIRE(stats.blas_count == groups_after);
  REQUIRE(stats.as_builds_total > builds_before);
  RequireEveryGroupBuilt(rig, world);
  for (const Key& key : removed) {
    REQUIRE(Query(rig, key).resident == 0);
  }
  RequireBuiltOnGpu(rig, world, 5, "after updates");

  // 4. Remove the entity: the BLASes are released only after the frame of the TLAS
  //    rebuild that drops them has completed.
  const CaelixSceneStats before_removal = Stats(rig.scene);
  REQUIRE(caelix_scene_remove_entity(rig.scene, rig.entity) == CAELIX_OK);
  stats = Stats(rig.scene);
  REQUIRE(stats.blas_count == 0);
  REQUIRE(stats.blas_bytes == before_removal.blas_bytes);
  REQUIRE(stats.as_builds_pending == 1);  // the TLAS
  const uint64_t frame = FlushWithoutRecycle(rig);
  stats = Stats(rig.scene);
  PrintAsStats("entity removed, flushed, frame not yet safe", stats);
  REQUIRE(stats.blas_bytes == before_removal.blas_bytes);
  REQUIRE(stats.as_builds_pending == 0);
  REQUIRE(stats.as_builds_total == before_removal.as_builds_total + 1);
  REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
  stats = Stats(rig.scene);
  PrintAsStats("entity removed, frame safe", stats);
  REQUIRE(stats.blas_bytes == 0);
  REQUIRE(stats.blas_bytes < before_removal.blas_bytes);
  REQUIRE(stats.tlas_bytes > 0);  // an empty TLAS
  CloseRig(rig);
}

TEST_CASE("acceleration structures: solid cube across group boundaries", "[as]") {
  Rig rig;
  OpenRig(rig, "caelix_as_test", RigDesc(0, 4, 0));
  const World world = SolidCubeWorld(7);
  REQUIRE(GroupCount(KeysOf(world), 4) == 8);
  Upload(rig, world, KeysOf(world));
  Drain(rig);
  const CaelixSceneStats stats = Stats(rig.scene);
  PrintAsStats("solid cube", stats);
  REQUIRE(stats.groups == 8);
  REQUIRE(stats.blas_count == 8);
  REQUIRE(stats.tlas_bytes > 0);
  const Tlas tlas = GetTlas(rig);
  REQUIRE(tlas.handle != 0);
  REQUIRE(tlas.address != 0);
  RequireEveryGroupBuilt(rig, world);
  RequireBuiltOnGpu(rig, world, 8, "solid cube");
  CloseRig(rig);
}

TEST_CASE("acceleration structures: a small scratch budget spreads the builds over flushes",
          "[as]") {
  Rig rig;
  // Group shift 1 (2^3 bricks per group): bricks two keys apart are each alone in
  // a group, so every brick needs its own BLAS build and its own scratch.
  CaelixSceneDesc desc = RigDesc(0, 1, 0);
  desc.staging_bytes = 16ull << 20;
  desc.scratch_bytes = 4ull << 20;
  OpenRig(rig, "caelix_as_test", desc);
  std::mt19937 rng(99);
  World world;
  for (int32_t z = 0; z < 40; ++z) {
    for (int32_t x = 0; x < 50; ++x) {
      world.bricks[{2 * x, 0, 2 * z}] = RandomBrick(rng);
    }
  }
  const std::vector<Key> keys = KeysOf(world);
  REQUIRE(GroupCount(keys, 1) == 2000);

  std::vector<CaelixBrickKey> brick_keys;
  std::vector<uint16_t> blocks;
  for (const Key& key : keys) {
    brick_keys.push_back({key[0], key[1], key[2]});
    const Brick& brick = world.bricks.at(key);
    blocks.insert(blocks.end(), brick.begin(), brick.end());
  }
  uint32_t consumed = 0;
  REQUIRE(caelix_scene_upload_bricks(rig.scene, rig.entity, static_cast<uint32_t>(keys.size()),
                                     brick_keys.data(), blocks.data(), &consumed) == CAELIX_OK);
  REQUIRE(consumed == keys.size());
  FlushFrame(rig.gpu, rig.scene, rig.frames);
  CaelixSceneStats stats = Stats(rig.scene);
  PrintAsStats("scratch budget, first flush", stats);
  REQUIRE(stats.scratch_bytes == (4ull << 20));
  REQUIRE(stats.as_builds_pending > 0);
  REQUIRE(stats.blas_count > 0);
  REQUIRE(stats.blas_count < 2000);
  const int flushes = Drain(rig);
  stats = Stats(rig.scene);
  PrintAsStats("scratch budget, drained", stats);
  std::printf("scratch budget: %d more flushes until no build is pending\n", flushes);
  REQUIRE(flushes >= 1);
  REQUIRE(stats.blas_count == 2000);
  REQUIRE(stats.groups == 2000);
  REQUIRE(stats.scratch_bytes == (4ull << 20));  // no single build needed more
  REQUIRE(stats.as_builds_total >= 2000 + 1 + static_cast<uint64_t>(flushes));
  RequireEveryGroupBuilt(rig, world);
  RequireBuiltOnGpu(rig, world, 5, "scratch budget");
  CloseRig(rig);
}

TEST_CASE("acceleration structures: a transform change rebuilds the TLAS, prev rows follow",
          "[as]") {
  Rig rig;
  OpenRig(rig, "caelix_as_test", RigDesc(0, 4, 0));
  const World world = SolidCubeWorld(11);
  Upload(rig, world, KeysOf(world));
  Drain(rig);
  const Key probe = world.bricks.begin()->first;

  const float p0[3] = {0.0f, 0.0f, 0.0f};
  const float q0[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> rows = ReadEntityRows(rig, probe);
  RequireRows(rows, 0, ExpectedRows(p0, q0), "initial cur");
  RequireRows(rows, 12, ExpectedRows(p0, q0), "initial prev");

  // 90 degrees about +Y, and a move.
  const float s = std::sqrt(0.5f);
  const float p1[3] = {10.0f, -5.0f, 3.5f};
  const float q1[4] = {0.0f, s, 0.0f, s};
  const CaelixSceneStats before = Stats(rig.scene);
  REQUIRE(before.as_builds_pending == 0);
  REQUIRE(caelix_scene_set_entity_transform(rig.scene, rig.entity, p1, q1) == CAELIX_OK);
  REQUIRE(Stats(rig.scene).as_builds_pending == 1);
  FlushFrame(rig.gpu, rig.scene, rig.frames);
  CaelixSceneStats stats = Stats(rig.scene);
  REQUIRE(stats.as_builds_total == before.as_builds_total + 1);  // the TLAS only
  REQUIRE(stats.as_builds_pending == 0);
  REQUIRE(stats.blas_count == before.blas_count);
  rows = ReadEntityRows(rig, probe);
  RequireRows(rows, 0, ExpectedRows(p1, q1), "cur after the change");
  RequireRows(rows, 12, ExpectedRows(p0, q0), "prev after the change");

  // One flush later prev == cur, and nothing is rebuilt.
  FlushFrame(rig.gpu, rig.scene, rig.frames);
  stats = Stats(rig.scene);
  REQUIRE(stats.as_builds_total == before.as_builds_total + 1);
  rows = ReadEntityRows(rig, probe);
  RequireRows(rows, 0, ExpectedRows(p1, q1), "cur one flush later");
  RequireRows(rows, 12, ExpectedRows(p1, q1), "prev one flush later");
  RequireBuiltOnGpu(rig, world, 2, "after the transform change");
  CloseRig(rig);
}
