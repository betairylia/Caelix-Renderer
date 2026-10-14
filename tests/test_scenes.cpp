#include "test_scenes.h"

#include <set>

namespace test {

std::vector<Key> KeysOf(const World& world) {
  std::vector<Key> keys;
  for (const auto& [key, brick] : world.bricks) {
    keys.push_back(key);
  }
  return keys;
}

size_t GroupCount(const std::vector<Key>& keys, uint32_t group_shift) {
  std::set<Key> groups;
  for (const Key& key : keys) {
    groups.insert({key[0] >> group_shift, key[1] >> group_shift, key[2] >> group_shift});
  }
  return groups.size();
}

Brick RandomBrick(std::mt19937& rng) {
  Brick brick{};
  for (uint16_t& block : brick) {
    const uint32_t r = rng();
    block = (r % 100) < 30 ? 0 : static_cast<uint16_t>(0x8000u | ((r >> 8) & 0x7FFFu));
  }
  return brick;
}

std::vector<Key> RandomBlobs(std::mt19937& rng, size_t count) {
  std::set<Key> set;
  std::vector<Key> keys;
  constexpr int32_t kDir[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  while (keys.size() < count) {
    std::vector<Key> blob;
    blob.push_back(Key{static_cast<int32_t>(rng() % 80) - 40, static_cast<int32_t>(rng() % 40) - 20,
                       static_cast<int32_t>(rng() % 80) - 40});
    if (!set.insert(blob[0]).second) {
      continue;
    }
    keys.push_back(blob[0]);
    for (int k = 0; k < 29 && keys.size() < count; ++k) {
      const Key& from = blob[rng() % blob.size()];
      const uint32_t d = rng() % 6;
      const Key next{from[0] + kDir[d][0], from[1] + kDir[d][1], from[2] + kDir[d][2]};
      if (set.insert(next).second) {
        blob.push_back(next);
        keys.push_back(next);
      }
    }
  }
  return keys;
}

World RandomOpaqueWorld(std::mt19937& rng, size_t count) {
  World world;
  for (const Key& key : RandomBlobs(rng, count)) {
    world.bricks[key] = RandomBrick(rng);
  }
  return world;
}

World SolidCubeWorld(uint32_t seed) {
  World world;
  std::mt19937 rng(seed);
  for (int32_t z = 31; z <= 33; ++z) {
    for (int32_t y = -1; y <= 1; ++y) {
      for (int32_t x = 15; x <= 17; ++x) {
        Brick brick{};
        for (uint16_t& block : brick) {
          block = static_cast<uint16_t>(0xC000u | (rng() & 0x3FFFu));
        }
        world.bricks[{x, y, z}] = brick;
      }
    }
  }
  return world;
}

CaelixSceneDesc RigDesc(uint32_t encoding, uint32_t group_shift, uint32_t records_per_page) {
  CaelixSceneDesc desc{};
  desc.staging_bytes = 8ull << 20;
  desc.records_per_page = records_per_page;
  desc.group_shift = group_shift;
  desc.encoding = encoding;
  return desc;
}

void OpenRig(Rig& rig, const char* name, const CaelixSceneDesc& desc) {
  caelix_set_log_callback(LibraryLog, nullptr);
  CreateGpu(rig.gpu, name);
  rig.device = AdoptDevice(rig.gpu);
  REQUIRE(caelix_scene_create(rig.device, &desc, &rig.scene) == CAELIX_OK);
  CaelixEntityDesc entity{};
  entity.rotation[3] = 1.0f;
  REQUIRE(caelix_scene_add_entity(rig.scene, &entity, &rig.entity) == CAELIX_OK);
  rig.group_shift = desc.group_shift != 0 ? desc.group_shift : 4;
  rig.encoding = desc.encoding;
}

void CloseRig(Rig& rig) {
  REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
  caelix_scene_destroy(rig.scene);
  caelix_device_destroy(rig.device);
  DestroyGpu(rig.gpu);
  caelix_set_log_callback(nullptr, nullptr);
  RequireNoMessages();
}

void Upload(Rig& rig, const World& world, const std::vector<Key>& keys) {
  std::vector<CaelixBrickKey> brick_keys;
  std::vector<uint16_t> blocks;
  for (const Key& key : keys) {
    brick_keys.push_back({key[0], key[1], key[2]});
    const Brick& brick = world.bricks.at(key);
    blocks.insert(blocks.end(), brick.begin(), brick.end());
  }
  size_t done = 0;
  int iterations = 0;
  while (done < brick_keys.size()) {
    uint32_t consumed = 0;
    REQUIRE(caelix_scene_upload_bricks(rig.scene, rig.entity,
                                       static_cast<uint32_t>(brick_keys.size() - done),
                                       brick_keys.data() + done, blocks.data() + done * 512,
                                       &consumed) == CAELIX_OK);
    done += consumed;
    FlushFrame(rig.gpu, rig.scene, rig.frames);
    REQUIRE(++iterations < 100);
  }
  const CaelixSceneStats before = Stats(rig.scene);
  FlushFrame(rig.gpu, rig.scene, rig.frames);
  const CaelixSceneStats after = Stats(rig.scene);
  REQUIRE(after.finalized_bricks_total == before.finalized_bricks_total);
}

void Remove(Rig& rig, const std::vector<Key>& keys) {
  std::vector<CaelixBrickKey> brick_keys;
  for (const Key& key : keys) {
    brick_keys.push_back({key[0], key[1], key[2]});
  }
  REQUIRE(caelix_scene_remove_bricks(rig.scene, rig.entity, static_cast<uint32_t>(brick_keys.size()),
                                     brick_keys.data()) == CAELIX_OK);
  FlushFrame(rig.gpu, rig.scene, rig.frames);
}

CaelixSceneStats Stats(const CaelixScene* scene) {
  CaelixSceneStats stats{};
  caelix_scene_get_stats(scene, &stats);
  return stats;
}

}  // namespace test
