// The finalize kernel against a CPU reference. Each scenario uploads synthetic
// bricks through the public C ABI, flushes (the flush runs the kernel), reads
// every record and AABB row back from the GPU and compares them word by word
// with Reference(), a straightforward scalar implementation of the rules in
// shaders/caelix_finalize.hlsl.
#include "test_device.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

using namespace test;

namespace {

constexpr uint32_t kRecordWords = 274;
constexpr uint32_t kRecordBytes = kRecordWords * 4;
constexpr uint32_t kAabbWords = 6;
constexpr uint32_t kAabbBytes = kAabbWords * 4;
constexpr uint32_t kQuietNaN = 0x7FC00000u;

using Key = std::array<int32_t, 3>;
using Brick = std::array<uint16_t, 512>;  // voxel index x + 8y + 64z

struct Encoding {
  uint32_t opaque_mask;
  uint32_t id_mask;
  uint32_t face_shift;
};

Encoding EncodingOf(uint32_t encoding) {
  return encoding == 0 ? Encoding{0xC000u, 0x00FFu, 8} : Encoding{0x8000u, 0x01FFu, 9};
}

// The host's raw bricks of one entity.
struct World {
  std::map<Key, Brick> bricks;
};

uint32_t BlockAt(const World& world, Key key, int32_t x, int32_t y, int32_t z) {
  int32_t c[3] = {x, y, z};
  for (int a = 0; a < 3; ++a) {
    if (c[a] < 0) {
      key[a] -= 1;
      c[a] += 8;
    } else if (c[a] >= 8) {
      key[a] += 1;
      c[a] -= 8;
    }
  }
  const auto found = world.bricks.find(key);
  if (found == world.bricks.end()) {
    return 0;  // absent brick = air
  }
  return found->second[c[0] + 8 * c[1] + 64 * c[2]];
}

struct Expected {
  std::array<uint32_t, kRecordWords> record{};
  std::array<uint32_t, kAabbWords> aabb{};
};

uint32_t FloatBits(float value) { return std::bit_cast<uint32_t>(value); }

// The finalized record and AABB row of `key`.
Expected Reference(const World& world, const Key& key, uint32_t group_shift, uint32_t encoding) {
  const Encoding enc = EncodingOf(encoding);
  const auto raw = [&](uint32_t v) { return (v & enc.opaque_mask) != 0 ? v : (v & enc.id_mask); };
  const auto opaque = [&](uint32_t v) { return (v & enc.opaque_mask) != 0; };
  constexpr int32_t kDir[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

  Expected out;
  int32_t lo[3] = {8, 8, 8};
  int32_t hi[3] = {-1, -1, -1};
  for (int32_t z = 0; z < 8; ++z) {
    for (int32_t y = 0; y < 8; ++y) {
      for (int32_t x = 0; x < 8; ++x) {
        const uint32_t v = raw(BlockAt(world, key, x, y, z));
        uint32_t opaque_neighbours = 0;
        uint32_t faces = 0;
        for (uint32_t d = 0; d < 6; ++d) {
          const uint32_t n =
              raw(BlockAt(world, key, x + kDir[d][0], y + kDir[d][1], z + kDir[d][2]));
          if (opaque(n)) {
            ++opaque_neighbours;
          } else if ((n & enc.id_mask) != (v & enc.id_mask)) {
            faces |= 1u << d;
          }
        }
        const bool alive = opaque_neighbours < 6;
        const uint32_t value = opaque(v) ? v : ((v & enc.id_mask) | (faces << enc.face_shift));
        const uint32_t word = 18 + (x >> 1) + 4 * y + 32 * z;
        out.record[word] |= (x & 1) != 0 ? value : (value << 16);
        if (alive && value != 0) {
          const uint32_t coarse = (x >> 2) | ((y >> 2) << 1) | ((z >> 2) << 2);
          const uint32_t micro = (x & 3) | ((y & 3) << 2) | ((z & 3) << 4);
          out.record[2 + coarse * 2 + (micro >> 5)] |= 1u << (micro & 31);
          const int32_t c[3] = {x, y, z};
          for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], c[a]);
            hi[a] = std::max(hi[a], c[a]);
          }
        }
      }
    }
  }
  uint32_t coarse_byte = 0;
  for (uint32_t c = 0; c < 8; ++c) {
    if ((out.record[2 + 2 * c] | out.record[3 + 2 * c]) != 0) {
      coarse_byte |= 1u << c;
    }
  }
  const uint32_t mask = (1u << group_shift) - 1;
  const uint32_t pos = (static_cast<uint32_t>(key[0]) & mask) |
                       ((static_cast<uint32_t>(key[1]) & mask) << group_shift) |
                       ((static_cast<uint32_t>(key[2]) & mask) << (2 * group_shift));
  out.record[0] = (pos & 0xFFFFu) | (coarse_byte << 16);
  const bool empty = hi[0] < 0;
  if (!empty) {
    out.record[1] = static_cast<uint32_t>(lo[0] | (lo[1] << 3) | (lo[2] << 6) | (hi[0] << 9) |
                                          (hi[1] << 12) | (hi[2] << 15));
    for (int a = 0; a < 3; ++a) {
      out.aabb[a] = FloatBits(static_cast<float>(key[a] * 8 + lo[a]));
      out.aabb[3 + a] = FloatBits(static_cast<float>(key[a] * 8 + hi[a] + 1));
    }
  } else {
    out.aabb.fill(kQuietNaN);
  }
  return out;
}

// What the GPU holds for one brick.
struct Actual {
  std::array<uint32_t, kRecordWords> record{};
  std::array<uint32_t, kAabbWords> aabb{};
};

// ---------------------------------------------------------------------------
// Scene helpers

struct Rig {
  Gpu gpu;
  CaelixDevice* device = nullptr;
  CaelixScene* scene = nullptr;
  CaelixEntityId entity = CAELIX_INVALID_ENTITY;
  Frames frames;
  uint32_t group_shift = 4;
  uint32_t encoding = 0;
};

void OpenRig(Rig& rig, const char* name, uint32_t encoding, uint32_t group_shift,
             uint32_t records_per_page) {
  caelix_set_log_callback(LibraryLog, nullptr);
  CreateGpu(rig.gpu, name);
  rig.device = AdoptDevice(rig.gpu);
  CaelixSceneDesc desc{};
  desc.staging_bytes = 8ull << 20;
  desc.records_per_page = records_per_page;
  desc.group_shift = group_shift;
  desc.encoding = encoding;
  REQUIRE(caelix_scene_create(rig.device, &desc, &rig.scene) == CAELIX_OK);
  CaelixEntityDesc entity{};
  entity.rotation[3] = 1.0f;
  REQUIRE(caelix_scene_add_entity(rig.scene, &entity, &rig.entity) == CAELIX_OK);
  rig.group_shift = group_shift;
  rig.encoding = encoding;
}

void CloseRig(Rig& rig) {
  REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
  caelix_scene_destroy(rig.scene);
  caelix_device_destroy(rig.device);
  DestroyGpu(rig.gpu);
  caelix_set_log_callback(nullptr, nullptr);
  RequireNoMessages();
}

// Uploads `keys` (taken from `world`), flushing whenever the ring is full, then
// flushes twice more: the first finalizes, the second must find nothing to do.
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
  CaelixSceneStats before{};
  caelix_scene_get_stats(rig.scene, &before);
  FlushFrame(rig.gpu, rig.scene, rig.frames);
  CaelixSceneStats after{};
  caelix_scene_get_stats(rig.scene, &after);
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

// Reads the record and the AABB row of every brick of `world` from the GPU.
std::map<Key, Actual> ReadBack(Rig& rig, const World& world) {
  std::vector<Key> keys;
  std::vector<CaelixBrickInfo> infos;
  for (const auto& [key, brick] : world.bricks) {
    CaelixBrickInfo info{};
    REQUIRE(caelix_scene_query_brick(rig.scene, rig.entity, CaelixBrickKey{key[0], key[1], key[2]},
                                     &info) == CAELIX_OK);
    INFO("key (" << key[0] << ", " << key[1] << ", " << key[2] << ")");
    REQUIRE(info.resident == 1);
    REQUIRE(info.aabb_buffer != 0);
    keys.push_back(key);
    infos.push_back(info);
  }
  constexpr VkDeviceSize kStride = kRecordBytes + kAabbBytes;
  Readback readback;
  CreateReadback(rig.gpu, std::max<VkDeviceSize>(1, keys.size()) * kStride, readback);
  SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
    for (size_t i = 0; i < keys.size(); ++i) {
      const VkBufferCopy record{infos[i].record_page_offset_bytes, i * kStride, kRecordBytes};
      vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(infos[i].record_page_buffer), readback.buffer,
                      1, &record);
      const VkBufferCopy aabb{infos[i].aabb_row_offset_bytes, i * kStride + kRecordBytes, kAabbBytes};
      vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(infos[i].aabb_buffer), readback.buffer, 1,
                      &aabb);
    }
    CopyToHostBarrier(cmd);
  });
  std::map<Key, Actual> out;
  for (size_t i = 0; i < keys.size(); ++i) {
    const uint32_t* words = readback.words + i * (kStride / 4);
    Actual& actual = out[keys[i]];
    std::copy(words, words + kRecordWords, actual.record.begin());
    std::copy(words + kRecordWords, words + kRecordWords + kAabbWords, actual.aabb.begin());
  }
  DestroyReadback(rig.gpu, readback);
  return out;
}

// Compares every brick of `world` with the reference; fails at the first mismatch.
std::map<Key, Actual> Verify(Rig& rig, const World& world, const char* label) {
  const std::map<Key, Actual> actual = ReadBack(rig, world);
  size_t occupied = 0;
  for (const auto& [key, brick] : world.bricks) {
    const Expected expected = Reference(world, key, rig.group_shift, rig.encoding);
    const Actual& got = actual.at(key);
    for (uint32_t w = 0; w < kRecordWords; ++w) {
      if (got.record[w] != expected.record[w]) {
        FAIL(label << ": key (" << key[0] << ", " << key[1] << ", " << key[2] << ") record word "
                   << w << ": expected 0x" << std::hex << expected.record[w] << ", actual 0x"
                   << got.record[w]);
      }
    }
    for (uint32_t w = 0; w < kAabbWords; ++w) {
      if (got.aabb[w] != expected.aabb[w]) {
        FAIL(label << ": key (" << key[0] << ", " << key[1] << ", " << key[2] << ") AABB word "
                   << w << ": expected 0x" << std::hex << expected.aabb[w] << ", actual 0x"
                   << got.aabb[w]);
      }
    }
    if (expected.record[1] != 0) {
      ++occupied;
    }
  }
  std::printf("[%s] %zu bricks match the reference (%zu with occupied voxels)\n", label,
              world.bricks.size(), occupied);
  std::fflush(stdout);
  return actual;
}

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

// A random opaque brick with about 30 % air.
Brick RandomBrick(std::mt19937& rng) {
  Brick brick{};
  for (uint16_t& block : brick) {
    const uint32_t r = rng();
    block = (r % 100) < 30 ? 0 : static_cast<uint16_t>(0x8000u | ((r >> 8) & 0x7FFFu));
  }
  return brick;
}

// Blobs of face-connected bricks around random seeds, so that many bricks have
// neighbours, some across group boundaries.
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

// Scenario 3/4: a glass slab inside air on an opaque floor. Bricks (0..2, 0, 0)
// are the floor, (0..2, 1, 0) hold the glass slab x 2..21, y 8..11, z 2..5 in air.
World GlassWorld(uint16_t glass, uint16_t floor) {
  World world;
  for (int32_t bx = 0; bx < 3; ++bx) {
    Brick floor_brick{};
    floor_brick.fill(floor);
    world.bricks[{bx, 0, 0}] = floor_brick;
    Brick glass_brick{};
    for (int32_t z = 0; z < 8; ++z) {
      for (int32_t y = 0; y < 8; ++y) {
        for (int32_t x = 0; x < 8; ++x) {
          const int32_t gx = bx * 8 + x;
          const int32_t gy = 8 + y;
          const bool inside = gx >= 2 && gx < 22 && gy >= 8 && gy < 12 && z >= 2 && z < 6;
          glass_brick[x + 8 * y + 64 * z] = inside ? glass : 0;
        }
      }
    }
    world.bricks[{bx, 1, 0}] = glass_brick;
  }
  return world;
}

uint32_t PayloadValue(const Actual& actual, uint32_t x, uint32_t y, uint32_t z) {
  const uint32_t word = actual.record[18 + (x >> 1) + 4 * y + 32 * z];
  return (x & 1) != 0 ? (word & 0xFFFFu) : (word >> 16);
}

bool OccupiedBit(const Actual& actual, uint32_t x, uint32_t y, uint32_t z) {
  const uint32_t coarse = (x >> 2) | ((y >> 2) << 1) | ((z >> 2) << 2);
  const uint32_t micro = (x & 3) | ((y & 3) << 2) | ((z & 3) << 4);
  return (actual.record[2 + coarse * 2 + (micro >> 5)] >> (micro & 31) & 1u) != 0;
}

void GlassScenario(const char* name, uint32_t encoding, uint16_t glass) {
  Rig rig;
  OpenRig(rig, name, encoding, 4, 0);
  const World world = GlassWorld(glass, 0x8123);
  Upload(rig, world, KeysOf(world));
  const std::map<Key, Actual> actual = Verify(rig, world, name);

  const Encoding enc = EncodingOf(encoding);
  const Actual& brick = actual.at({0, 1, 0});
  // Air above the slab top (block y 12 = local y 4): +Y neighbour is air, -Y is
  // glass, so face bit 3 (-Y) is set; it is occupied with id 0.
  const uint32_t air = PayloadValue(brick, 4, 4, 3);
  REQUIRE((air & enc.id_mask) == 0);
  REQUIRE(((air >> enc.face_shift) & 0x3Fu) == (1u << 3));
  REQUIRE(OccupiedBit(brick, 4, 4, 3));
  // Glass on the floor (local y 0): no face bit towards the opaque floor (-Y,
  // bit 3), a face towards the air at -X (bit 1) for local x 2.
  const uint32_t bottom = PayloadValue(brick, 2, 0, 3);
  REQUIRE((bottom & enc.id_mask) == (glass & enc.id_mask));
  REQUIRE(((bottom >> enc.face_shift) & (1u << 3)) == 0);
  REQUIRE(((bottom >> enc.face_shift) & (1u << 1)) != 0);
  // Glass inside the slab: same-medium neighbours, no faces.
  const uint32_t inner = PayloadValue(actual.at({1, 1, 0}), 3, 1, 3);
  REQUIRE(inner == (glass & enc.id_mask));
  // Air far from the glass: id 0, no faces, not occupied.
  REQUIRE(PayloadValue(brick, 0, 7, 7) == 0);
  REQUIRE_FALSE(OccupiedBit(brick, 0, 7, 7));
  CloseRig(rig);
}

}  // namespace

TEST_CASE("finalize: random opaque bricks, then removal, replacement and re-upload",
          "[finalize]") {
  Rig rig;
  // Group shift 3 and small record pages: many groups and several pages.
  OpenRig(rig, "caelix_finalize_test", 0, 3, 128);
  std::mt19937 rng(20261014);
  World world;
  const std::vector<Key> keys = RandomBlobs(rng, 300);
  for (const Key& key : keys) {
    world.bricks[key] = RandomBrick(rng);
  }
  const size_t groups = GroupCount(keys, rig.group_shift);
  bool negative = false;
  for (const Key& key : keys) {
    negative = negative || key[0] < 0 || key[1] < 0 || key[2] < 0;
  }
  std::printf("scenario 1: %zu bricks in %zu groups\n", keys.size(), groups);
  REQUIRE(keys.size() == 300);
  REQUIRE(groups >= 15);
  REQUIRE(negative);
  Upload(rig, world, keys);
  CaelixSceneStats stats{};
  caelix_scene_get_stats(rig.scene, &stats);
  REQUIRE(stats.finalized_bricks_total >= 300);
  REQUIRE(stats.record_pages >= 3);
  Verify(rig, world, "scenario 1 random");

  // Scenario 5: remove 20 bricks, replace 20 others, flush; every brick must
  // match the reference of the updated world (neighbours were re-finalized).
  std::vector<Key> shuffled = KeysOf(world);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  const std::vector<Key> removed(shuffled.begin(), shuffled.begin() + 20);
  const std::vector<Key> replaced(shuffled.begin() + 20, shuffled.begin() + 40);
  for (const Key& key : removed) {
    world.bricks.erase(key);
  }
  for (const Key& key : replaced) {
    world.bricks[key] = RandomBrick(rng);
  }
  Remove(rig, removed);
  Upload(rig, world, replaced);
  const std::map<Key, Actual> before = Verify(rig, world, "scenario 5 after removal and replacement");
  for (const Key& key : removed) {
    CaelixBrickInfo info{};
    REQUIRE(caelix_scene_query_brick(rig.scene, rig.entity, CaelixBrickKey{key[0], key[1], key[2]},
                                     &info) == CAELIX_OK);
    REQUIRE(info.resident == 0);
  }

  // Re-upload 50 unchanged bricks: the lossless rule makes finalization
  // idempotent, so every record stays bit-identical.
  const std::vector<Key> all = KeysOf(world);
  const std::vector<Key> again(all.begin() + 10, all.begin() + 60);
  Upload(rig, world, again);
  const std::map<Key, Actual> after = Verify(rig, world, "scenario 5 after identical re-upload");
  for (const auto& [key, actual] : before) {
    INFO("key (" << key[0] << ", " << key[1] << ", " << key[2] << ")");
    REQUIRE(after.at(key).record == actual.record);
    REQUIRE(after.at(key).aabb == actual.aabb);
  }
  CloseRig(rig);
}

TEST_CASE("finalize: solid cube across group boundaries", "[finalize]") {
  Rig rig;
  OpenRig(rig, "caelix_finalize_test", 0, 4, 0);
  // 3 x 3 x 3 bricks around key (16, 0, 32): crosses the group boundaries x 15|16,
  // y -1|0 and z 31|32.
  World world;
  std::mt19937 rng(7);
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
  REQUIRE(GroupCount(KeysOf(world), 4) == 8);
  Upload(rig, world, KeysOf(world));
  const std::map<Key, Actual> actual = Verify(rig, world, "scenario 2 cube");
  // The interior brick: no voxel alive.
  const Actual& center = actual.at({16, 0, 32});
  REQUIRE((center.record[0] >> 16) == 0);
  REQUIRE(center.record[1] == 0);
  for (const uint32_t word : center.aabb) {
    REQUIRE(word == kQuietNaN);
  }
  // A face brick (-X face): only its x = 0 layer is alive.
  const Actual& face = actual.at({15, 0, 32});
  REQUIRE(face.record[1] == (0u | (0u << 3) | (0u << 6) | (0u << 9) | (7u << 12) | (7u << 15)));
  REQUIRE(face.aabb[0] == FloatBits(120.0f));
  REQUIRE(face.aabb[3] == FloatBits(121.0f));
  REQUIRE(face.aabb[2] == FloatBits(256.0f));
  REQUIRE(face.aabb[5] == FloatBits(264.0f));
  CloseRig(rig);
}

TEST_CASE("finalize: glass slab, packed encoding", "[finalize]") {
  GlassScenario("scenario 3 glass packed", 0, 0x0005);
}

TEST_CASE("finalize: glass slab, block-ID encoding", "[finalize]") {
  GlassScenario("scenario 4 glass block-ID", 1, 0x0105);
}

TEST_CASE("finalize: a neighbour across a group boundary arrives later", "[finalize]") {
  Rig rig;
  OpenRig(rig, "caelix_finalize_test", 0, 4, 0);
  World world;
  Brick solid{};
  solid.fill(0x8001);
  world.bricks[{15, 0, 0}] = solid;
  Upload(rig, world, {{15, 0, 0}});
  const Actual alone = Verify(rig, world, "scenario 6 first brick").at({15, 0, 0});
  // Alone, the +X face (x = 7) borders an absent brick (air): its voxels are alive.
  REQUIRE(OccupiedBit(alone, 7, 3, 3));
  REQUIRE_FALSE(OccupiedBit(alone, 3, 3, 3));

  // The neighbour in the next group arrives: the first brick is finalized again,
  // and its +X face is now buried.
  world.bricks[{16, 0, 0}] = solid;
  Upload(rig, world, {{16, 0, 0}});
  const std::map<Key, Actual> both = Verify(rig, world, "scenario 6 both bricks");
  const Actual& first = both.at({15, 0, 0});
  REQUIRE_FALSE(OccupiedBit(first, 7, 3, 3));
  REQUIRE(first.record != alone.record);
  REQUIRE(((first.record[1] >> 9) & 7u) == 7u);  // max x stays 7 (edges of the x = 7 face)
  CloseRig(rig);
}
