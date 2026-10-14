// Synthetic scenes and a one-entity scene rig, shared by the GPU tests. Uses the
// public C ABI only.
#pragma once
#include "test_device.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace test {

using Key = std::array<int32_t, 3>;
using Brick = std::array<uint16_t, 512>;  // voxel index x + 8y + 64z

// The host's raw bricks of one entity.
struct World {
  std::map<Key, Brick> bricks;
};

std::vector<Key> KeysOf(const World& world);
// Number of distinct render groups of `keys`.
size_t GroupCount(const std::vector<Key>& keys, uint32_t group_shift);

// A random opaque brick with about 30 % air.
Brick RandomBrick(std::mt19937& rng);
// Blobs of face-connected bricks around random seeds, so that many bricks have
// neighbours, some across group boundaries. Keys around the origin, some negative.
std::vector<Key> RandomBlobs(std::mt19937& rng, size_t count);
// The random opaque scene: `count` RandomBlobs keys with RandomBrick contents.
World RandomOpaqueWorld(std::mt19937& rng, size_t count);
// A solid cube of 3 x 3 x 3 opaque bricks around key (16, 0, 32): it crosses the
// group boundaries x 15|16, y -1|0 and z 31|32 (8 groups at group shift 4).
World SolidCubeWorld(uint32_t seed);

// One device, one scene, one entity.
struct Rig {
  Gpu gpu;
  CaelixDevice* device = nullptr;
  CaelixScene* scene = nullptr;
  CaelixEntityId entity = CAELIX_INVALID_ENTITY;
  Frames frames;
  uint32_t group_shift = 4;
  uint32_t encoding = 0;
};

// An 8 MiB staging ring and the given settings (0 = default).
CaelixSceneDesc RigDesc(uint32_t encoding, uint32_t group_shift, uint32_t records_per_page);
// Sets the library log callback, creates the device, the scene and one entity at
// the origin with no rotation.
void OpenRig(Rig& rig, const char* name, const CaelixSceneDesc& desc);
// Waits for the device, destroys everything and requires no validation or library
// message.
void CloseRig(Rig& rig);

// Uploads `keys` (taken from `world`), flushing whenever the ring is full, then
// flushes once more: the first flush finalizes, the second must find nothing to do.
void Upload(Rig& rig, const World& world, const std::vector<Key>& keys);
// Removes `keys` and flushes.
void Remove(Rig& rig, const std::vector<Key>& keys);

CaelixSceneStats Stats(const CaelixScene* scene);

}  // namespace test
