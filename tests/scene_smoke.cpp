// Headless smoke test of the scene path: upload -> flush -> GPU pools, checked by
// reading records back from the GPU. Uses the PUBLIC C ABI only; vk-bootstrap
// creates the instance and the device the way a host would (tests/test_device.h).
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <vector>

using namespace test;

namespace {

// ---------------------------------------------------------------------------
// Synthetic bricks and the host-side model

constexpr uint32_t kBlocksPerBrick = 512;
constexpr uint32_t kRecordWords = 274;
constexpr uint32_t kRecordBytes = kRecordWords * 4;
constexpr int32_t kGroupShift = 4;  // the scene default

uint32_t Hash(const CaelixBrickKey& key, uint32_t i, uint32_t version)
{
    uint32_t h = static_cast<uint32_t>(key.x) * 73856093u ^ static_cast<uint32_t>(key.y) * 19349663u ^
                 static_cast<uint32_t>(key.z) * 83492791u ^ i * 2654435761u ^ version * 0x9E3779B9u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

uint16_t BlockValue(const CaelixBrickKey& key, uint32_t i, uint32_t version)
{
    return static_cast<uint16_t>(0x8000u | (Hash(key, i, version) & 0x7FFFu));
}

void FillBlocks(const std::vector<CaelixBrickKey>& keys, uint32_t version,
                std::vector<uint16_t>& blocks)
{
    blocks.resize(keys.size() * kBlocksPerBrick);
    for (size_t b = 0; b < keys.size(); ++b)
    {
        for (uint32_t i = 0; i < kBlocksPerBrick; ++i)
        {
            blocks[b * kBlocksPerBrick + i] = BlockValue(keys[b], i, version);
        }
    }
}

uint32_t GroupLocalPos(const CaelixBrickKey& key)
{
    const int32_t mask = (1 << kGroupShift) - 1;
    return static_cast<uint32_t>(key.x & mask) | (static_cast<uint32_t>(key.y & mask) << kGroupShift) |
           (static_cast<uint32_t>(key.z & mask) << (2 * kGroupShift));
}

using GroupId = std::tuple<CaelixEntityId, int32_t, int32_t, int32_t>;
using BrickId = std::tuple<CaelixEntityId, int32_t, int32_t, int32_t>;

GroupId GroupOf(CaelixEntityId entity, const CaelixBrickKey& key)
{
    return {entity, key.x >> kGroupShift, key.y >> kGroupShift, key.z >> kGroupShift};
}

// What the scene should hold: brick -> block value version.
struct Model
{
    std::map<BrickId, uint32_t> bricks;

    void Set(CaelixEntityId entity, const CaelixBrickKey& key, uint32_t version)
    {
        bricks[{entity, key.x, key.y, key.z}] = version;
    }
    void Erase(CaelixEntityId entity, const CaelixBrickKey& key)
    {
        bricks.erase({entity, key.x, key.y, key.z});
    }
    void EraseEntity(CaelixEntityId entity)
    {
        std::erase_if(bricks, [entity](const auto& item) { return std::get<0>(item.first) == entity; });
    }
    std::map<GroupId, uint32_t> GroupCounts() const
    {
        std::map<GroupId, uint32_t> counts;
        for (const auto& [brick, version] : bricks)
        {
            const CaelixBrickKey key{std::get<1>(brick), std::get<2>(brick), std::get<3>(brick)};
            ++counts[GroupOf(std::get<0>(brick), key)];
        }
        return counts;
    }
};

CaelixSceneStats Stats(const CaelixScene* scene)
{
    CaelixSceneStats stats{};
    caelix_scene_get_stats(scene, &stats);
    return stats;
}

void PrintStats(const char* label, const CaelixSceneStats& s)
{
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

// Copies the records of `bricks` from the pools to the host and checks the position
// and every payload word.
void VerifyRecords(Gpu& gpu, const CaelixScene* scene, const Model& model,
                   const std::vector<BrickId>& bricks)
{
    Readback readback;
    CreateReadback(gpu, static_cast<VkDeviceSize>(bricks.size()) * kRecordBytes, readback);
    std::vector<CaelixBrickInfo> infos(bricks.size());
    for (size_t i = 0; i < bricks.size(); ++i)
    {
        const CaelixBrickKey key{std::get<1>(bricks[i]), std::get<2>(bricks[i]), std::get<3>(bricks[i])};
        REQUIRE(caelix_scene_query_brick(scene, std::get<0>(bricks[i]), key, &infos[i]) == CAELIX_OK);
        REQUIRE(infos[i].resident == 1);
        REQUIRE(infos[i].record_page_buffer != 0);
        REQUIRE(infos[i].record_page_address != 0);
    }
    SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
        for (size_t i = 0; i < bricks.size(); ++i)
        {
            const VkBufferCopy region{infos[i].record_page_offset_bytes,
                                      static_cast<VkDeviceSize>(i) * kRecordBytes, kRecordBytes};
            vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(infos[i].record_page_buffer),
                            readback.buffer, 1, &region);
        }
        CopyToHostBarrier(cmd);
    });

    for (size_t b = 0; b < bricks.size(); ++b)
    {
        const CaelixBrickKey key{std::get<1>(bricks[b]), std::get<2>(bricks[b]), std::get<3>(bricks[b])};
        const uint32_t version = model.bricks.at(bricks[b]);
        const uint32_t* record = readback.words + b * kRecordWords;
        INFO("brick entity " << std::get<0>(bricks[b]) << " key (" << key.x << ", " << key.y << ", "
                             << key.z << ") version " << version);
        // Position in bits 0..15. The finalize kernel owns the rest of the header
        // (checked by caelix_finalize_test); the payload of opaque blocks stays raw.
        REQUIRE((record[0] & 0xFFFFu) == GroupLocalPos(key));
        for (uint32_t z = 0; z < 8; ++z)
        {
            for (uint32_t y = 0; y < 8; ++y)
            {
                for (uint32_t x = 0; x < 8; x += 2)
                {
                    const uint32_t i = x + 8 * y + 64 * z;
                    const uint32_t expected = (static_cast<uint32_t>(BlockValue(key, i, version)) << 16) |
                                              BlockValue(key, i + 1, version);
                    const uint32_t word = 18 + (x >> 1) + 4 * y + 32 * z;
                    if (record[word] != expected)
                    {
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
void VerifySlotRows(Gpu& gpu, const CaelixScene* scene, const std::vector<BrickId>& bricks)
{
    constexpr uint32_t kRowBytes = 16;
    Readback readback;
    CreateReadback(gpu, static_cast<VkDeviceSize>(bricks.size()) * kRowBytes, readback);
    std::vector<CaelixBrickInfo> infos(bricks.size());
    for (size_t i = 0; i < bricks.size(); ++i)
    {
        const CaelixBrickKey key{std::get<1>(bricks[i]), std::get<2>(bricks[i]), std::get<3>(bricks[i])};
        REQUIRE(caelix_scene_query_brick(scene, std::get<0>(bricks[i]), key, &infos[i]) == CAELIX_OK);
        REQUIRE(infos[i].resident == 1);
        REQUIRE(infos[i].slot_table_buffer != 0);
    }
    SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
        for (size_t i = 0; i < bricks.size(); ++i)
        {
            const VkBufferCopy region{infos[i].slot_row_offset_bytes,
                                      static_cast<VkDeviceSize>(i) * kRowBytes, kRowBytes};
            vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(infos[i].slot_table_buffer), readback.buffer,
                            1, &region);
        }
        CopyToHostBarrier(cmd);
    });
    for (size_t i = 0; i < bricks.size(); ++i)
    {
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

std::vector<BrickId> Sample(const Model& model, size_t count, uint32_t seed)
{
    std::vector<BrickId> all;
    for (const auto& [brick, brick_version] : model.bricks)
    {
        all.push_back(brick);
    }
    std::mt19937 rng(seed);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(std::min(count, all.size()));
    return all;
}

}  // namespace

TEST_CASE("scene upload, flush, removal and GPU readback", "[scene]")
{
    caelix_set_log_callback(LibraryLog, nullptr);

    CaelixVersion version{};
    caelix_get_version(&version);
    REQUIRE(version.abi == CAELIX_ABI_VERSION);
    REQUIRE(version.abi == 2);

    Gpu gpu;
    CreateGpu(gpu, "caelix_scene_smoke");

    // 1. Adopt the device; create the scene.
    CaelixDevice* device = AdoptDevice(gpu);

    CaelixSceneDesc scene_desc{};
    scene_desc.staging_bytes = 8ull << 20;
    scene_desc.records_per_page = 1024;
    CaelixScene* scene = nullptr;
    REQUIRE(caelix_scene_create(device, &scene_desc, &scene) == CAELIX_OK);
    // The free staging stat of the empty ring: the record space (8 MiB minus the row reserve
    // 8 MiB / 16 + 1 MiB) in 1096-byte records.
    const uint64_t initial_free = Stats(scene).upload_bricks_free;
    std::printf("upload_bricks_free of the empty ring: %llu\n",
                static_cast<unsigned long long>(initial_free));
    REQUIRE(initial_free > 0);
    REQUIRE(initial_free == ((8ull << 20) - (8ull << 20) / 16 - (1ull << 20)) / kRecordBytes);

    ExpectLibraryErrors(4, [&] {
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
        d = CaelixSceneDesc{};
        d.encoding = 2;
        REQUIRE(caelix_scene_create(device, &d, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
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
    for (int32_t k = 0; k < 10; ++k)
    {
        for (int32_t j = 0; j < 20; ++j)
        {
            for (int32_t i = 0; i < 25; ++i)
            {
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
    // Without a flush, the 5000 staged records lower the free stat by at least 5000.
    REQUIRE(Stats(scene).upload_bricks_free + 5000 <= initial_free);
    for (const CaelixBrickKey& key : keys0)
    {
        model.Set(e0, key, 0);
    }
    const size_t groups_e0 = model.GroupCounts().size();
    std::printf("entity 0: %zu bricks in %zu groups\n", keys0.size(), groups_e0);
    REQUIRE(groups_e0 >= 30);
    REQUIRE(Stats(scene).pending_bricks == 5000);

    // Entity 1: a dense 30 x 20 x 20 box = 12000 bricks, more than the ring holds.
    std::vector<CaelixBrickKey> keys1;
    for (int32_t z = -10; z < 10; ++z)
    {
        for (int32_t y = -10; y < 10; ++y)
        {
            for (int32_t x = 0; x < 30; ++x)
            {
                keys1.push_back({x, y, z});
            }
        }
    }
    std::vector<uint16_t> blocks1;
    FillBlocks(keys1, 0, blocks1);
    size_t done = 0;
    int iterations = 0;
    while (done < keys1.size())
    {
        const uint64_t free_before = Stats(scene).upload_bricks_free;
        REQUIRE(caelix_scene_upload_bricks(scene, e1, static_cast<uint32_t>(keys1.size() - done),
                                           keys1.data() + done, blocks1.data() + done * kBlocksPerBrick,
                                           &consumed) == CAELIX_OK);
        std::printf("entity 1 upload %d: consumed %u of %zu\n", iterations, consumed,
                    keys1.size() - done);
        REQUIRE(consumed > 0);
        REQUIRE(consumed <= free_before);  // the stat is an upper bound
        done += consumed;
        FlushFrame(gpu, scene, frames);
        ++iterations;
        REQUIRE(iterations < 100);
    }
    REQUIRE(iterations > 1);  // the batch did not fit one flush
    // Every frame was flushed and recycled: the ring is empty again.
    REQUIRE(Stats(scene).upload_bricks_free == initial_free);
    for (const CaelixBrickKey& key : keys1)
    {
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
    for (size_t i = 7; i < keys0.size() && replaced.size() < 100; i += 41)
    {
        if (GroupOf(e0, keys0[i]) != emptied_group)
        {
            replaced.push_back(keys0[i]);
        }
    }
    REQUIRE(replaced.size() == 100);
    std::vector<uint16_t> replaced_blocks;
    FillBlocks(replaced, 1, replaced_blocks);
    REQUIRE(caelix_scene_upload_bricks(scene, e0, 100, replaced.data(), replaced_blocks.data(),
                                       &consumed) == CAELIX_OK);
    REQUIRE(consumed == 100);
    for (const CaelixBrickKey& key : replaced)
    {
        model.Set(e0, key, 1);
    }
    REQUIRE(Stats(scene).bricks_resident == 17000);
    REQUIRE(Stats(scene).pending_bricks == 100);

    // 3b. Remove 300 bricks of entity 0: all of group (0,0,0), then single bricks of
    //     other groups (never their last brick, never a replaced brick).
    const size_t groups_before = model.GroupCounts().size();
    std::set<BrickId> replaced_set;
    for (const CaelixBrickKey& key : replaced)
    {
        replaced_set.insert({e0, key.x, key.y, key.z});
    }
    std::vector<CaelixBrickKey> removed;
    for (const CaelixBrickKey& key : keys0)
    {
        if (GroupOf(e0, key) == emptied_group)
        {
            removed.push_back(key);
        }
    }
    const size_t emptied_count = removed.size();
    std::printf("group (0,0,0) of entity 0 holds %zu bricks\n", emptied_count);
    REQUIRE(emptied_count > 0);
    {
        std::map<GroupId, uint32_t> counts = model.GroupCounts();
        for (size_t i = 3; i < keys0.size() && removed.size() < 300; i += 13)
        {
            const CaelixBrickKey& key = keys0[i];
            const GroupId group = GroupOf(e0, key);
            if (group == emptied_group || replaced_set.count({e0, key.x, key.y, key.z}) != 0 ||
                counts[group] <= 1)
            {
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
    for (const CaelixBrickKey& key : removed)
    {
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
    for (const auto& [brick, brick_version] : model.bricks)
    {
        survivors.push_back(brick);
    }
    std::mt19937 rng(12345);
    std::shuffle(survivors.begin(), survivors.end(), rng);
    std::vector<BrickId> to_check(survivors.begin(), survivors.begin() + 20);
    for (size_t i = 0; i < 3; ++i)
    {
        to_check.push_back({e0, replaced[i * 30].x, replaced[i * 30].y, replaced[i * 30].z});
    }
    VerifyRecords(gpu, scene, model, to_check);
    VerifySlotRows(gpu, scene, to_check);
    std::printf("verified %zu records and slot rows (3 of them replaced)\n", to_check.size());

    for (size_t i = 0; i < 5; ++i)
    {
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
    for (const CaelixBrickKey& key : keys2)
    {
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
    for (int32_t i = 0; i < 4200; ++i)
    {
        keys3.push_back({16 * (i - 2100), 160, -48});
    }
    std::vector<uint16_t> blocks3;
    FillBlocks(keys3, 3, blocks3);
    done = 0;
    iterations = 0;
    while (done < keys3.size())
    {
        REQUIRE(caelix_scene_upload_bricks(scene, e3, static_cast<uint32_t>(keys3.size() - done),
                                           keys3.data() + done, blocks3.data() + done * kBlocksPerBrick,
                                           &consumed) == CAELIX_OK);
        REQUIRE(consumed > 0);
        done += consumed;
        if (iterations == 0)
        {
            REQUIRE(Stats(scene).slot_table_bytes > table_bytes_before);
        }
        FlushFrame(gpu, scene, frames);
        ++iterations;
        REQUIRE(iterations < 100);
    }
    for (const CaelixBrickKey& key : keys3)
    {
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
    RequireNoMessages();
}
