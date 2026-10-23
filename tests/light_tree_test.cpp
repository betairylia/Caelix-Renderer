// The light tree (docs/LIGHT_TREE.md) through the public C ABI and a probe kernel that runs the
// device API (tests/shaders/caelix_light_tree_probe.hlsl):
//   - a CPU reference of every light face (exposure across bricks and groups) against the group
//     rows, the block headers, leaves, brick light records, face entries and internal nodes;
//   - normalisation: the sum of LightTreePointPdf over every light face is 1 for every receiver;
//   - sample / pdf consistency, the drawn point on its face, a histogram against N * pdf, the bound
//     property (a face that can contribute has a positive pdf);
//   - updates (bricks inside a group and across a group border, min_emission up and down, an
//     entity moved and rotated, an entity removed) with the generation rule, record copies, heap
//     growth from a tiny heap, the packed encoding;
//   - [.bench]: draws, the group pass and the BVH builds at about 10k, 54k and 1M light bricks.
#include "test_kernels.h"
#include "test_scenes.h"

#include "scene/gpu_structs.h"
#include "scene/light_tree_layout.h"
#include "scene/record_layout.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#ifndef CAELIX_TEST_SHADER_DIR
#error "CAELIX_TEST_SHADER_DIR must name the build's shader folder"
#endif

namespace {

using test::Brick;
using test::Key;
using test::World;

// The GPU layouts are the library's own (lib/src/scene/light_tree_layout.h); the test reads them
// through memory.
using caelix::BrickLightRecordHeader;
using caelix::GroupInfo;
using caelix::GroupLightRow;
using caelix::LightBvhLeaf;
using caelix::LightBvhNode;
using caelix::LightControl;
using caelix::LightIndexEntry;
using caelix::LightScanPush;
using caelix::LightSortPush;
using caelix::LightTreeDescriptor;
using caelix::LightTreeHeader;
using caelix::kLtChildLeafBit;
using caelix::kLtNone;
using caelix::kLtUnitBytes;

uint32_t FamilyCount(uint32_t lo, uint32_t hi, uint32_t f)
{
    const uint32_t word = f < caelix::kLtFamiliesPerCountWord ? lo : hi;
    const uint32_t shift = caelix::kLtFamilyCountBits * (f % caelix::kLtFamiliesPerCountWord);
    return (word >> shift) & caelix::kLtFamilyCountMask;
}

uint32_t PackVoxel(uint32_t x, uint32_t y, uint32_t z)
{
    return x | (y << caelix::kLtVoxelBits) | (z << (2 * caelix::kLtVoxelBits));
}

uint32_t UnpackVoxel(uint32_t packed, int axis)
{
    return (packed >> (caelix::kLtVoxelBits * axis)) & caelix::kLtVoxelMask;
}

// Bytes of a brick light record with `entries` face entries (whole units).
uint64_t RecordBytes(uint32_t entries)
{
    const uint64_t bytes = caelix::kLtRecordHeaderBytes + uint64_t{caelix::kLtFaceEntryBytes} * entries;
    return (bytes + kLtUnitBytes - 1) / kLtUnitBytes * kLtUnitBytes;
}

uint32_t Part1By2(uint32_t v)
{
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

// ---------------------------------------------------------------------------
// Host-visible buffers with a device address, and the probe kernel

// The shared helpers (test_kernels.h); kernels load from this build's shader folder.
using test::CreateHostBuffer;
using test::DestroyHostBuffer;
using test::HostBuffer;

class ComputeKernel : public test::ComputeKernel
{
public:
    ComputeKernel(test::Gpu& gpu, const char* name, uint32_t push_bytes)
        : test::ComputeKernel(gpu, CAELIX_TEST_SHADER_DIR, name, push_bytes)
    {
    }
};

// Push constants of tests/shaders/caelix_light_tree_probe.hlsl (the same member names).
struct ProbePush
{
    uint64_t descriptor;
    uint64_t input;
    uint64_t input2;
    uint64_t output;
    uint32_t mode;
    uint32_t count;
    uint32_t draws;
    uint32_t seed;
    uint32_t receivers;  // mode 3: receivers in input2
    uint32_t pad;
};
static_assert(sizeof(ProbePush) == 56);

constexpr uint32_t kModeCopy = 0;
constexpr uint32_t kModeSample = 1;
constexpr uint32_t kModePdf = 2;
constexpr uint32_t kModeBench = 3;
constexpr uint32_t kModeDraw = 4;

// One draw of mode 1 (32 words).
struct SampleRecord
{
    uint32_t ok, group, leaf, generation, voxel, face, block;
    float pdf;
    float position[3];
    float normal[3];
    uint32_t voxel_in_group;
    float point_pdf, brick_pdf;
    uint32_t valid;
    uint32_t ok2;
    float pdf_brick, brick_pdf2, inner_pdf2, point_pdf2;
    uint32_t group2, voxel_in_group2, face2;
    uint32_t pad[6];
};
static_assert(sizeof(SampleRecord) == 128);

struct Receiver
{
    float position[4];
    float normal[4];
};
static_assert(sizeof(Receiver) == 32);

class Probe
{
public:
    explicit Probe(test::Gpu& gpu)
        : gpu_(gpu), kernel_(gpu, "caelix_light_tree_probe", sizeof(ProbePush))
    {
        VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = 2;
        REQUIRE(vkCreateQueryPool(gpu.device.device, &query_info, nullptr, &queries_) == VK_SUCCESS);
    }
    ~Probe()
    {
        DestroyHostBuffer(gpu_, input_);
        DestroyHostBuffer(gpu_, input2_);
        DestroyHostBuffer(gpu_, output_);
        vkDestroyQueryPool(gpu_.device.device, queries_, nullptr);
    }

    // Runs one dispatch of `count` threads; inputs are copied into host buffers first. Returns the
    // GPU milliseconds of the dispatch.
    double Run(uint64_t descriptor, uint32_t mode, uint32_t count, uint32_t draws, uint32_t seed,
               const void* input, size_t input_bytes, const void* input2, size_t input2_bytes,
               size_t output_bytes, uint64_t input_address = 0)
    {
        REQUIRE(count <= 64u * 65535u);  // one dispatch
        Ensure(input_, input_bytes);
        Ensure(input2_, input2_bytes);
        Ensure(output_, output_bytes);
        if (input_bytes > 0)
        {
            std::memcpy(input_.data, input, input_bytes);
        }
        if (input2_bytes > 0)
        {
            std::memcpy(input2_.data, input2, input2_bytes);
        }
        const uint32_t receivers = static_cast<uint32_t>(input2_bytes / sizeof(Receiver));
        ProbePush push{descriptor, input_address != 0 ? input_address : input_.address, input2_.address,
                       output_.address, mode, count, draws, seed, receivers, 0};
        test::SubmitAndWait(gpu_, [&](VkCommandBuffer cmd) {
            vkCmdResetQueryPool(cmd, queries_, 0, 2);
            VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            before.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
            before.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
            before.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            before.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &before;
            vkCmdPipelineBarrier2(cmd, &dependency);
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries_, 0);
            kernel_.Record(cmd, &push, (count + 63) / 64);
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries_, 1);
            VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            after.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            after.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            after.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
            after.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
            dependency.pMemoryBarriers = &after;
            vkCmdPipelineBarrier2(cmd, &dependency);
        });
        uint64_t ticks[2] = {};
        REQUIRE(vkGetQueryPoolResults(gpu_.device.device, queries_, 0, 2, sizeof(ticks), ticks,
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
                VK_SUCCESS);
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(gpu_.device.physical_device.physical_device, &properties);
        return static_cast<double>(ticks[1] - ticks[0]) * properties.limits.timestampPeriod * 1e-6;
    }

    // `bytes` of GPU memory at a device address (16-byte aligned).
    std::vector<uint8_t> Read(uint64_t address, size_t bytes)
    {
        const uint32_t count = static_cast<uint32_t>((bytes + 15) / 16);
        std::vector<uint8_t> out(bytes);
        if (count == 0)
        {
            return out;
        }
        Run(0, kModeCopy, count, 1, 0, nullptr, 0, nullptr, 0, static_cast<size_t>(count) * 16, address);
        std::memcpy(out.data(), output_.data, bytes);
        return out;
    }

    const uint8_t* output() const
    {
        return output_.data;
    }

private:
    void Ensure(HostBuffer& buffer, size_t bytes)
    {
        if (buffer.size < bytes || buffer.buffer == VK_NULL_HANDLE)
        {
            DestroyHostBuffer(gpu_, buffer);
            buffer = CreateHostBuffer(gpu_, std::max<size_t>(bytes, 4096));
        }
    }

    test::Gpu& gpu_;
    ComputeKernel kernel_;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    HostBuffer input_;
    HostBuffer input2_;
    HostBuffer output_;
};

template <typename T>
T ReadStruct(Probe& probe, uint64_t address)
{
    const std::vector<uint8_t> bytes = probe.Read(address, sizeof(T));
    T out;
    std::memcpy(&out, bytes.data(), sizeof(T));
    return out;
}

// ---------------------------------------------------------------------------
// The CPU model of the scene and the reference light faces

struct Encoding
{
    uint32_t opaque_mask;
    uint32_t id_mask;
};

Encoding EncodingOf(uint32_t encoding)
{
    return encoding == caelix::kEncodingPacked ? Encoding{caelix::kPackedOpaqueMask, caelix::kPackedIdMask}
                                               : Encoding{caelix::kBlockIdOpaqueMask, caelix::kBlockIdIdMask};
}

struct EntityModel
{
    CaelixEntityId id = CAELIX_INVALID_ENTITY;
    World world;
    float position[3] = {};
    float rotation[4] = {0, 0, 0, 1};
    uint32_t motion = CAELIX_ENTITY_STATIC;
};

// Object-to-world rotation (rows) of a unit quaternion, as the library computes it.
std::array<std::array<float, 3>, 3> Rotation(const float q[4])
{
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
             {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
             {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

struct Model
{
    uint32_t encoding = 1;
    uint32_t group_shift = 4;
    float min_emission = 20.0f;
    std::vector<float> luminance;  // by block code (the material table's entry of the canonical code)
    std::vector<EntityModel> entities;
};

void ReadLuminance(CaelixScene* scene, Model& model)
{
    model.luminance.assign(65536, 0.0f);
    for (uint32_t code = 0; code < 65536; ++code)
    {
        CaelixMaterial m{};
        REQUIRE(caelix_scene_get_material(scene, code, &m) == CAELIX_OK);
        model.luminance[code] = caelix::LightLuminance(m.emission);
    }
}

uint16_t BlockAt(const World& world, Key key, int32_t x, int32_t y, int32_t z)
{
    int32_t c[3] = {x, y, z};
    for (int a = 0; a < 3; ++a)
    {
        if (c[a] < 0)
        {
            key[a] -= 1;
            c[a] += 8;
        }
        else if (c[a] >= 8)
        {
            key[a] += 1;
            c[a] -= 8;
        }
    }
    const auto found = world.bricks.find(key);
    return found == world.bricks.end() ? uint16_t{0} : found->second[c[0] + 8 * c[1] + 64 * c[2]];
}

struct RefFace
{
    size_t entity;  // index in Model::entities
    Key key;
    uint32_t voxel;  // x | y << 3 | z << 6
    uint32_t face;
    uint16_t block;
    float weight;
    uint32_t group = 0;
    uint32_t id = 0;
    uint32_t voxel_in_group = 0;  // packed
};

constexpr int32_t kDir[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

struct Reference
{
    std::vector<RefFace> faces;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, size_t> by_key;  // (group, voxel in group, face)
    std::map<uint32_t, std::array<int32_t, 3>> group_coord;
    std::map<uint32_t, size_t> group_entity;
    std::map<std::pair<uint32_t, uint32_t>, Key> brick_of_id;  // (group, id) -> key
};

Reference BuildReference(const test::Rig& rig, const Model& model)
{
    const Encoding enc = EncodingOf(model.encoding);
    const auto code_of = [&](uint32_t v) { return (v & enc.opaque_mask) != 0 ? v : (v & enc.id_mask); };
    const int32_t mask = (1 << model.group_shift) - 1;
    Reference ref;
    for (size_t e = 0; e < model.entities.size(); ++e)
    {
        const EntityModel& entity = model.entities[e];
        for (const auto& [key, brick] : entity.world.bricks)
        {
            CaelixBrickInfo info{};
            const CaelixBrickKey brick_key{key[0], key[1], key[2]};
            REQUIRE(caelix_scene_query_brick(rig.scene, entity.id, brick_key, &info) == CAELIX_OK);
            REQUIRE(info.resident == 1);
            ref.group_coord[info.group_index] = {key[0] >> model.group_shift, key[1] >> model.group_shift,
                                                 key[2] >> model.group_shift};
            ref.group_entity[info.group_index] = e;
            ref.brick_of_id[{info.group_index, info.group_local_id}] = key;
            for (uint32_t vi = 0; vi < 512; ++vi)
            {
                const int32_t x = vi & 7, y = (vi >> 3) & 7, z = vi >> 6;
                const uint16_t v = brick[vi];
                const float lum = model.luminance[code_of(v)];
                if (!(lum >= model.min_emission))
                {
                    continue;
                }
                for (uint32_t f = 0; f < 6; ++f)
                {
                    const uint16_t n =
                        BlockAt(entity.world, key, x + kDir[f][0], y + kDir[f][1], z + kDir[f][2]);
                    if ((n & enc.opaque_mask) != 0)
                    {
                        continue;
                    }
                    RefFace face{e, key, vi, f, v, lum};
                    face.group = info.group_index;
                    face.id = info.group_local_id;
                    face.voxel_in_group = PackVoxel(static_cast<uint32_t>((key[0] & mask) * 8 + x),
                                                    static_cast<uint32_t>((key[1] & mask) * 8 + y),
                                                    static_cast<uint32_t>((key[2] & mask) * 8 + z));
                    ref.by_key[{face.group, face.voxel_in_group, f}] = ref.faces.size();
                    ref.faces.push_back(face);
                }
            }
        }
    }
    return ref;
}

// ---------------------------------------------------------------------------
// Structure checks: rows, blocks (index entries, records, face entries), BVH leaves and nodes

struct GpuState
{
    LightTreeDescriptor descriptor{};
    LightControl control{};
    std::vector<GroupLightRow> rows;
    std::vector<GroupInfo> group_infos;
    std::map<uint32_t, std::vector<uint8_t>> blocks;  // by group
    std::array<std::vector<LightBvhLeaf>, 2> leaves;
    std::array<std::vector<LightBvhNode>, 2> nodes;
};

GpuState ReadGpu(const test::Rig& rig, Probe& probe)
{
    GpuState s;
    const uint64_t address = caelix_scene_get_light_tree(rig.scene);
    REQUIRE(address != 0);
    s.descriptor = ReadStruct<LightTreeDescriptor>(probe, address);
    s.control = ReadStruct<LightControl>(probe, s.descriptor.control);
    const std::vector<uint8_t> rows =
        probe.Read(s.descriptor.group_rows, s.descriptor.group_count * sizeof(GroupLightRow));
    s.rows.resize(s.descriptor.group_count);
    std::memcpy(s.rows.data(), rows.data(), rows.size());
    const std::vector<uint8_t> infos =
        probe.Read(s.descriptor.group_infos, s.descriptor.group_count * sizeof(GroupInfo));
    s.group_infos.resize(s.descriptor.group_count);
    std::memcpy(s.group_infos.data(), infos.data(), infos.size());
    const uint64_t heap = s.descriptor.heap[s.control.current & 1];
    for (uint32_t g = 0; g < s.rows.size(); ++g)
    {
        if (s.rows[g].block_units != 0)
        {
            s.blocks[g] = probe.Read(heap + uint64_t{s.rows[g].block_offset} * kLtUnitBytes,
                                     uint64_t{s.rows[g].block_units} * kLtUnitBytes);
        }
    }
    for (int t = 0; t < 2; ++t)
    {
        const LightTreeHeader& header = s.control.headers[t];
        s.leaves[t].resize(header.leaf_count);
        s.nodes[t].resize(header.node_count);
        const std::vector<uint8_t> leaves =
            probe.Read(s.descriptor.leaves[t], header.leaf_count * sizeof(LightBvhLeaf));
        const std::vector<uint8_t> nodes =
            probe.Read(s.descriptor.nodes[t], header.node_count * sizeof(LightBvhNode));
        std::memcpy(s.leaves[t].data(), leaves.data(), leaves.size());
        std::memcpy(s.nodes[t].data(), nodes.data(), nodes.size());
    }
    return s;
}

bool Near(double a, double b, double rel, double abs_tol = 1e-6)
{
    return std::abs(a - b) <= abs_tol + rel * std::max(std::abs(a), std::abs(b));
}

// A BVH: every leaf reached exactly once from the root, parent links, slot summaries (a leaf slot is
// the leaf; a node slot holds the sums of the child node's slots and a box that contains them),
// the header summary.
void CheckBvh(const GpuState& s, int tree)
{
    INFO("tree " << tree);
    const LightTreeHeader& header = s.control.headers[tree];
    const std::vector<LightBvhLeaf>& leaves = s.leaves[tree];
    const std::vector<LightBvhNode>& nodes = s.nodes[tree];
    REQUIRE(s.control.states[tree].leaves_dropped == 0);
    REQUIRE(s.control.states[tree].node_overflow == 0);
    if (header.leaf_count == 0)
    {
        REQUIRE(header.root == kLtNone);
        return;
    }
    REQUIRE(header.root == 0);
    REQUIRE(nodes.size() >= 1);
    REQUIRE(nodes[0].parent == kLtNone);
    std::vector<int> leaf_seen(leaves.size(), 0);
    std::vector<int> node_seen(nodes.size(), 0);
    // Sum of the slot summaries of a node.
    const auto node_sum = [&](uint32_t n, double face[6], float lo[3], float hi[3]) {
        for (int f = 0; f < 6; ++f)
        {
            face[f] = 0.0;
        }
        for (int a = 0; a < 3; ++a)
        {
            lo[a] = INFINITY;
            hi[a] = -INFINITY;
        }
        for (int slot = 0; slot < 4; ++slot)
        {
            if (nodes[n].child[slot] == kLtNone)
            {
                continue;
            }
            double power = 0.0;
            for (int f = 0; f < 6; ++f)
            {
                face[f] += nodes[n].child_face[slot][f];
                power += nodes[n].child_face[slot][f];
            }
            if (power > 0.0)
            {
                for (int a = 0; a < 3; ++a)
                {
                    lo[a] = std::min(lo[a], nodes[n].child_lo[slot][a]);
                    hi[a] = std::max(hi[a], nodes[n].child_hi[slot][a]);
                }
            }
        }
    };
    std::vector<uint32_t> stack = {0};
    while (!stack.empty())
    {
        const uint32_t n = stack.back();
        stack.pop_back();
        REQUIRE(n < nodes.size());
        REQUIRE(node_seen[n] == 0);
        node_seen[n] = 1;
        const LightBvhNode& node = nodes[n];
        uint32_t used = 0;
        for (uint32_t slot = 0; slot < 4; ++slot)
        {
            const uint32_t child = node.child[slot];
            if (child == kLtNone)
            {
                continue;
            }
            ++used;
            if ((child & kLtChildLeafBit) != 0)
            {
                const uint32_t leaf = child & ~kLtChildLeafBit;
                REQUIRE(leaf < leaves.size());
                REQUIRE(leaf_seen[leaf] == 0);
                leaf_seen[leaf] = 1;
                REQUIRE(leaves[leaf].parent == ((n << caelix::kLtParentSlotBits) | slot));
                const caelix::LightSummary& summary = leaves[leaf].summary;
                REQUIRE(std::memcmp(node.child_face[slot], summary.face, sizeof(summary.face)) == 0);
                REQUIRE(std::memcmp(node.child_lo[slot], summary.lo, sizeof(summary.lo)) == 0);
                REQUIRE(std::memcmp(node.child_hi[slot], summary.hi, sizeof(summary.hi)) == 0);
            }
            else
            {
                REQUIRE(child < nodes.size());
                REQUIRE(nodes[child].parent == n);
                REQUIRE(nodes[child].parent_slot == slot);
                double face[6];
                float lo[3];
                float hi[3];
                node_sum(child, face, lo, hi);
                double power = 0.0;
                for (int f = 0; f < 6; ++f)
                {
                    REQUIRE(Near(node.child_face[slot][f], face[f], 1e-5, 1e-4));
                    power += face[f];
                }
                if (power > 0.0)
                {
                    for (int a = 0; a < 3; ++a)
                    {
                        REQUIRE(node.child_lo[slot][a] <= lo[a]);
                        REQUIRE(node.child_hi[slot][a] >= hi[a]);
                    }
                }
                stack.push_back(child);
            }
        }
        REQUIRE(used >= (header.leaf_count == 1 ? 1u : 2u));
    }
    for (size_t l = 0; l < leaves.size(); ++l)
    {
        INFO("leaf " << l);
        REQUIRE(leaf_seen[l] == 1);
    }
    for (size_t n = 0; n < nodes.size(); ++n)
    {
        REQUIRE(node_seen[n] == 1);
    }
    double face[6];
    float lo[3];
    float hi[3];
    node_sum(0, face, lo, hi);
    for (int f = 0; f < 6; ++f)
    {
        REQUIRE(Near(header.summary.face[f], face[f], 1e-5, 1e-4));
    }
    for (int a = 0; a < 3; ++a)
    {
        REQUIRE(header.summary.lo[a] <= lo[a]);
        REQUIRE(header.summary.hi[a] >= hi[a]);
    }
}

void CheckStructure(const test::Rig& rig, const Model& model, const Reference& ref, const GpuState& s)
{
    const Encoding enc = EncodingOf(model.encoding);
    const auto raw = [&](uint32_t v) { return (v & enc.opaque_mask) != 0 ? v : (v & enc.id_mask); };
    struct BrickRef
    {
        std::array<std::vector<std::pair<uint32_t, uint32_t>>, 6> entries;  // (voxel, raw block)
        std::array<double, 6> sums{};
        uint32_t lo[3] = {0xFFFF, 0xFFFF, 0xFFFF};
        uint32_t hi[3] = {0, 0, 0};
        uint32_t faces = 0;
        size_t entity = 0;
    };
    std::map<uint32_t, std::map<uint32_t, BrickRef>> by_group;  // group -> id -> brick
    for (const RefFace& face : ref.faces)
    {
        BrickRef& brick = by_group[face.group][face.id];
        brick.entries[face.face].emplace_back(face.voxel, raw(face.block));
        brick.sums[face.face] += face.weight;
        brick.entity = face.entity;
        const uint32_t v[3] = {UnpackVoxel(face.voxel_in_group, 0), UnpackVoxel(face.voxel_in_group, 1),
                               UnpackVoxel(face.voxel_in_group, 2)};
        for (int a = 0; a < 3; ++a)
        {
            brick.lo[a] = std::min(brick.lo[a], v[a]);
            brick.hi[a] = std::max(brick.hi[a], v[a] + 1);
        }
        ++brick.faces;
    }
    uint64_t total_bricks = 0;
    uint64_t total_faces = 0;
    std::array<uint32_t, 2> tree_bricks{};
    const uint32_t mask = (1u << model.group_shift) - 1;
    const auto morton_of_pos = [&](uint32_t pos) {
        return Part1By2(pos & mask) | (Part1By2((pos >> model.group_shift) & mask) << 1) |
               (Part1By2((pos >> (2 * model.group_shift)) & mask) << 2);
    };
    for (uint32_t g = 0; g < s.rows.size(); ++g)
    {
        INFO("group " << g);
        const GroupLightRow& row = s.rows[g];
        const auto found = by_group.find(g);
        if (found == by_group.end())
        {
            REQUIRE((row.bricks & caelix::kLtRowBricksMask) == 0);
            REQUIRE(row.light_faces == 0);
            REQUIRE(row.block_units == 0);
            continue;
        }
        const std::map<uint32_t, BrickRef>& bricks = found->second;
        uint32_t faces = 0;
        for (const auto& [id, brick] : bricks)
        {
            faces += brick.faces;
        }
        total_bricks += bricks.size();
        total_faces += faces;
        REQUIRE((row.bricks & caelix::kLtRowBricksMask) == bricks.size());
        REQUIRE(row.light_faces == faces);
        REQUIRE(row.block_units != 0);
        REQUIRE((row.cpu_flags | row.gpu_flags) == 0);
        const std::vector<uint8_t>& block = s.blocks.at(g);
        caelix::LightBlockHeader header;
        std::memcpy(&header, block.data(), sizeof(header));
        REQUIRE(header.record_count == bricks.size());
        const uint64_t records_offset =
            caelix::kLtBlockIndexOffset + uint64_t{header.record_count} * sizeof(LightIndexEntry);
        uint32_t previous_morton = 0;
        uint64_t record_bytes = 0;
        for (uint32_t r = 0; r < header.record_count; ++r)
        {
            LightIndexEntry entry;
            const uint64_t entry_at = caelix::kLtBlockIndexOffset + uint64_t{r} * sizeof(entry);
            std::memcpy(&entry, block.data() + entry_at, sizeof(entry));
            INFO("index entry " << r << " id " << entry.dense_id);
            REQUIRE(bricks.count(entry.dense_id) == 1);
            const BrickRef& brick = bricks.at(entry.dense_id);
            const Key key = ref.brick_of_id.at({g, entry.dense_id});
            REQUIRE(entry.pos == ((key[0] & mask) | ((key[1] & mask) << model.group_shift) |
                                  ((key[2] & mask) << (2 * model.group_shift))));
            const uint32_t morton = morton_of_pos(entry.pos);
            REQUIRE((r == 0 || morton > previous_morton));
            previous_morton = morton;
            BrickLightRecordHeader record;
            const uint64_t record_at = uint64_t{entry.record_offset} * kLtUnitBytes;
            REQUIRE(record_at >= records_offset);
            REQUIRE(record_at + sizeof(record) <= block.size());
            std::memcpy(&record, block.data() + record_at, sizeof(record));
            REQUIRE(record.pos == entry.pos);
            REQUIRE(record.box_lo == PackVoxel(brick.lo[0], brick.lo[1], brick.lo[2]));
            REQUIRE(record.box_hi == PackVoxel(brick.hi[0], brick.hi[1], brick.hi[2]));
            REQUIRE(record.entry_count == brick.faces);
            record_bytes += RecordBytes(record.entry_count);
            const uint32_t* entries =
                reinterpret_cast<const uint32_t*>(block.data() + record_at + sizeof(record));
            uint32_t first = 0;
            for (uint32_t f = 0; f < 6; ++f)
            {
                const uint32_t count = FamilyCount(record.counts_lo, record.counts_hi, f);
                std::vector<std::pair<uint32_t, uint32_t>> expected = brick.entries[f];
                std::sort(expected.begin(), expected.end());
                REQUIRE(count == expected.size());
                for (uint32_t k = 0; k < count; ++k)
                {
                    const uint32_t value = entries[first + k];
                    REQUIRE((value & caelix::kLtFaceEntryVoxelMask) == expected[k].first);
                    REQUIRE(raw(value >> caelix::kLtFaceEntryBlockShift) == expected[k].second);
                }
                REQUIRE(Near(record.face[f], brick.sums[f], 1e-5));
                first += count;
            }
            // The brick's BVH leaf: in the tree of its entity's motion, pointing back at the entry.
            const uint32_t tree = model.entities[brick.entity].motion;
            REQUIRE(entry.bvh_leaf != kLtNone);
            REQUIRE((entry.bvh_leaf >> caelix::kLtBvhLeafTreeShift) == tree);
            const uint32_t leaf_index = entry.bvh_leaf & caelix::kLtBvhLeafIndexMask;
            REQUIRE(leaf_index < s.leaves[tree].size());
            const LightBvhLeaf& leaf = s.leaves[tree][leaf_index];
            REQUIRE(leaf.group == g);
            REQUIRE(leaf.entry == r);
            REQUIRE(s.group_infos[g].entity == model.entities[brick.entity].id);
            double local_power = 0.0;
            double world_power = 0.0;
            for (int f = 0; f < 6; ++f)
            {
                local_power += record.face[f];
                world_power += leaf.summary.face[f];
            }
            REQUIRE(world_power >= local_power * (1.0 - 1e-5));
            ++tree_bricks[tree];
        }
        REQUIRE(records_offset + record_bytes == block.size());
    }
    for (int t = 0; t < 2; ++t)
    {
        REQUIRE(s.control.headers[t].leaf_count == tree_bricks[t]);
        CheckBvh(s, t);
    }
    CaelixLightStats stats{};
    caelix_scene_get_light_stats(rig.scene, &stats);
    REQUIRE(stats.enabled == 1);
    REQUIRE(stats.light_bricks == total_bricks);
    REQUIRE(stats.light_faces == total_faces);
    REQUIRE(stats.light_groups == by_group.size());
    REQUIRE(stats.bricks_dropped == 0);
    for (int t = 0; t < 2; ++t)
    {
        REQUIRE(stats.tree_leaves[t] == tree_bricks[t]);
        REQUIRE(stats.tree_nodes[t] == s.control.headers[t].node_count);
        REQUIRE(stats.tree_leaves_dropped[t] == 0);
    }
    REQUIRE(s.control.light_bricks == total_bricks);
}

// ---------------------------------------------------------------------------
// Sampling checks

std::vector<Receiver> MakeReceivers(const Model& model, const Reference& ref, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::vector<Receiver> receivers;
    // A receiver inside a light group (at the voxel next to a light face, in the face's direction),
    // without a normal and with the face's normal flipped (looking back at it).
    if (!ref.faces.empty())
    {
        const RefFace& face = ref.faces[ref.faces.size() / 2];
        const EntityModel& entity = model.entities[face.entity];
        const auto r = Rotation(entity.rotation);
        float local[3];
        for (int a = 0; a < 3; ++a)
        {
            const int32_t v = face.key[a] * 8 + static_cast<int32_t>((face.voxel >> (3 * a)) & 7);
            local[a] = static_cast<float>(v) + 0.5f + 1.0f * static_cast<float>(kDir[face.face][a]);
        }
        Receiver inside{};
        for (int i = 0; i < 3; ++i)
        {
            inside.position[i] =
                r[i][0] * local[0] + r[i][1] * local[1] + r[i][2] * local[2] + entity.position[i];
        }
        receivers.push_back(inside);
        for (int i = 0; i < 3; ++i)
        {
            inside.normal[i] =
                -(r[i][0] * kDir[face.face][0] + r[i][1] * kDir[face.face][1] + r[i][2] * kDir[face.face][2]);
        }
        receivers.push_back(inside);
    }
    // Random receivers around the entities' bricks, half without a normal.
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    for (int k = 0; k < 6; ++k)
    {
        const EntityModel& entity = model.entities[static_cast<size_t>(k) % model.entities.size()];
        const long pick = static_cast<long>(rng() % entity.world.bricks.size());
        const Key key = std::next(entity.world.bricks.begin(), pick)->first;
        const auto r = Rotation(entity.rotation);
        float local[3];
        for (int a = 0; a < 3; ++a)
        {
            local[a] = static_cast<float>(key[a] * 8) + 4.0f + unit(rng) * (k < 3 ? 12.0f : 120.0f);
        }
        Receiver receiver{};
        for (int i = 0; i < 3; ++i)
        {
            receiver.position[i] =
                r[i][0] * local[0] + r[i][1] * local[1] + r[i][2] * local[2] + entity.position[i];
        }
        if (k % 2 == 1)
        {
            float n[3] = {unit(rng), unit(rng), unit(rng)};
            const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            for (int i = 0; i < 3; ++i)
            {
                receiver.normal[i] = n[i] / len;
            }
        }
        receivers.push_back(receiver);
    }
    return receivers;
}

// LightTreePointPdf of every reference face for every receiver: [receiver][face].
std::vector<std::vector<float>> EvaluatePdfs(Probe& probe, uint64_t descriptor, const Reference& ref,
                                             const std::vector<Receiver>& receivers)
{
    std::vector<uint32_t> queries;
    queries.reserve(receivers.size() * ref.faces.size() * 4);
    for (uint32_t r = 0; r < receivers.size(); ++r)
    {
        for (const RefFace& face : ref.faces)
        {
            queries.insert(queries.end(),
                           {r, face.group, face.voxel_in_group, face.face | (uint32_t{face.block} << 16)});
        }
    }
    const uint32_t count = static_cast<uint32_t>(queries.size() / 4);
    probe.Run(descriptor, kModePdf, count, 1, 0, queries.data(), queries.size() * 4, receivers.data(),
              receivers.size() * sizeof(Receiver), static_cast<size_t>(count) * 4);
    std::vector<std::vector<float>> out(receivers.size(), std::vector<float>(ref.faces.size()));
    const float* pdfs = reinterpret_cast<const float*>(probe.output());
    for (size_t r = 0; r < receivers.size(); ++r)
    {
        std::memcpy(out[r].data(), pdfs + r * ref.faces.size(), ref.faces.size() * sizeof(float));
    }
    return out;
}

void CheckNormalisation(const std::vector<std::vector<float>>& pdfs)
{
    for (size_t r = 0; r < pdfs.size(); ++r)
    {
        double sum = 0.0;
        for (const float p : pdfs[r])
        {
            REQUIRE(p >= 0.0f);
            sum += p;
        }
        INFO("receiver " << r << ": sum of the face pdfs " << sum);
        REQUIRE(std::abs(sum - 1.0) <= 1e-4);
    }
}

void CheckDraws(Probe& probe, uint64_t descriptor, const Model& model, const Reference& ref,
                const std::vector<Receiver>& receivers, uint32_t draws)
{
    const uint32_t count = static_cast<uint32_t>(receivers.size()) * draws;
    probe.Run(descriptor, kModeSample, count, draws, 12345, nullptr, 0, receivers.data(),
              receivers.size() * sizeof(Receiver), static_cast<size_t>(count) * sizeof(SampleRecord));
    std::vector<SampleRecord> records(count);
    std::memcpy(records.data(), probe.output(), records.size() * sizeof(SampleRecord));
    const Encoding enc = EncodingOf(model.encoding);
    const auto raw = [&](uint32_t v) { return (v & enc.opaque_mask) != 0 ? v : (v & enc.id_mask); };
    for (uint32_t i = 0; i < count; ++i)
    {
        const SampleRecord& s = records[i];
        INFO("draw " << i << " receiver " << i / draws);
        REQUIRE(s.ok == 1);
        REQUIRE(s.valid == 1);
        REQUIRE(s.pdf > 0.0f);
        REQUIRE(Near(s.point_pdf, s.pdf, 1e-4, 0.0));
        REQUIRE(s.brick_pdf > 0.0f);
        const auto found = ref.by_key.find({s.group, s.voxel_in_group, s.face});
        REQUIRE(found != ref.by_key.end());
        const RefFace& face = ref.faces[found->second];
        REQUIRE(raw(s.block) == raw(face.block));
        REQUIRE(s.voxel == face.voxel);
        // The point lies on the face (entity-local frame).
        const EntityModel& entity = model.entities[face.entity];
        const auto r = Rotation(entity.rotation);
        float d[3];
        for (int a = 0; a < 3; ++a)
        {
            d[a] = s.position[a] - entity.position[a];
        }
        for (int a = 0; a < 3; ++a)
        {
            const float local = r[0][a] * d[0] + r[1][a] * d[1] + r[2][a] * d[2];
            const float v =
                static_cast<float>(face.key[a] * 8 + static_cast<int32_t>((face.voxel >> (3 * a)) & 7));
            const float tolerance = 2e-3f + 2e-5f * std::abs(v);
            if (static_cast<uint32_t>(a) == face.face / 2)
            {
                REQUIRE(std::abs(local - (v + ((face.face & 1) != 0 ? 0.0f : 1.0f))) <= tolerance);
            }
            else
            {
                REQUIRE(local >= v - tolerance);
                REQUIRE(local <= v + 1.0f + tolerance);
            }
            const float normal =
                r[a][0] * kDir[face.face][0] + r[a][1] * kDir[face.face][1] + r[a][2] * kDir[face.face][2];
            REQUIRE(std::abs(s.normal[a] - normal) <= 1e-4f);
        }
        // The two halves.
        REQUIRE(s.ok2 == 1);
        REQUIRE(Near(s.pdf_brick, s.brick_pdf2, 1e-4, 0.0));
        REQUIRE(Near(static_cast<double>(s.pdf_brick) * s.inner_pdf2, s.point_pdf2, 1e-4, 0.0));
        REQUIRE(ref.by_key.count({s.group2, s.voxel_in_group2, s.face2}) == 1);
    }
}

void CheckHistogram(Probe& probe, uint64_t descriptor, const Reference& ref, const Receiver& receiver,
                    const std::vector<float>& pdfs)
{
    constexpr uint32_t kDraws = 1u << 21;
    probe.Run(descriptor, kModeDraw, kDraws, kDraws, 777, nullptr, 0, &receiver, sizeof(receiver),
              static_cast<size_t>(kDraws) * 16);
    std::vector<uint32_t> counts(ref.faces.size(), 0);
    const uint32_t* out = reinterpret_cast<const uint32_t*>(probe.output());
    for (uint32_t i = 0; i < kDraws; ++i)
    {
        REQUIRE(out[4 * i + 3] == 1);
        const auto found = ref.by_key.find({out[4 * i], out[4 * i + 1], out[4 * i + 2]});
        REQUIRE(found != ref.by_key.end());
        ++counts[found->second];
    }
    uint32_t checked = 0;
    for (size_t f = 0; f < ref.faces.size(); ++f)
    {
        const double expected = static_cast<double>(pdfs[f]) * kDraws;
        if (pdfs[f] == 0.0f)
        {
            REQUIRE(counts[f] == 0);
        }
        if (expected >= 50.0)
        {
            INFO("face " << f << ": " << counts[f] << " draws, expected " << expected);
            REQUIRE(std::abs(counts[f] - expected) <= 5.0 * std::sqrt(expected));
            ++checked;
        }
    }
    REQUIRE(checked > 0);
}

void CheckBound(Probe& probe, uint64_t descriptor, const Model& model, const Reference& ref, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<Receiver> receivers;
    std::vector<uint32_t> queries;
    for (int k = 0; k < 4096; ++k)
    {
        const size_t index = rng() % ref.faces.size();
        const RefFace& face = ref.faces[index];
        const EntityModel& entity = model.entities[face.entity];
        const auto r = Rotation(entity.rotation);
        float centre_local[3];
        float normal_local[3];
        for (int a = 0; a < 3; ++a)
        {
            const float v =
                static_cast<float>(face.key[a] * 8 + static_cast<int32_t>((face.voxel >> (3 * a)) & 7));
            centre_local[a] = v + 0.5f + 0.5f * static_cast<float>(kDir[face.face][a]);
            normal_local[a] = static_cast<float>(kDir[face.face][a]);
        }
        float centre[3];
        float normal[3];
        for (int i = 0; i < 3; ++i)
        {
            centre[i] = r[i][0] * centre_local[0] + r[i][1] * centre_local[1] + r[i][2] * centre_local[2] +
                        entity.position[i];
            normal[i] = r[i][0] * normal_local[0] + r[i][1] * normal_local[1] + r[i][2] * normal_local[2];
        }
        Receiver receiver{};
        const float range = (k % 3 == 0) ? 4.0f : 60.0f;
        for (int i = 0; i < 3; ++i)
        {
            receiver.position[i] = centre[i] + unit(rng) * range;
        }
        float n[3] = {unit(rng), unit(rng), unit(rng)};
        const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        const bool has_normal = k % 2 == 0;
        for (int i = 0; i < 3; ++i)
        {
            receiver.normal[i] = has_normal ? n[i] / len : 0.0f;
        }
        // A clearly positive contribution at the face centre: both cosines above 0.05.
        float d[3];
        float dist2 = 0.0f;
        for (int i = 0; i < 3; ++i)
        {
            d[i] = receiver.position[i] - centre[i];
            dist2 += d[i] * d[i];
        }
        const float dist = std::sqrt(dist2);
        const float emitter = (normal[0] * d[0] + normal[1] * d[1] + normal[2] * d[2]) / dist;
        const float facing =
            -(receiver.normal[0] * d[0] + receiver.normal[1] * d[1] + receiver.normal[2] * d[2]);
        const float receiving = has_normal ? facing / dist : 1.0f;
        if (dist < 0.5f || emitter < 0.05f || receiving < 0.05f)
        {
            continue;
        }
        queries.insert(queries.end(), {static_cast<uint32_t>(receivers.size()), face.group,
                                       face.voxel_in_group, face.face | (uint32_t{face.block} << 16)});
        receivers.push_back(receiver);
    }
    const uint32_t count = static_cast<uint32_t>(receivers.size());
    REQUIRE(count > 100);
    probe.Run(descriptor, kModePdf, count, 1, 0, queries.data(), queries.size() * 4, receivers.data(),
              receivers.size() * sizeof(Receiver), static_cast<size_t>(count) * 4);
    const float* pdfs = reinterpret_cast<const float*>(probe.output());
    for (uint32_t i = 0; i < count; ++i)
    {
        INFO("pair " << i);
        REQUIRE(pdfs[i] > 0.0f);
    }
}

// Every check of one scene state.
void CheckAll(const test::Rig& rig, Probe& probe, const Model& model, uint32_t seed, bool histogram = true)
{
    const Reference ref = BuildReference(rig, model);
    REQUIRE(!ref.faces.empty());
    const GpuState gpu = ReadGpu(rig, probe);
    CheckStructure(rig, model, ref, gpu);
    CaelixLightStats stats{};
    caelix_scene_get_light_stats(rig.scene, &stats);
    std::printf("light tree check %u: %zu light faces, %llu light bricks in %u groups, "
                "heap %llu / %llu bytes, passes %u (failed %u), last pass: %u records recomputed, "
                "%u copied, %.3f ms\n",
                seed, ref.faces.size(), static_cast<unsigned long long>(stats.light_bricks),
                stats.light_groups, static_cast<unsigned long long>(stats.heap_bytes_used),
                static_cast<unsigned long long>(stats.heap_bytes_capacity), stats.passes,
                stats.failed_passes, stats.records_recomputed, stats.records_copied,
                static_cast<double>(stats.gpu_ms));
    const uint64_t descriptor = caelix_scene_get_light_tree(rig.scene);
    const std::vector<Receiver> receivers = MakeReceivers(model, ref, seed);
    const std::vector<std::vector<float>> pdfs = EvaluatePdfs(probe, descriptor, ref, receivers);
    CheckNormalisation(pdfs);
    CheckDraws(probe, descriptor, model, ref, receivers, 2048);
    if (histogram)
    {
        CheckHistogram(probe, descriptor, ref, receivers.back(), pdfs.back());
    }
    CheckBound(probe, descriptor, model, ref, seed + 1);
}

// ---------------------------------------------------------------------------
// Scenes

// Block-ID codes of the test materials.
constexpr uint16_t kStone = 0x8000 | 1;      // opaque, no emission
constexpr uint16_t kLampBright = 0x8000 | 5;  // opaque, luminance 60
constexpr uint16_t kLampDim = 0x8000 | 6;     // opaque, luminance 15 (below the default threshold)
constexpr uint16_t kLampWarm = 0x8000 | 7;    // opaque, luminance about 23
constexpr uint16_t kGlassLamp = 9;            // transparent id 9, luminance about 35

void SetTestMaterials(CaelixScene* scene)
{
    const auto material = [](float r, float g, float b, float ior) {
        CaelixMaterial m{};
        m.albedo[0] = m.albedo[1] = m.albedo[2] = 0.7f;
        m.emission[0] = r;
        m.emission[1] = g;
        m.emission[2] = b;
        m.ior = ior;
        return m;
    };
    const CaelixMaterialEntry entries[] = {
        {kStone, material(0, 0, 0, 1.5f)},
        {kLampBright, material(60, 60, 60, 1.5f)},
        {kLampDim, material(15, 15, 15, 1.5f)},
        {kLampWarm, material(25, 22, 30, 1.5f)},
        {kGlassLamp, material(40, 35, 30, 1.3f)},
    };
    REQUIRE(caelix_scene_set_material_entries(scene, entries, 5) == CAELIX_OK);
}

// A brick with air, stone and the four lamp kinds (light voxels next to air, stone and other bricks).
Brick LampBrick(std::mt19937& rng, uint32_t lamp_percent)
{
    Brick brick{};
    for (uint16_t& v : brick)
    {
        const uint32_t r = rng() % 100;
        if (r < 50)
        {
            v = 0;
        }
        else if (r < 100 - lamp_percent)
        {
            v = kStone;
        }
        else
        {
            const uint16_t lamps[4] = {kLampBright, kLampDim, kLampWarm, kGlassLamp};
            v = lamps[rng() % 4];
        }
    }
    return brick;
}

// Bricks around the group borders x 15|16, y -1|0, z 31|32 (8 groups at group shift 4), and a
// cluster of 4 x 4 x 4 bricks inside group (1, 0, 2) away from its borders.
World BorderWorld(std::mt19937& rng)
{
    World world;
    for (int32_t x = 14; x <= 17; ++x)
    {
        for (int32_t y = -2; y <= 1; ++y)
        {
            for (int32_t z = 30; z <= 33; ++z)
            {
                if (rng() % 3 != 0)
                {
                    world.bricks[{x, y, z}] = LampBrick(rng, 8);
                }
            }
        }
    }
    for (int32_t x = 20; x <= 23; ++x)
    {
        for (int32_t y = 4; y <= 7; ++y)
        {
            for (int32_t z = 36; z <= 39; ++z)
            {
                world.bricks[{x, y, z}] = LampBrick(rng, 3);
            }
        }
    }
    return world;
}

World SmallWorld(std::mt19937& rng)
{
    World world;
    for (int32_t x = -1; x <= 2; ++x)
    {
        for (int32_t y = 0; y <= 1; ++y)
        {
            for (int32_t z = 0; z <= 1; ++z)
            {
                world.bricks[{x, y, z}] = LampBrick(rng, 6);
            }
        }
    }
    return world;
}

void UploadWorld(test::Rig& rig, CaelixEntityId entity, const World& world, const std::vector<Key>& keys)
{
    const CaelixEntityId saved = rig.entity;
    rig.entity = entity;
    test::Upload(rig, world, keys);
    rig.entity = saved;
}

void RemoveKeys(test::Rig& rig, CaelixEntityId entity, const std::vector<Key>& keys)
{
    const CaelixEntityId saved = rig.entity;
    rig.entity = entity;
    test::Remove(rig, keys);
    rig.entity = saved;
}

void SetLightOptions(CaelixScene* scene, float min_emission, uint64_t heap_bytes = 0)
{
    CaelixLightOptions options{};
    options.enabled = 1;
    options.min_emission = min_emission;
    options.heap_bytes = heap_bytes;
    REQUIRE(caelix_scene_set_light_options(scene, &options) == CAELIX_OK);
}

// Rotation of `degrees` about a unit axis.
void AxisAngle(const float axis[3], float degrees, float q[4])
{
    const float half = degrees * 3.14159265f / 360.0f;
    const float len = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    for (int i = 0; i < 3; ++i)
    {
        q[i] = axis[i] / len * std::sin(half);
    }
    q[3] = std::cos(half);
}

std::vector<uint32_t> Generations(Probe& probe, const test::Rig& rig)
{
    const GpuState s = ReadGpu(rig, probe);
    std::vector<uint32_t> out;
    for (const GroupLightRow& row : s.rows)
    {
        out.push_back(row.generation);
    }
    return out;
}

uint32_t GroupOf(const test::Rig& rig, CaelixEntityId entity, const Key& key)
{
    CaelixBrickInfo info{};
    const CaelixBrickKey brick_key{key[0], key[1], key[2]};
    REQUIRE(caelix_scene_query_brick(rig.scene, entity, brick_key, &info) == CAELIX_OK);
    REQUIRE(info.resident == 1);
    return info.group_index;
}

void RequireChangedGenerations(const std::vector<uint32_t>& before, const std::vector<uint32_t>& after,
                               const std::set<uint32_t>& allowed, const std::set<uint32_t>& required)
{
    for (size_t g = 0; g < after.size(); ++g)
    {
        const uint32_t old = g < before.size() ? before[g] : 0u;
        INFO("group " << g);
        if (allowed.count(static_cast<uint32_t>(g)) == 0)
        {
            REQUIRE(after[g] == old);
        }
        if (required.count(static_cast<uint32_t>(g)) != 0)
        {
            REQUIRE(after[g] != old);
        }
    }
}

// The bytes of every brick light record of a group, by brick key.
std::map<Key, std::vector<uint8_t>> RecordsByKey(const test::Rig& rig, Probe& probe, uint32_t group,
                                                 const Reference& ref)
{
    const GpuState s = ReadGpu(rig, probe);
    std::map<Key, std::vector<uint8_t>> out;
    const std::vector<uint8_t>& block = s.blocks.at(group);
    caelix::LightBlockHeader header;
    std::memcpy(&header, block.data(), sizeof(header));
    for (uint32_t r = 0; r < header.record_count; ++r)
    {
        LightIndexEntry entry;
        const uint64_t entry_at = caelix::kLtBlockIndexOffset + uint64_t{r} * sizeof(entry);
        std::memcpy(&entry, block.data() + entry_at, sizeof(entry));
        BrickLightRecordHeader record;
        std::memcpy(&record, block.data() + uint64_t{entry.record_offset} * kLtUnitBytes, sizeof(record));
        const size_t bytes = static_cast<size_t>(RecordBytes(record.entry_count));
        const uint8_t* first = block.data() + uint64_t{entry.record_offset} * kLtUnitBytes;
        out[ref.brick_of_id.at({group, entry.dense_id})] = std::vector<uint8_t>(first, first + bytes);
    }
    return out;
}

CaelixLightStats LightStats(const test::Rig& rig)
{
    CaelixLightStats stats{};
    caelix_scene_get_light_stats(rig.scene, &stats);
    return stats;
}

}  // namespace

TEST_CASE("light tree: structure, sampling and updates (block-ID)", "[light_tree]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_light_tree_test", test::RigDesc(1, 4, 0));
    {
        Probe probe(rig.gpu);
        SetTestMaterials(rig.scene);
        SetLightOptions(rig.scene, 0.0f);  // default threshold 20
        REQUIRE(caelix_scene_get_light_tree(rig.scene) != 0);

        Model model;
        model.encoding = 1;
        ReadLuminance(rig.scene, model);
        std::mt19937 rng(42);
        model.entities.resize(2);
        model.entities[0].id = rig.entity;
        model.entities[0].world = BorderWorld(rng);
        // A second entity, rotated by 37 degrees and moved.
        {
            CaelixEntityDesc desc{};
            const float axis[3] = {0.3f, 0.8f, 0.5f};
            AxisAngle(axis, 37.0f, desc.rotation);
            desc.position[0] = 300.0f;
            desc.position[1] = 50.0f;
            desc.position[2] = -200.0f;
            REQUIRE(caelix_scene_add_entity(rig.scene, &desc, &model.entities[1].id) == CAELIX_OK);
            std::copy(desc.position, desc.position + 3, model.entities[1].position);
            std::copy(desc.rotation, desc.rotation + 4, model.entities[1].rotation);
            model.entities[1].world = SmallWorld(rng);
        }
        for (const EntityModel& entity : model.entities)
        {
            UploadWorld(rig, entity.id, entity.world, test::KeysOf(entity.world));
        }
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        SECTION("initial state")
        {
            CheckAll(rig, probe, model, 1);
        }

        SECTION("updates")
        {
            // Light bricks added inside the cluster group: only that group is rebuilt.
            const uint32_t cluster = GroupOf(rig, model.entities[0].id, {21, 5, 37});
            std::vector<uint32_t> before = Generations(probe, rig);
            {
                Brick brick = LampBrick(rng, 20);
                model.entities[0].world.bricks[{21, 5, 37}] = brick;
                UploadWorld(rig, model.entities[0].id, model.entities[0].world, {{21, 5, 37}});
            }
            RequireChangedGenerations(before, Generations(probe, rig), {cluster}, {cluster});
            CheckAll(rig, probe, model, 2, false);

            // Light bricks added across a group border.
            {
                const std::vector<Key> keys = {{15, 2, 31}, {16, 2, 31}, {15, 2, 32}};
                for (const Key& key : keys)
                {
                    model.entities[0].world.bricks[key] = LampBrick(rng, 25);
                }
                UploadWorld(rig, model.entities[0].id, model.entities[0].world, keys);
            }
            CheckAll(rig, probe, model, 3, false);

            // min_emission up (the warm lamps drop out), then down (the dim lamps join).
            SetLightOptions(rig.scene, 24.0f);
            model.min_emission = 24.0f;
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
            CheckAll(rig, probe, model, 4, false);
            SetLightOptions(rig.scene, 10.0f);
            model.min_emission = 10.0f;
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
            CheckAll(rig, probe, model, 5, false);

            // Entity 1 moved and rotated by 37 degrees about another axis: no group is rebuilt.
            before = Generations(probe, rig);
            {
                const float axis[3] = {-0.6f, 0.1f, 0.9f};
                AxisAngle(axis, 37.0f, model.entities[1].rotation);
                model.entities[1].position[0] = -150.0f;
                model.entities[1].position[1] = 20.0f;
                model.entities[1].position[2] = 80.0f;
                const EntityModel& moved = model.entities[1];
                REQUIRE(caelix_scene_set_entity_transform(rig.scene, moved.id, moved.position,
                                                          moved.rotation) == CAELIX_OK);
                test::FlushFrame(rig.gpu, rig.scene, rig.frames);
            }
            RequireChangedGenerations(before, Generations(probe, rig), {}, {});
            CheckAll(rig, probe, model, 6);

            // Light bricks removed (inside the cluster and at the border).
            {
                const std::vector<Key> keys = {{22, 6, 38}, {16, 2, 31}};
                for (const Key& key : keys)
                {
                    model.entities[0].world.bricks.erase(key);
                }
                RemoveKeys(rig, model.entities[0].id, keys);
            }
            CheckAll(rig, probe, model, 7, false);

            // Entity 1 removed.
            REQUIRE(caelix_scene_remove_entity(rig.scene, model.entities[1].id) == CAELIX_OK);
            model.entities.pop_back();
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
            CheckAll(rig, probe, model, 8);
        }

        SECTION("an edit recomputes only the changed brick records")
        {
            const Key edited = {21, 5, 37};
            const uint32_t cluster = GroupOf(rig, model.entities[0].id, edited);
            const std::map<Key, std::vector<uint8_t>> before =
                RecordsByKey(rig, probe, cluster, BuildReference(rig, model));
            model.entities[0].world.bricks[edited] = LampBrick(rng, 30);
            UploadWorld(rig, model.entities[0].id, model.entities[0].world, {edited});
            const Reference ref = BuildReference(rig, model);
            const std::map<Key, std::vector<uint8_t>> after = RecordsByKey(rig, probe, cluster, ref);
            // Recomputed: the edited brick and its re-finalized face neighbours that are light bricks.
            std::set<Key> touched = {edited};
            for (const auto& d : kDir)
            {
                touched.insert({edited[0] + d[0], edited[1] + d[1], edited[2] + d[2]});
            }
            uint32_t recomputed = 0;
            uint32_t copied = 0;
            for (const auto& [key, bytes] : after)
            {
                if (touched.count(key) != 0)
                {
                    ++recomputed;
                }
                else
                {
                    ++copied;
                    REQUIRE(before.count(key) == 1);
                    REQUIRE(before.at(key) == bytes);
                }
            }
            CaelixLightStats stats{};
            caelix_scene_get_light_stats(rig.scene, &stats);
            REQUIRE(stats.records_recomputed == recomputed);
            REQUIRE(stats.records_copied == copied);
            REQUIRE(copied > 40);
            CheckAll(rig, probe, model, 9, false);
        }
    }
    test::CloseRig(rig);
}

TEST_CASE("light tree: radix sort against std::sort", "[light_tree]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_light_tree_test", test::RigDesc(1, 4, 0));
    {
        const ComputeKernel histogram(rig.gpu, "caelix_light_bvh_sort_histogram", sizeof(LightSortPush));
        const ComputeKernel scan(rig.gpu, "caelix_light_bvh_scan", sizeof(LightScanPush));
        const ComputeKernel scatter(rig.gpu, "caelix_light_bvh_sort_scatter", sizeof(LightSortPush));
        std::mt19937 rng(3);
        for (const uint32_t n : {1u, 2u, 1000u, 100000u})
        {
            INFO("n = " << n);
            const uint32_t blocks = (n + caelix::kLtSortBlock - 1) / caelix::kLtSortBlock;
            HostBuffer keys[2] = {CreateHostBuffer(rig.gpu, n * 4), CreateHostBuffer(rig.gpu, n * 4)};
            HostBuffer values[2] = {CreateHostBuffer(rig.gpu, n * 4), CreateHostBuffer(rig.gpu, n * 4)};
            HostBuffer counts =
                CreateHostBuffer(rig.gpu, uint64_t{caelix::kLtSortDigits} * sizeof(uint32_t) * blocks);
            HostBuffer count = CreateHostBuffer(rig.gpu, 16);
            std::vector<std::pair<uint32_t, uint32_t>> expected(n);
            for (uint32_t i = 0; i < n; ++i)
            {
                // 30-bit keys with many duplicates.
                const uint32_t key = (rng() % 3 == 0) ? (rng() % 64) * 0x00FFFFFu : (rng() & 0x3FFFFFFFu);
                reinterpret_cast<uint32_t*>(keys[0].data)[i] = key;
                reinterpret_cast<uint32_t*>(values[0].data)[i] = i;
                expected[i] = {key, i};
            }
            std::memcpy(count.data, &n, 4);
            std::sort(expected.begin(), expected.end());
            test::SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
                for (uint32_t pass = 0; pass < caelix::kLtSortPasses; ++pass)
                {
                    const uint32_t in = pass & 1u;
                    const LightSortPush push{keys[in].address, values[in].address, keys[1 - in].address,
                                             values[1 - in].address, counts.address, count.address, 0,
                                             caelix::kLtSortBits * pass, blocks};
                    histogram.Dispatch(cmd, &push, blocks);
                    const LightScanPush scan_push{counts.address, 0, 0, caelix::kLtSortDigits * blocks, 0};
                    scan.Dispatch(cmd, &scan_push, 1);
                    scatter.Dispatch(cmd, &push, blocks);
                }
            });
            for (uint32_t i = 0; i < n; ++i)
            {
                INFO("element " << i);
                REQUIRE(reinterpret_cast<const uint32_t*>(keys[0].data)[i] == expected[i].first);
                REQUIRE(reinterpret_cast<const uint32_t*>(values[0].data)[i] == expected[i].second);
            }
            for (HostBuffer* buffer : {&keys[0], &keys[1], &values[0], &values[1], &counts, &count})
            {
                DestroyHostBuffer(rig.gpu, *buffer);
            }
        }
    }
    test::CloseRig(rig);
}

TEST_CASE("light tree: static and dynamic entities", "[light_tree]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_light_tree_test", test::RigDesc(1, 4, 0));
    {
        Probe probe(rig.gpu);
        SetTestMaterials(rig.scene);
        SetLightOptions(rig.scene, 0.0f);
        Model model;
        ReadLuminance(rig.scene, model);
        std::mt19937 rng(23);
        model.entities.resize(2);
        model.entities[0].id = rig.entity;
        model.entities[0].world = BorderWorld(rng);
        {
            CaelixEntityDesc desc{};
            desc.rotation[3] = 1.0f;
            desc.position[0] = -80.0f;
            REQUIRE(caelix_scene_add_entity(rig.scene, &desc, &model.entities[1].id) == CAELIX_OK);
            model.entities[1].position[0] = -80.0f;
            model.entities[1].world = SmallWorld(rng);
        }
        for (const EntityModel& entity : model.entities)
        {
            UploadWorld(rig, entity.id, entity.world, test::KeysOf(entity.world));
        }
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        CheckAll(rig, probe, model, 31, false);
        REQUIRE(LightStats(rig).tree_leaves[1] == 0);

        // Entity 1 becomes dynamic: its lights move to tree 1.
        const CaelixEntityId second = model.entities[1].id;
        test::ExpectLibraryErrors(1, [&] {
            REQUIRE(caelix_scene_set_entity_motion(rig.scene, second, 2) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        REQUIRE(caelix_scene_set_entity_motion(rig.scene, second, CAELIX_ENTITY_DYNAMIC) == CAELIX_OK);
        model.entities[1].motion = CAELIX_ENTITY_DYNAMIC;
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        CheckAll(rig, probe, model, 32, false);
        REQUIRE(LightStats(rig).tree_leaves[1] > 0);

        // The dynamic entity moves every frame: only tree 1 is rebuilt.
        const CaelixLightStats before = LightStats(rig);
        for (int frame = 0; frame < 5; ++frame)
        {
            const float axis[3] = {0.2f, 1.0f, 0.1f};
            AxisAngle(axis, 37.0f + 11.0f * static_cast<float>(frame), model.entities[1].rotation);
            model.entities[1].position[1] = 3.0f * static_cast<float>(frame);
            const EntityModel& moved = model.entities[1];
            REQUIRE(caelix_scene_set_entity_transform(rig.scene, moved.id, moved.position, moved.rotation) ==
                    CAELIX_OK);
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        }
        CaelixLightStats after = LightStats(rig);
        REQUIRE(after.tree_builds[0] == before.tree_builds[0]);
        REQUIRE(after.tree_builds[1] == before.tree_builds[1] + 5);
        CheckAll(rig, probe, model, 33);

        // The static entity moves: tree 0 is rebuilt.
        model.entities[0].position[2] = 12.0f;
        REQUIRE(caelix_scene_set_entity_transform(rig.scene, model.entities[0].id, model.entities[0].position,
                                                  model.entities[0].rotation) == CAELIX_OK);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        const CaelixLightStats moved = LightStats(rig);
        REQUIRE(moved.tree_builds[0] == after.tree_builds[0] + 1);
        REQUIRE(moved.tree_builds[1] == after.tree_builds[1]);
        CheckAll(rig, probe, model, 34, false);

        // A light edit in the dynamic entity: only tree 1 is rebuilt.
        {
            const Key key = model.entities[1].world.bricks.begin()->first;
            model.entities[1].world.bricks[key] = LampBrick(rng, 30);
            UploadWorld(rig, model.entities[1].id, model.entities[1].world, {key});
        }
        const CaelixLightStats edited = LightStats(rig);
        REQUIRE(edited.tree_builds[0] == moved.tree_builds[0]);
        REQUIRE(edited.tree_builds[1] == moved.tree_builds[1] + 1);
        CheckAll(rig, probe, model, 36, false);

        // Back to static: tree 1 is empty.
        REQUIRE(caelix_scene_set_entity_motion(rig.scene, second, CAELIX_ENTITY_STATIC) == CAELIX_OK);
        model.entities[1].motion = CAELIX_ENTITY_STATIC;
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        CheckAll(rig, probe, model, 35, false);
        REQUIRE(LightStats(rig).tree_leaves[1] == 0);
    }
    test::CloseRig(rig);
}

TEST_CASE("light tree: re-uploaded bricks", "[light_tree]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_light_tree_test", test::RigDesc(1, 4, 0));
    {
        Probe probe(rig.gpu);
        SetTestMaterials(rig.scene);
        SetLightOptions(rig.scene, 0.0f);
        Model model;
        ReadLuminance(rig.scene, model);
        std::mt19937 rng(61);
        model.entities.resize(1);
        model.entities[0].id = rig.entity;
        World& world = model.entities[0].world;
        // One group: an isolated light brick (its only neighbour is stone), and two other light
        // bricks away from it.
        Brick stone{};
        stone.fill(kStone);
        const Key isolated = {2, 2, 2};
        const Key lamp = {7, 7, 7};
        world.bricks[isolated] = LampBrick(rng, 20);
        world.bricks[{3, 2, 2}] = stone;
        world.bricks[lamp] = LampBrick(rng, 20);
        world.bricks[{11, 3, 9}] = LampBrick(rng, 20);
        UploadWorld(rig, rig.entity, world, test::KeysOf(world));
        CheckAll(rig, probe, model, 41, false);
        const uint32_t group = GroupOf(rig, rig.entity, isolated);

        // The isolated light brick becomes stone and air: no light neighbour marks the group, so
        // only the brick's own old light state (record word 1, kept by the upload) removes it.
        {
            Brick dark{};
            for (uint32_t v = 0; v < 512; ++v)
            {
                dark[v] = (v % 3 == 0) ? kStone : uint16_t{0};
            }
            world.bricks[isolated] = dark;
            const std::vector<uint32_t> before = Generations(probe, rig);
            UploadWorld(rig, rig.entity, world, {isolated});
            RequireChangedGenerations(before, Generations(probe, rig), {group}, {group});
            REQUIRE(LightStats(rig).light_bricks == 2);
            CheckAll(rig, probe, model, 42, false);
        }

        // A light brick with a new light layout: its record is recomputed, the other one copied.
        {
            world.bricks[lamp] = LampBrick(rng, 35);
            UploadWorld(rig, rig.entity, world, {lamp});
            const CaelixLightStats stats = LightStats(rig);
            REQUIRE(stats.records_recomputed == 1);
            REQUIRE(stats.records_copied == 1);
            CheckAll(rig, probe, model, 43, false);
        }

        // One window: a new brick uploaded twice, a resident brick replaced and then removed while a
        // new brick takes its record, a resident brick replaced twice. Every record must end with the
        // newest content.
        {
            const auto upload = [&](const Key& key) {
                const CaelixBrickKey k{key[0], key[1], key[2]};
                uint32_t consumed = 0;
                REQUIRE(caelix_scene_upload_bricks(rig.scene, rig.entity, 1, &k, world.bricks.at(key).data(),
                                                   &consumed) == CAELIX_OK);
                REQUIRE(consumed == 1);
            };
            const Key fresh = {5, 9, 1};
            const Key reused = {12, 12, 12};
            world.bricks[fresh] = LampBrick(rng, 20);
            upload(fresh);
            world.bricks[fresh] = LampBrick(rng, 40);
            upload(fresh);
            world.bricks[isolated] = LampBrick(rng, 25);
            upload(isolated);
            const CaelixBrickKey removed{isolated[0], isolated[1], isolated[2]};
            REQUIRE(caelix_scene_remove_bricks(rig.scene, rig.entity, 1, &removed) == CAELIX_OK);
            world.bricks.erase(isolated);
            world.bricks[reused] = LampBrick(rng, 30);
            upload(reused);
            world.bricks[lamp] = LampBrick(rng, 15);
            upload(lamp);
            world.bricks[lamp] = LampBrick(rng, 45);
            upload(lamp);
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
            CheckAll(rig, probe, model, 44, false);
        }
    }
    test::CloseRig(rig);
}

TEST_CASE("light tree: heap growth from a tiny heap", "[light_tree]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_light_tree_test", test::RigDesc(1, 4, 0));
    {
        Probe probe(rig.gpu);
        SetTestMaterials(rig.scene);
        SetLightOptions(rig.scene, 0.0f, 4096);
        Model model;
        ReadLuminance(rig.scene, model);
        std::mt19937 rng(7);
        model.entities.resize(1);
        model.entities[0].id = rig.entity;
        model.entities[0].world = BorderWorld(rng);
        UploadWorld(rig, rig.entity, model.entities[0].world, test::KeysOf(model.entities[0].world));
        for (int i = 0; i < 4; ++i)
        {
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        }
        CaelixLightStats stats{};
        caelix_scene_get_light_stats(rig.scene, &stats);
        REQUIRE(stats.failed_passes >= 1);
        REQUIRE(stats.passes > stats.failed_passes);
        REQUIRE(stats.heap_bytes_capacity > 4096);
        REQUIRE(stats.heap_bytes_used <= stats.heap_bytes_capacity);
        CheckAll(rig, probe, model, 11, false);
    }
    test::CloseRig(rig);
}

TEST_CASE("light tree: enabled after the upload", "[light_tree]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_light_tree_test", test::RigDesc(1, 4, 0));
    {
        Probe probe(rig.gpu);
        SetTestMaterials(rig.scene);
        Model model;
        ReadLuminance(rig.scene, model);
        std::mt19937 rng(17);
        model.entities.resize(1);
        model.entities[0].id = rig.entity;
        model.entities[0].world = BorderWorld(rng);
        UploadWorld(rig, rig.entity, model.entities[0].world, test::KeysOf(model.entities[0].world));
        // Every resident brick is finalized again with the light code set.
        SetLightOptions(rig.scene, 0.0f);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        CheckAll(rig, probe, model, 13, false);
    }
    test::CloseRig(rig);
}

TEST_CASE("light tree: packed encoding", "[light_tree]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_light_tree_test", test::RigDesc(0, 4, 0));
    {
        Probe probe(rig.gpu);
        SetLightOptions(rig.scene, 0.0f);
        Model model;
        model.encoding = 0;
        ReadLuminance(rig.scene, model);
        // Packed emissive codes (bits 15..14 = 01): albedo rgb4 / 15, emission albedo * 2^(3k).
        const auto emissive = [](uint32_t r, uint32_t g, uint32_t b, uint32_t k) {
            return static_cast<uint16_t>(0x4000u | (r << 10) | (g << 6) | (b << 2) | k);
        };
        const uint16_t codes[] = {emissive(15, 15, 15, 2), emissive(1, 1, 1, 3), emissive(4, 4, 4, 1),
                                  emissive(15, 8, 2, 1), 0xC000u | 0x1234u, 0x8000u | 0x0421u, 5};
        REQUIRE(model.luminance[codes[0]] >= 20.0f);
        REQUIRE(model.luminance[codes[2]] < 20.0f);
        std::mt19937 rng(99);
        model.entities.resize(1);
        model.entities[0].id = rig.entity;
        for (int32_t x = -2; x <= 1; ++x)
        {
            for (int32_t y = 0; y <= 1; ++y)
            {
                Brick brick{};
                for (uint16_t& v : brick)
                {
                    v = rng() % 2 == 0 ? uint16_t{0} : codes[rng() % 7];
                }
                model.entities[0].world.bricks[{x, y, 15}] = brick;
                model.entities[0].world.bricks[{x, y, 16}] = brick;
            }
        }
        UploadWorld(rig, rig.entity, model.entities[0].world, test::KeysOf(model.entities[0].world));
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        CheckAll(rig, probe, model, 12);
    }
    test::CloseRig(rig);
}

TEST_CASE("light tree: argument checks and disabling", "[light_tree]")
{
    test::Rig rig;
    test::OpenRig(rig, "caelix_light_tree_test", test::RigDesc(1, 4, 0));
    CaelixLightOptions options{};
    const auto invalid = [&](const CaelixLightOptions* argument) {
        test::ExpectLibraryErrors(1, [&] {
            REQUIRE(caelix_scene_set_light_options(rig.scene, argument) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
    };
    options.enabled = 2;
    invalid(&options);
    options.enabled = 1;
    options.min_emission = -1.0f;
    invalid(&options);
    options.min_emission = NAN;
    invalid(&options);
    options.min_emission = 0.0f;
    options.heap_bytes = caelix::kLtMaxHeapUnits * caelix::kLtUnitBytes + caelix::kLtUnitBytes;
    invalid(&options);
    options.heap_bytes = 0;
    invalid(nullptr);
    REQUIRE(caelix_scene_get_light_tree(rig.scene) == 0);
    CaelixLightStats stats{};
    caelix_scene_get_light_stats(rig.scene, &stats);
    REQUIRE(stats.enabled == 0);
    options.min_emission = 0.0f;
    REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_OK);
    REQUIRE(caelix_scene_get_light_tree(rig.scene) != 0);
    test::FlushFrame(rig.gpu, rig.scene, rig.frames);
    options.enabled = 0;
    REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_OK);
    REQUIRE(caelix_scene_get_light_tree(rig.scene) == 0);
    test::FlushFrame(rig.gpu, rig.scene, rig.frames);
    test::CloseRig(rig);
}

namespace {

// One benchmark scene: `gx * gy * gz` groups of 67 light bricks (4 lamp voxels in air each).
void RunBench(int32_t gx_count, int32_t gy_count, int32_t gz_count)
{
    test::Rig rig;
    CaelixSceneDesc desc = test::RigDesc(1, 4, 0);
    desc.staging_bytes = 256ull << 20;
    test::OpenRig(rig, "caelix_light_tree_bench", desc);
    {
        Probe probe(rig.gpu);
        SetTestMaterials(rig.scene);
        SetLightOptions(rig.scene, 0.0f, 64ull << 20);
        std::mt19937 rng(5);
        std::vector<CaelixBrickKey> keys;
        for (int32_t gz = 0; gz < gz_count; ++gz)
        {
            for (int32_t gy = 0; gy < gy_count; ++gy)
            {
                for (int32_t gx = 0; gx < gx_count; ++gx)
                {
                    for (int32_t b = 0; b < 67; ++b)
                    {
                        keys.push_back(
                            {gx * 16 + (b & 3) * 2, gy * 16 + ((b >> 2) & 3) * 2, gz * 16 + (b >> 4) * 2});
                    }
                }
            }
        }
        const auto make_brick = [&](uint16_t* brick) {
            std::fill(brick, brick + 512, uint16_t{0});
            for (int k = 0; k < 4; ++k)
            {
                brick[rng() % 512] = k == 0 ? kLampBright : kLampWarm;
            }
        };
        const auto start = std::chrono::steady_clock::now();
        constexpr size_t kChunk = 1u << 16;
        std::vector<uint16_t> blocks(kChunk * 512);
        for (size_t first = 0; first < keys.size(); first += kChunk)
        {
            const size_t chunk = std::min(kChunk, keys.size() - first);
            for (size_t i = 0; i < chunk; ++i)
            {
                make_brick(blocks.data() + i * 512);
            }
            size_t done = 0;
            while (done < chunk)
            {
                uint32_t consumed = 0;
                REQUIRE(caelix_scene_upload_bricks(rig.scene, rig.entity, static_cast<uint32_t>(chunk - done),
                                                   keys.data() + first + done, blocks.data() + done * 512,
                                                   &consumed) == CAELIX_OK);
                done += consumed;
                if (done < chunk)
                {
                    test::FlushFrame(rig.gpu, rig.scene, rig.frames);
                }
            }
        }
        for (int i = 0; i < 6; ++i)
        {
            test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        }
        const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        CaelixLightStats stats = LightStats(rig);
        std::printf("bench %zu bricks: %llu light bricks, %llu faces, %u groups, heap %llu / %llu bytes, "
                    "passes %u (failed %u), static BVH %u leaves / %u nodes, load %.1f s\n",
                    keys.size(), static_cast<unsigned long long>(stats.light_bricks),
                    static_cast<unsigned long long>(stats.light_faces), stats.light_groups,
                    static_cast<unsigned long long>(stats.heap_bytes_used),
                    static_cast<unsigned long long>(stats.heap_bytes_capacity), stats.passes,
                    stats.failed_passes, stats.tree_leaves[0], stats.tree_nodes[0], load_s);
        REQUIRE(stats.light_bricks == keys.size());
        REQUIRE(stats.tree_leaves[0] == keys.size());

        // The static BVH alone (the entity moves: gather + build, no group pass).
        const float position[3] = {1.0f, 0.0f, 0.0f};
        const float rotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        REQUIRE(caelix_scene_set_entity_transform(rig.scene, rig.entity, position, rotation) == CAELIX_OK);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        stats = LightStats(rig);
        std::printf("bench %zu bricks: static BVH gather + build: %.3f ms GPU\n", keys.size(),
                    static_cast<double>(stats.gpu_ms));

        // A group pass with one changed group (every other block is copied), then both BVHs.
        make_brick(blocks.data());
        blocks[100] = kLampBright;
        uint32_t consumed = 0;
        REQUIRE(caelix_scene_upload_bricks(rig.scene, rig.entity, 1, keys.data(), blocks.data(), &consumed) ==
                CAELIX_OK);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        stats = LightStats(rig);
        std::printf("bench %zu bricks: group pass with one rebuilt group + BVHs: %.3f ms GPU "
                    "(%u records recomputed, %u copied)\n",
                    keys.size(), static_cast<double>(stats.gpu_ms), stats.records_recomputed,
                    stats.records_copied);

        // Every group rebuilt (the threshold moves the warm lamps out: every record is recomputed).
        SetLightOptions(rig.scene, 24.0f);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        test::FlushFrame(rig.gpu, rig.scene, rig.frames);
        stats = LightStats(rig);
        std::printf("bench %zu bricks: group pass with every group rebuilt + BVHs: %.3f ms GPU, "
                    "%u records recomputed\n",
                    keys.size(), static_cast<double>(stats.gpu_ms), stats.records_recomputed);
        REQUIRE(stats.records_recomputed == stats.light_bricks);

        // Draws: 1M threads x 16 draws for 64 receivers spread over the scene.
        std::vector<Receiver> receivers(64);
        for (Receiver& receiver : receivers)
        {
            receiver.position[0] = static_cast<float>(rng() % static_cast<uint32_t>(gx_count * 128));
            receiver.position[1] = static_cast<float>(rng() % static_cast<uint32_t>(gy_count * 128));
            receiver.position[2] = static_cast<float>(rng() % static_cast<uint32_t>(gz_count * 128));
            receiver.normal[1] = (rng() % 2 == 0) ? 1.0f : 0.0f;
        }
        constexpr uint32_t kThreads = 1u << 20;
        constexpr uint32_t kDrawsPerThread = 16;
        const uint64_t descriptor = caelix_scene_get_light_tree(rig.scene);
        probe.Run(descriptor, kModeBench, kThreads, kDrawsPerThread, 0, nullptr, 0, receivers.data(),
                  receivers.size() * sizeof(Receiver), kThreads * 4);
        const double ms = probe.Run(descriptor, kModeBench, kThreads, kDrawsPerThread, 0, nullptr, 0,
                                    receivers.data(), receivers.size() * sizeof(Receiver), kThreads * 4);
        const double ns_per_draw = ms * 1e6 / (static_cast<double>(kThreads) * kDrawsPerThread);
        std::printf("bench %zu bricks: %u LightTreeSample draws in %.3f ms = %.3f ns per draw\n", keys.size(),
                    kThreads * kDrawsPerThread, ms, ns_per_draw);
    }
    test::CloseRig(rig);
}

}  // namespace

TEST_CASE("light tree: benchmark at about 10k, 54k and 1M light bricks", "[.bench]")
{
    RunBench(5, 6, 5);     // 150 groups, 10050 bricks
    RunBench(10, 10, 8);   // 800 groups, 53600 bricks
    RunBench(25, 25, 24);  // 15000 groups, 1005000 bricks
}
