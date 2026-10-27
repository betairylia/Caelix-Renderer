// Upload benchmark (not a ctest): the record upload of a synthetic FractalCave, through the PUBLIC C
// ABI. One entity holds a resident box of 48 x 128 x 48 bricks; every measured frame re-uploads one
// slab of 8 brick rows (y in [8 * (p % 16), 8 * (p % 16) + 8), 18,432 bricks) with toggled content,
// and the GPU time of caelix_scene_flush is read from two timestamps around it. The edit loop of
// FractalCave (CorruptionTick) re-uploads about the same number of bricks per frame.
//
//   caelix_upload_bench [--path auto|bar|bounce] [--frames N] [--warmup N] [--bricks N]
//                       [--staging-mib N] [--no-validation]
//
// --path sets the environment variable CAELIX_UPLOAD_PATH before caelix_scene_create (the library
// reads it there). --bricks limits the per-frame set to the first N bricks of the slab, or extends it
// over the following slabs when N is above one slab (up to the whole box, ring permitting). Use
// --no-validation for timings. The helpers of tests/test_device.h use Catch2 assertions, so the
// benchmark runs as the only test case of a Catch2 session.
#include "test_device.h"

#include <catch2/catch_session.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace test;

namespace {

constexpr uint32_t kBlocksPerBrick = 512;
constexpr uint32_t kRecordWords = 274;  // 16-bit blocks
constexpr uint32_t kRecordPayloadWord = 18;
constexpr uint32_t kVerifyStride = 97;  // every 97th brick of the box is read back at the end
constexpr int32_t kBoxX = 48;
constexpr int32_t kBoxY = 128;
constexpr int32_t kBoxZ = 48;
constexpr int32_t kSlabRows = 8;  // brick rows (y) per measured frame
constexpr int32_t kSlabs = kBoxY / kSlabRows;
constexpr uint32_t kSlabBricks = kBoxX * kSlabRows * kBoxZ;  // 18,432
constexpr uint32_t kBoxBricks = kBoxX * kBoxY * kBoxZ;         // 294,912
constexpr uint32_t kLoadChunkBricks = 32768;  // bricks per upload call while loading
// Packed encoding: an opaque value (bits 14..15 set).
constexpr uint16_t kSolidBlock = 0xC3F9;

struct Options
{
    std::string path = "auto";
    uint32_t frames = 200;
    uint32_t warmup = 30;
    uint32_t bricks = kSlabBricks;
    uint64_t staging_mib = 256;
    bool validation = true;
};
Options g_options;

void PrintUsage()
{
    std::printf("usage: caelix_upload_bench [--path auto|bar|bounce] [--frames N] [--warmup N] "
                "[--bricks N (1..%u)] [--staging-mib N] [--no-validation]\n",
                kSlabBricks);
}

bool ParseUint(const char* text, uint64_t min, uint64_t max, uint64_t& out)
{
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value < min || value > max)
    {
        return false;
    }
    out = value;
    return true;
}

bool ParseArguments(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        uint64_t value = 0;
        if (arg == "--no-validation")
        {
            g_options.validation = false;
        }
        else if (arg == "--path" && has_value)
        {
            g_options.path = argv[++i];
            if (g_options.path != "auto" && g_options.path != "bar" && g_options.path != "bounce")
            {
                return false;
            }
        }
        else if (arg == "--frames" && has_value && ParseUint(argv[++i], 1, 1000000, value))
        {
            g_options.frames = static_cast<uint32_t>(value);
        }
        else if (arg == "--warmup" && has_value && ParseUint(argv[++i], 0, 1000000, value))
        {
            g_options.warmup = static_cast<uint32_t>(value);
        }
        else if (arg == "--bricks" && has_value && ParseUint(argv[++i], 1, kBoxBricks, value))
        {
            g_options.bricks = static_cast<uint32_t>(value);
        }
        else if (arg == "--staging-mib" && has_value && ParseUint(argv[++i], 4, 1u << 16, value))
        {
            g_options.staging_mib = value;
        }
        else
        {
            return false;
        }
    }
    return true;
}

// Brick index of a key in the box: x fastest, then y, then z.
uint32_t BrickIndex(int32_t x, int32_t y, int32_t z)
{
    return static_cast<uint32_t>(x + kBoxX * (y + kBoxY * z));
}

// The two contents of a brick: A = a voxel checkerboard, B = its inverse.
void MakeContents(std::vector<uint16_t>& a, std::vector<uint16_t>& b)
{
    a.resize(kBlocksPerBrick);
    b.resize(kBlocksPerBrick);
    for (uint32_t i = 0; i < kBlocksPerBrick; ++i)
    {
        const uint32_t x = i & 7;
        const uint32_t y = (i >> 3) & 7;
        const uint32_t z = i >> 6;
        const bool odd = ((x + y + z) & 1) != 0;
        a[i] = odd ? kSolidBlock : 0;
        b[i] = odd ? 0 : kSolidBlock;
    }
}

double Mean(const std::vector<double>& values)
{
    double sum = 0.0;
    for (const double v : values)
    {
        sum += v;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

// The value at quantile q (0..1) of the sorted values (nearest rank).
double Quantile(std::vector<double> values, double q)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t rank = static_cast<size_t>(q * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(rank, values.size() - 1)];
}

// Reads back the records of every kVerifyStride-th brick and compares the position and the payload
// with the content the brick got last (finalize leaves these values raw: opaque blocks and air
// between opaque or equal-id neighbours). Returns the number of bricks checked.
uint32_t VerifyRecords(Gpu& gpu, const CaelixScene* scene, CaelixEntityId entity,
                       const std::vector<CaelixBrickKey>& keys, const std::vector<uint8_t>& phase,
                       const std::vector<uint16_t> (&content)[2])
{
    std::vector<uint32_t> picked;
    for (uint32_t i = 0; i < keys.size(); i += kVerifyStride)
    {
        picked.push_back(i);
    }
    constexpr VkDeviceSize kRecordBytes = kRecordWords * sizeof(uint32_t);
    Readback readback;
    CreateReadback(gpu, picked.size() * kRecordBytes, readback);
    std::vector<CaelixBrickInfo> infos(picked.size());
    for (size_t k = 0; k < picked.size(); ++k)
    {
        REQUIRE(caelix_scene_query_brick(scene, entity, keys[picked[k]], &infos[k]) == CAELIX_OK);
        REQUIRE(infos[k].resident == 1);
    }
    SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
        for (size_t k = 0; k < picked.size(); ++k)
        {
            const VkBufferCopy region{infos[k].record_page_offset_bytes, k * kRecordBytes, kRecordBytes};
            vkCmdCopyBuffer(cmd, reinterpret_cast<VkBuffer>(infos[k].record_page_buffer), readback.buffer, 1,
                            &region);
        }
        CopyToHostBarrier(cmd);
    });
    for (size_t k = 0; k < picked.size(); ++k)
    {
        const CaelixBrickKey& key = keys[picked[k]];
        const uint32_t* record = readback.words + k * kRecordWords;
        const uint32_t pos = static_cast<uint32_t>(key.x & 15) | (static_cast<uint32_t>(key.y & 15) << 4) |
                             (static_cast<uint32_t>(key.z & 15) << 8);
        INFO("brick (" << key.x << ", " << key.y << ", " << key.z << ")");
        REQUIRE((record[0] & 0xFFFFu) == pos);
        const std::vector<uint16_t>& blocks = content[phase[picked[k]]];
        REQUIRE(std::memcmp(record + kRecordPayloadWord, blocks.data(), kBlocksPerBrick * sizeof(uint16_t)) == 0);
    }
    DestroyReadback(gpu, readback);
    return static_cast<uint32_t>(picked.size());
}

double MsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// One frame of the bench host: a command buffer with a timestamp pair around caelix_scene_flush,
// submitted and waited for, then the frame is recycled.
class FrameRunner
{
public:
    explicit FrameRunner(Gpu& gpu)
        : gpu_(gpu)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(gpu.device.physical_device.physical_device, &properties);
        timestamp_period_ns_ = properties.limits.timestampPeriod;
        VkQueryPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        pool_info.queryCount = 2;
        REQUIRE(vkCreateQueryPool(gpu.device.device, &pool_info, nullptr, &queries_) == VK_SUCCESS);
        VkCommandBufferAllocateInfo allocate_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate_info.commandPool = gpu.pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = 1;
        REQUIRE(vkAllocateCommandBuffers(gpu.device.device, &allocate_info, &cmd_) == VK_SUCCESS);
    }
    ~FrameRunner()
    {
        vkFreeCommandBuffers(gpu_.device.device, gpu_.pool, 1, &cmd_);
        vkDestroyQueryPool(gpu_.device.device, queries_, nullptr);
    }
    FrameRunner(const FrameRunner&) = delete;
    FrameRunner& operator=(const FrameRunner&) = delete;

    // Flushes the scene as frame `next_frame_`; returns the GPU ms between the timestamps and the
    // CPU ms of the caelix_scene_flush call.
    void Run(CaelixScene* scene, double& gpu_ms, double& cpu_flush_ms)
    {
        const uint64_t frame = next_frame_++;
        REQUIRE(vkResetCommandPool(gpu_.device.device, gpu_.pool, 0) == VK_SUCCESS);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        REQUIRE(vkBeginCommandBuffer(cmd_, &begin) == VK_SUCCESS);
        vkCmdResetQueryPool(cmd_, queries_, 0, 2);
        vkCmdWriteTimestamp2(cmd_, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries_, 0);
        CaelixFrameDesc desc{};
        desc.command_buffer = reinterpret_cast<uint64_t>(cmd_);
        desc.frame_number = frame;
        desc.safe_frame_number = frame - 1;
        const auto flush_start = std::chrono::steady_clock::now();
        const CaelixResult result = caelix_scene_flush(scene, &desc);
        cpu_flush_ms = MsSince(flush_start);
        REQUIRE(result == CAELIX_OK);
        vkCmdWriteTimestamp2(cmd_, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries_, 1);
        REQUIRE(vkEndCommandBuffer(cmd_) == VK_SUCCESS);
        VkCommandBufferSubmitInfo cmd_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cmd_info.commandBuffer = cmd_;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &cmd_info;
        REQUIRE(vkQueueSubmit2(gpu_.queue, 1, &submit, gpu_.fence) == VK_SUCCESS);
        REQUIRE(vkWaitForFences(gpu_.device.device, 1, &gpu_.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);
        REQUIRE(vkResetFences(gpu_.device.device, 1, &gpu_.fence) == VK_SUCCESS);
        uint64_t stamps[2] = {};
        REQUIRE(vkGetQueryPoolResults(gpu_.device.device, queries_, 0, 2, sizeof(stamps), stamps,
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
                VK_SUCCESS);
        gpu_ms = static_cast<double>(stamps[1] - stamps[0]) * timestamp_period_ns_ * 1e-6;
        // The frame has completed: its staging space can be reused right away.
        REQUIRE(caelix_scene_recycle(scene, frame) == CAELIX_OK);
    }

private:
    Gpu& gpu_;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    double timestamp_period_ns_ = 1.0;
    uint64_t next_frame_ = 1;
};

}  // namespace

TEST_CASE("upload bench")
{
    const Options& o = g_options;
    caelix_set_log_callback(LibraryLog, nullptr);
    Gpu gpu;
    CreateGpu(gpu, "caelix_upload_bench", o.validation);
    CaelixDevice* device = AdoptDevice(gpu);

    CaelixSceneDesc scene_desc{};
    scene_desc.staging_bytes = o.staging_mib << 20;
    CaelixScene* scene = nullptr;
    REQUIRE(caelix_scene_create(device, &scene_desc, &scene) == CAELIX_OK);
    CaelixEntityDesc entity_desc{};
    entity_desc.rotation[3] = 1.0f;
    CaelixEntityId entity = CAELIX_INVALID_ENTITY;
    REQUIRE(caelix_scene_add_entity(scene, &entity_desc, &entity) == CAELIX_OK);

    std::vector<uint16_t> content[2];
    MakeContents(content[0], content[1]);
    std::vector<uint8_t> phase(static_cast<size_t>(kBoxX) * kBoxY * kBoxZ, 0);
    std::vector<CaelixBrickKey> keys;
    std::vector<uint16_t> blocks;
    std::optional<FrameRunner> frames;  // destroyed before the device
    frames.emplace(gpu);
    FrameRunner& runner = *frames;

    // Load: the whole box with content A, as many bricks per frame as the ring takes.
    std::vector<CaelixBrickKey> all_keys;
    all_keys.reserve(phase.size());
    for (int32_t z = 0; z < kBoxZ; ++z)
    {
        for (int32_t y = 0; y < kBoxY; ++y)
        {
            for (int32_t x = 0; x < kBoxX; ++x)
            {
                all_keys.push_back({x, y, z});
            }
        }
    }
    double load_gpu_ms = 0.0;
    uint32_t load_frames = 0;
    size_t done = 0;
    const auto load_start = std::chrono::steady_clock::now();
    while (done < all_keys.size())
    {
        for (;;)
        {
            const uint32_t n = static_cast<uint32_t>(std::min<size_t>(kLoadChunkBricks, all_keys.size() - done));
            if (n == 0)
            {
                break;
            }
            blocks.resize(static_cast<size_t>(n) * kBlocksPerBrick);
            for (uint32_t i = 0; i < n; ++i)
            {
                std::memcpy(blocks.data() + static_cast<size_t>(i) * kBlocksPerBrick, content[0].data(),
                            kBlocksPerBrick * sizeof(uint16_t));
            }
            uint32_t consumed = 0;
            REQUIRE(caelix_scene_upload_bricks(scene, entity, n, all_keys.data() + done, blocks.data(),
                                               &consumed) == CAELIX_OK);
            done += consumed;
            if (consumed < n)
            {
                break;  // the ring is full: flush first
            }
        }
        double gpu_ms = 0.0;
        double cpu_ms = 0.0;
        runner.Run(scene, gpu_ms, cpu_ms);
        load_gpu_ms += gpu_ms;
        ++load_frames;
        REQUIRE(load_frames < 10000);
    }
    // Settle: the acceleration structures that waited for scratch space.
    for (int i = 0; i < 4; ++i)
    {
        double gpu_ms = 0.0;
        double cpu_ms = 0.0;
        runner.Run(scene, gpu_ms, cpu_ms);
        load_gpu_ms += gpu_ms;
        ++load_frames;
    }
    const double load_wall_ms = MsSince(load_start);
    CaelixSceneStats stats{};
    caelix_scene_get_stats(scene, &stats);
    REQUIRE(stats.bricks_resident == all_keys.size());
    std::printf("load: %zu bricks in %u frames, flush GPU total %.1f ms, wall %.1f ms\n", all_keys.size(),
                load_frames, load_gpu_ms, load_wall_ms);

    // Edit loop: frame p re-uploads the first `bricks` bricks of slab p % 16 with toggled content.
    std::vector<double> gpu_ms_list;
    std::vector<double> upload_ms_list;
    std::vector<double> flush_ms_list;
    const uint32_t total_frames = o.warmup + o.frames;
    for (uint32_t p = 0; p < total_frames; ++p)
    {
        // Slab p % 16 first; more than one slab (--bricks above 18,432) continues with the next
        // slabs, wrapping around the box.
        keys.clear();
        blocks.resize(static_cast<size_t>(o.bricks) * kBlocksPerBrick);
        for (int32_t s = 0; s < kSlabs && keys.size() < o.bricks; ++s)
        {
            const int32_t y0 = kSlabRows * static_cast<int32_t>((p + static_cast<uint32_t>(s)) % kSlabs);
            for (int32_t z = 0; z < kBoxZ && keys.size() < o.bricks; ++z)
            {
                for (int32_t y = y0; y < y0 + kSlabRows && keys.size() < o.bricks; ++y)
                {
                    for (int32_t x = 0; x < kBoxX && keys.size() < o.bricks; ++x)
                    {
                        uint8_t& bit = phase[BrickIndex(x, y, z)];
                        bit ^= 1;
                        std::memcpy(blocks.data() + keys.size() * kBlocksPerBrick, content[bit].data(),
                                    kBlocksPerBrick * sizeof(uint16_t));
                        keys.push_back({x, y, z});
                    }
                }
            }
        }
        uint32_t consumed = 0;
        const auto upload_start = std::chrono::steady_clock::now();
        const CaelixResult result = caelix_scene_upload_bricks(
            scene, entity, static_cast<uint32_t>(keys.size()), keys.data(), blocks.data(), &consumed);
        const double upload_ms = MsSince(upload_start);
        REQUIRE(result == CAELIX_OK);
        if (consumed < keys.size())
        {
            FAIL("frame " << p << ": the scene consumed " << consumed << " of " << keys.size()
                          << " bricks (the staging ring is too small for this --bricks)");
        }
        double gpu_ms = 0.0;
        double flush_ms = 0.0;
        runner.Run(scene, gpu_ms, flush_ms);
        if (p >= o.warmup)
        {
            gpu_ms_list.push_back(gpu_ms);
            upload_ms_list.push_back(upload_ms);
            flush_ms_list.push_back(flush_ms);
        }
    }

    // The records hold what was uploaded last (the phase of every brick).
    const uint32_t verified = VerifyRecords(gpu, scene, entity, all_keys, phase, content);
    std::printf("verified %u records (position and payload)\n", verified);

    caelix_scene_get_stats(scene, &stats);
    std::printf("scene: upload_path %llu, staging %llu bytes, record pages %llu, bricks resident %llu, "
                "groups %u, AS builds pending %llu\n",
                static_cast<unsigned long long>(stats.upload_path),
                static_cast<unsigned long long>(stats.staging_bytes),
                static_cast<unsigned long long>(stats.record_pages),
                static_cast<unsigned long long>(stats.bricks_resident), stats.groups,
                static_cast<unsigned long long>(stats.as_builds_pending));
    std::printf("RESULT path=%s bricks=%u staging_mib=%llu frames=%u warmup=%u validation=%s | flush GPU ms "
                "mean %.3f median %.3f p95 %.3f | CPU ms upload %.3f flush %.3f\n",
                o.path.c_str(), o.bricks, static_cast<unsigned long long>(o.staging_mib), o.frames, o.warmup,
                o.validation ? "on" : "off", Mean(gpu_ms_list), Quantile(gpu_ms_list, 0.5),
                Quantile(gpu_ms_list, 0.95), Mean(upload_ms_list), Mean(flush_ms_list));
    std::fflush(stdout);

    REQUIRE(vkDeviceWaitIdle(gpu.device.device) == VK_SUCCESS);
    frames.reset();
    caelix_scene_destroy(scene);
    caelix_device_destroy(device);
    DestroyGpu(gpu);
    caelix_set_log_callback(nullptr, nullptr);
    std::printf("messages: validation %d errors, %d warnings; library %d errors, %d warnings\n",
                g_validation_errors.load(), g_validation_warnings.load(), g_library_errors.load(),
                g_library_warnings.load());
}

int main(int argc, char** argv)
{
    if (!ParseArguments(argc, argv))
    {
        PrintUsage();
        return 2;
    }
#if defined(_MSC_VER)
    _putenv_s("CAELIX_UPLOAD_PATH", g_options.path.c_str());
#else
    setenv("CAELIX_UPLOAD_PATH", g_options.path.c_str(), 1);
#endif
    // The Catch2 session runs the single test case; the bench options are not Catch2 options.
    Catch::Session session;
    char* catch_argv[] = {argv[0]};
    return session.run(1, catch_argv);
}
