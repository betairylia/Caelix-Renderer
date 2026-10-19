// The DLSS upscalers (CaelixRenderOptions::upscaler, docs/DLSS.md) through the public C ABI:
// the availability query, the jitter helpers, DLSS Super Resolution and Ray Reconstruction on the
// sunlit floor + floating block scene of caelix_nrd_test (smoke tests against an accumulated
// reference), argument checks, quality changes and the measured upscale cost at 1920x1080. Any
// Vulkan validation message or library error/warning fails a case (CloseRig). Without DLSS in the
// build, or without NGX on the machine, the GPU cases are skipped (the query reports why).
#include "test_scenes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace test;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kNear = 0.1f;
constexpr float kFar = 1000.0f;
constexpr uint16_t kGreyFloor = 0x8000u | (15u << 10) | (15u << 5) | 15u;  // packed rgb5 15/31

const char* UpscalerName(uint32_t upscaler)
{
    return upscaler == CAELIX_UPSCALER_DLSS_RR ? "DLSS RR" : "DLSS SR";
}

const char* ReasonName(uint32_t reason)
{
    switch (reason)
    {
        case CAELIX_UPSCALER_OK: return "OK";
        case CAELIX_UPSCALER_NOT_BUILT: return "NOT_BUILT";
        case CAELIX_UPSCALER_NO_NGX: return "NO_NGX";
        case CAELIX_UPSCALER_FEATURE_UNAVAILABLE: return "FEATURE_UNAVAILABLE";
        case CAELIX_UPSCALER_DRIVER_TOO_OLD: return "DRIVER_TOO_OLD";
        case CAELIX_UPSCALER_QUALITY_UNAVAILABLE: return "QUALITY_UNAVAILABLE";
        case CAELIX_UPSCALER_BAD_ARGUMENT: return "BAD_ARGUMENT";
        default: return "unknown";
    }
}

// A storage image (STORAGE | TRANSFER_SRC, optimal tiling) and the last RGBA32F readback.
struct Target
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::vector<float> pixels;  // RGBA, row 0 = top (colour targets only)

    const float* at(uint32_t x, uint32_t y) const
    {
        return pixels.data() + (static_cast<size_t>(y) * width + x) * 4;
    }
    CaelixTargetImage desc() const
    {
        CaelixTargetImage out{};
        out.image = reinterpret_cast<uint64_t>(image);
        out.format = static_cast<uint32_t>(format);
        out.width = width;
        out.height = height;
        out.layout = static_cast<uint32_t>(layout);
        return out;
    }
};

Target CreateTarget(Gpu& gpu, uint32_t width, uint32_t height, VkFormat format)
{
    Target out;
    out.width = width;
    out.height = height;
    out.format = format;
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
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(gpu.device.physical_device.physical_device, &properties);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < properties.memoryTypeCount && type == UINT32_MAX; ++i)
    {
        if ((requirements.memoryTypeBits & (1u << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
        {
            type = i;
        }
    }
    REQUIRE(type != UINT32_MAX);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type;
    REQUIRE(vkAllocateMemory(gpu.device.device, &allocate, nullptr, &out.memory) == VK_SUCCESS);
    REQUIRE(vkBindImageMemory(gpu.device.device, out.image, out.memory, 0) == VK_SUCCESS);
    return out;
}

void DestroyTarget(Gpu& gpu, Target& target)
{
    vkDestroyImage(gpu.device.device, target.image, nullptr);
    vkFreeMemory(gpu.device.device, target.memory, nullptr);
}

// One frame: flush, render into `color` (and `depth` when given), optionally read the RGBA32F
// colour back. Returns the render call's result; nothing is read back unless it is CAELIX_OK.
CaelixResult RenderFrame(Rig& rig, CaelixRenderer* renderer, Target& color, const CaelixCamera& camera,
                         const CaelixRenderOptions& options, bool readback, Target* depth = nullptr)
{
    Readback buffer;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(color.width) * color.height * 16;
    if (readback)
    {
        REQUIRE(color.format == VK_FORMAT_R32G32B32A32_SFLOAT);
        CreateReadback(rig.gpu, bytes, buffer);
    }
    CaelixResult result = CAELIX_OK;
    const uint64_t frame = rig.frames.next++;
    SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
        CaelixFrameDesc desc{};
        desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
        desc.frame_number = frame;
        desc.safe_frame_number = frame - 1;
        REQUIRE(caelix_scene_flush(rig.scene, &desc) == CAELIX_OK);
        CaelixRenderTargets targets{};
        targets.color = color.desc();
        if (depth != nullptr)
        {
            targets.depth = depth->desc();
        }
        result = caelix_renderer_render(renderer, &desc, &targets, &camera, &options);
        if (!readback || result != CAELIX_OK)
        {
            return;
        }
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = color.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dependency);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {color.width, color.height, 1};
        vkCmdCopyImageToBuffer(cmd, color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.buffer, 1, &region);
        CopyToHostBarrier(cmd);
    });
    REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
    if (result == CAELIX_OK)
    {
        color.layout = readback ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
        if (depth != nullptr)
        {
            depth->layout = VK_IMAGE_LAYOUT_GENERAL;
        }
        if (readback)
        {
            const float* data = reinterpret_cast<const float*>(buffer.words);
            color.pixels.assign(data, data + static_cast<size_t>(color.width) * color.height * 4);
        }
    }
    if (readback)
    {
        DestroyReadback(rig.gpu, buffer);
    }
    return result;
}

void DrainBuilds(Rig& rig)
{
    int flushes = 0;
    while (Stats(rig.scene).as_builds_pending > 0)
    {
        FlushFrame(rig.gpu, rig.scene, rig.frames);
        REQUIRE(++flushes < 64);
    }
}

// The scene of caelix_nrd_test: a grey floor of 64 x 64 x 8 blocks (x, z in [0, 64), y in
// [-8, 0)) and a 16^3 grey block 8 blocks above its centre.
void UploadFloorAndBlock(Rig& rig)
{
    World world;
    Brick grey{};
    grey.fill(kGreyFloor);
    for (int32_t z = 0; z < 8; ++z)
    {
        for (int32_t x = 0; x < 8; ++x)
        {
            world.bricks[{x, -1, z}] = grey;
        }
    }
    for (int32_t z = 3; z <= 4; ++z)
    {
        for (int32_t y = 1; y <= 2; ++y)
        {
            for (int32_t x = 3; x <= 4; ++x)
            {
                world.bricks[{x, y, z}] = grey;
            }
        }
    }
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

// Camera at (32, 28, -6) looking at the block from the front and above, vertical field of view
// 60, normal-Z projection (column-major), as in caelix_nrd_test.
CaelixCamera FrontCamera(float aspect)
{
    CaelixCamera camera{};
    camera.position[0] = 32.0f;
    camera.position[1] = 28.0f;
    camera.position[2] = -6.0f;
    const double pitch = std::atan2(28.0, 42.0);
    const float s = static_cast<float>(std::sin(pitch));
    const float c = static_cast<float>(std::cos(pitch));
    camera.right[0] = 1.0f;
    camera.up[1] = c;
    camera.up[2] = s;
    camera.forward[1] = -s;
    camera.forward[2] = c;
    camera.tan_half_fov_y = static_cast<float>(std::tan(30.0 * kPi / 180.0));
    camera.aspect = aspect;
    camera.near_z = kNear;
    camera.far_z = kFar;
    float* m = camera.projection;
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = 1.0f / (camera.tan_half_fov_y * aspect);
    m[5] = 1.0f / camera.tan_half_fov_y;
    m[10] = kFar / (kFar - kNear);
    m[14] = -kFar * kNear / (kFar - kNear);
    m[11] = 1.0f;
    return camera;
}

// The lighting of caelix_nrd_test: 4 bounces, sun NEE, sky on miss, sun 1000 at 2 degrees.
CaelixRenderOptions LightOptions(uint32_t mode, uint32_t seed, uint32_t denoise)
{
    CaelixRenderOptions options{};
    options.debug_view = CAELIX_VIEW_ALBEDO;
    options.flags = CAELIX_RENDER_FLAG_SKY_ON_MISS | CAELIX_RENDER_FLAG_SUN_NEE;
    options.sun_dir[0] = 0.4f;
    options.sun_dir[1] = 1.0f;
    options.sun_dir[2] = -0.3f;
    options.miss_depth = 1.0f;
    options.miss_view_z = 1e30f;
    options.mode = mode;
    options.bounce_count = 4;
    options.denoise = denoise;
    options.exposure = 1.0f;
    options.sun_radiance[0] = options.sun_radiance[1] = options.sun_radiance[2] = 1000.0f;
    options.sun_angular_radius = static_cast<float>(2.0 * kPi / 180.0);
    const float zenith[3] = {0.2f, 0.4f, 0.9f};
    const float horizon[3] = {0.7f, 0.8f, 0.95f};
    const float ground[3] = {0.25f, 0.22f, 0.2f};
    std::memcpy(options.sky_zenith, zenith, sizeof(zenith));
    std::memcpy(options.sky_horizon, horizon, sizeof(horizon));
    std::memcpy(options.sky_ground, ground, sizeof(ground));
    options.frame_seed = seed;
    return options;
}

CaelixUpscalerQueryResult Query(CaelixDevice* device, uint32_t upscaler, uint32_t quality, uint32_t width,
                                uint32_t height)
{
    CaelixUpscalerQueryDesc desc{upscaler, quality, width, height};
    CaelixUpscalerQueryResult result{};
    REQUIRE(caelix_upscaler_query(device, &desc, &result) == CAELIX_OK);
    return result;
}

CaelixUpscalerInfo UpscalerInfo(const CaelixRenderer* renderer)
{
    CaelixUpscalerInfo info{};
    REQUIRE(caelix_renderer_get_upscaler_info(renderer, &info) == CAELIX_OK);
    return info;
}

// Mean of r, g, b over the floor in front of the block (rows 78 % .. 95 % of the image, columns
// 10 % .. 90 %), over the pixels that `mask` covers (alpha 1).
double FloorMean(const Target& image, const Target& mask)
{
    REQUIRE(image.width == mask.width);
    REQUIRE(image.height == mask.height);
    double total = 0.0;
    size_t count = 0;
    for (uint32_t y = image.height * 78 / 100; y < image.height * 95 / 100; ++y)
    {
        for (uint32_t x = image.width / 10; x < image.width * 9 / 10; ++x)
        {
            if (mask.at(x, y)[3] != 1.0f)
            {
                continue;
            }
            const float* p = image.at(x, y);
            total += static_cast<double>(p[0]) + p[1] + p[2];
            ++count;
        }
    }
    REQUIRE(count > 0);
    return total / (3.0 * static_cast<double>(count));
}

bool AnyNonZero(const Target& image)
{
    for (const float value : image.pixels)
    {
        if (value != 0.0f)
        {
            return true;
        }
    }
    return false;
}

bool AllFinite(const Target& image)
{
    for (const float value : image.pixels)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    return true;
}

// A 640 x 360 reference of the scene: the reference mode accumulated over 256 frames.
void RenderReference(Rig& rig, CaelixRenderer* renderer, Target& reference, const CaelixCamera& camera)
{
    constexpr uint32_t kFrames = 256;
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        CaelixRenderOptions options = LightOptions(CAELIX_MODE_REFERENCE, 40000 + i, 0);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        REQUIRE(RenderFrame(rig, renderer, reference, camera, options, i + 1 == kFrames) == CAELIX_OK);
    }
}

// Renders `frames` upscaled frames with Halton jitter and frame_seed = frame into `color`; the last
// one is read back. Every call must succeed.
void RenderUpscaled(Rig& rig, CaelixRenderer* renderer, Target& color, CaelixCamera camera,
                    CaelixRenderOptions options, uint32_t render_width, uint32_t frames)
{
    const uint32_t phases = caelix_upscaler_jitter_phases(options.upscaler, render_width, color.width);
    for (uint32_t i = 0; i < frames; ++i)
    {
        caelix_jitter_halton(i, phases, camera.jitter);
        options.frame_seed = i;
        REQUIRE(RenderFrame(rig, renderer, color, camera, options, i + 1 == frames) == CAELIX_OK);
    }
}

// The upscaler smoke test of one kind at Quality, 640 x 360: 12 frames, the image is not empty,
// the floor matches the reference within 25 %, the info and the timings report the upscaler.
void UpscalerSmokeTest(uint32_t upscaler)
{
    Rig rig;
    OpenRig(rig, "caelix_dlss_test", RigDesc(0, 0, 0));
    constexpr uint32_t kWidth = 640;
    constexpr uint32_t kHeight = 360;
    const CaelixUpscalerQueryResult query = Query(rig.device, upscaler, CAELIX_UPSCALE_QUALITY, kWidth, kHeight);
    if (query.supported == 0)
    {
        std::printf("%s unsupported (%s): skipped\n", UpscalerName(upscaler), ReasonName(query.reason));
        CloseRig(rig);
        SKIP("the upscaler is not available on this machine or in this build");
    }
    std::printf("%s Quality for %ux%u: render %ux%u\n", UpscalerName(upscaler), kWidth, kHeight,
                query.render_width, query.render_height);
    UploadFloorAndBlock(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    const CaelixCamera camera = FrontCamera(static_cast<float>(kWidth) / kHeight);
    Target reference = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32B32A32_SFLOAT);
    Target image = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32B32A32_SFLOAT);
    RenderReference(rig, renderer, reference, camera);
    const double reference_mean = FloorMean(reference, reference);

    // NRD stays on for SR; RR must ignore it and the delta checkerboard.
    CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 1);
    options.upscaler = upscaler;
    options.upscale_quality = CAELIX_UPSCALE_QUALITY;
    if (upscaler == CAELIX_UPSCALER_DLSS_RR)
    {
        options.flags |= CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD;
    }
    RenderUpscaled(rig, renderer, image, camera, options, query.render_width, 12);
    CHECK(AnyNonZero(image));
    CHECK(AllFinite(image));
    const double upscaled_mean = FloorMean(image, reference);
    std::printf("%s: floor mean %.5f, reference %.5f (ratio %.4f)\n", UpscalerName(upscaler), upscaled_mean,
                reference_mean, upscaled_mean / reference_mean);
    CHECK(std::abs(upscaled_mean / reference_mean - 1.0) < 0.25);

    const CaelixUpscalerInfo info = UpscalerInfo(renderer);
    CHECK(info.upscaler == upscaler);
    CHECK(info.quality == CAELIX_UPSCALE_QUALITY);
    CHECK(info.render_width == query.render_width);
    CHECK(info.render_height == query.render_height);
    CHECK(info.output_width == kWidth);
    CHECK(info.output_height == kHeight);
    CHECK(info.failed == 0);
    CHECK(info.depth_inverted == 0);
    CaelixGpuTimings timings{};
    caelix_renderer_get_gpu_timings(renderer, &timings);
    std::printf("%s timings at %ux%u -> %ux%u: trace %.4f ms, denoise %.4f ms, composite %.4f ms, upscale "
                "%.4f ms\n",
                UpscalerName(upscaler), query.render_width, query.render_height, kWidth, kHeight, timings.trace_ms,
                timings.denoise_ms, timings.composite_ms, timings.upscale_ms);
    CHECK(timings.upscale_ms > 0.0f);
    CaelixNrdInfo nrd{};
    REQUIRE(caelix_renderer_get_nrd_info(renderer, &nrd) == CAELIX_OK);
    if (upscaler == CAELIX_UPSCALER_DLSS_RR)
    {
        CHECK(nrd.width == 0);  // RR replaces NRD: no NRD instance was created
        CHECK(timings.denoise_ms == 0.0f);
    }
    else
    {
        CHECK(nrd.width == query.render_width);  // SR keeps NRD, at the render size
        CHECK(timings.denoise_ms > 0.0f);
    }

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, reference);
    DestroyTarget(rig.gpu, image);
    CloseRig(rig);
}

}  // namespace

TEST_CASE("Upscaler query: SR and RR at 1920x1080 for every quality, argument checks")
{
    Rig rig;
    OpenRig(rig, "caelix_dlss_test", RigDesc(0, 0, 0));
    const char* const quality_names[] = {"DLAA", "Quality", "Balanced", "Performance", "Ultra Performance"};
    for (const uint32_t upscaler : {CAELIX_UPSCALER_DLSS_SR, CAELIX_UPSCALER_DLSS_RR})
    {
        for (uint32_t quality = CAELIX_UPSCALE_DLAA; quality <= CAELIX_UPSCALE_ULTRA_PERFORMANCE; ++quality)
        {
            const CaelixUpscalerQueryResult result = Query(rig.device, upscaler, quality, 1920, 1080);
            if (result.supported == 0)
            {
                std::printf("%s %s: unsupported (%s, minimum driver %u.%u)\n", UpscalerName(upscaler),
                            quality_names[quality], ReasonName(result.reason), result.min_driver_major,
                            result.min_driver_minor);
                CHECK(result.reason != CAELIX_UPSCALER_OK);
                CHECK(result.render_width == 0);
                continue;
            }
            std::printf("%s %s: render %ux%u (dynamic %ux%u .. %ux%u)\n", UpscalerName(upscaler),
                        quality_names[quality], result.render_width, result.render_height, result.render_min_width,
                        result.render_min_height, result.render_max_width, result.render_max_height);
            CHECK(result.reason == CAELIX_UPSCALER_OK);
            CHECK(result.render_width > 0);
            CHECK(result.render_height > 0);
            CHECK(result.render_width <= 1920);
            CHECK(result.render_height <= 1080);
            if (quality == CAELIX_UPSCALE_DLAA)
            {
                CHECK(result.render_width == 1920);
                CHECK(result.render_height == 1080);
            }
        }
    }
    // No upscaler: always supported at the output size.
    const CaelixUpscalerQueryResult none = Query(rig.device, CAELIX_UPSCALER_NONE, CAELIX_UPSCALE_QUALITY, 800, 600);
    CHECK(none.supported == 1);
    CHECK(none.render_width == 800);
    CHECK(none.render_height == 600);
    // An output below 64 x 32 is unsupported (BAD_ARGUMENT), unless the build has no upscaler.
    const CaelixUpscalerQueryResult tiny = Query(rig.device, CAELIX_UPSCALER_DLSS_SR, CAELIX_UPSCALE_QUALITY, 48, 24);
    CHECK(tiny.supported == 0);
    CHECK((tiny.reason == CAELIX_UPSCALER_BAD_ARGUMENT || tiny.reason == CAELIX_UPSCALER_NOT_BUILT));
    // Unknown enums and null arguments are errors.
    ExpectLibraryErrors(3, [&] {
        CaelixUpscalerQueryDesc desc{3, CAELIX_UPSCALE_QUALITY, 1920, 1080};
        CaelixUpscalerQueryResult result{};
        CHECK(caelix_upscaler_query(rig.device, &desc, &result) == CAELIX_ERROR_INVALID_ARGUMENT);
        desc = CaelixUpscalerQueryDesc{CAELIX_UPSCALER_DLSS_SR, 5, 1920, 1080};
        CHECK(caelix_upscaler_query(rig.device, &desc, &result) == CAELIX_ERROR_INVALID_ARGUMENT);
        CHECK(caelix_upscaler_query(rig.device, &desc, nullptr) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
    // The extension lists the test device enabled (informative).
    uint32_t instance_count = 0;
    const char* const* instance_names = nullptr;
    caelix_upscaler_instance_extensions(&instance_count, &instance_names);
    for (uint32_t i = 0; i < instance_count; ++i)
    {
        std::printf("upscaler instance extension: %s\n", instance_names[i]);
    }
    uint32_t device_count = 0;
    const char* const* device_names = nullptr;
    REQUIRE(caelix_upscaler_device_extensions(reinterpret_cast<uint64_t>(rig.gpu.instance.instance),
                                              reinterpret_cast<uint64_t>(rig.gpu.device.physical_device.physical_device),
                                              &device_count, &device_names) == CAELIX_OK);
    for (uint32_t i = 0; i < device_count; ++i)
    {
        std::printf("upscaler device extension: %s\n", device_names[i]);
    }
    CloseRig(rig);
}

TEST_CASE("Jitter helpers: Halton (2, 3) points and phase counts")
{
    float jitter[2] = {};
    caelix_jitter_halton(0, 8, jitter);
    CHECK(jitter[0] == 0.0f);  // Halton point 1: (1/2, 1/3) - 0.5
    CHECK(std::abs(jitter[1] - (1.0f / 3.0f - 0.5f)) < 1e-6f);
    caelix_jitter_halton(1, 8, jitter);
    CHECK(std::abs(jitter[0] - (0.25f - 0.5f)) < 1e-6f);  // point 2: (1/4, 2/3)
    CHECK(std::abs(jitter[1] - (2.0f / 3.0f - 0.5f)) < 1e-6f);
    caelix_jitter_halton(2, 8, jitter);
    CHECK(std::abs(jitter[0] - (0.75f - 0.5f)) < 1e-6f);  // point 3: (3/4, 1/9)
    CHECK(std::abs(jitter[1] - (1.0f / 9.0f - 0.5f)) < 1e-6f);
    // Every value lies in [-0.5, 0.5) and the sequence repeats after `phases` points.
    constexpr uint32_t kPhases = 32;
    for (uint32_t i = 0; i < 3 * kPhases; ++i)
    {
        float a[2] = {};
        float b[2] = {};
        caelix_jitter_halton(i, kPhases, a);
        caelix_jitter_halton(i + kPhases, kPhases, b);
        CHECK(a[0] >= -0.5f);
        CHECK(a[0] < 0.5f);
        CHECK(a[1] >= -0.5f);
        CHECK(a[1] < 0.5f);
        CHECK(a[0] == b[0]);
        CHECK(a[1] == b[1]);
    }
    caelix_jitter_halton(0, 0, nullptr);  // ignored
    CHECK(caelix_upscaler_jitter_phases(CAELIX_UPSCALER_DLSS_SR, 960, 1920) == 32);
    CHECK(caelix_upscaler_jitter_phases(CAELIX_UPSCALER_DLSS_RR, 1280, 1920) == 32);
    CHECK(caelix_upscaler_jitter_phases(CAELIX_UPSCALER_DLSS_SR, 1920, 1920) == 8);
    CHECK(caelix_upscaler_jitter_phases(CAELIX_UPSCALER_DLSS_SR, 1280, 1920) == 24);
    CHECK(caelix_upscaler_jitter_phases(CAELIX_UPSCALER_DLSS_SR, 640, 1920) == 72);
    CHECK(caelix_upscaler_jitter_phases(CAELIX_UPSCALER_DLSS_RR, 1920, 1920) == 32);
    CHECK(caelix_upscaler_jitter_phases(CAELIX_UPSCALER_DLSS_SR, 0, 1920) == 8);
}

TEST_CASE("DLSS Super Resolution: Quality 640x360, 12 frames against an accumulated reference")
{
    UpscalerSmokeTest(CAELIX_UPSCALER_DLSS_SR);
}

TEST_CASE("DLSS Ray Reconstruction: Quality 640x360, 12 frames; denoise and checkerboard ignored")
{
    UpscalerSmokeTest(CAELIX_UPSCALER_DLSS_RR);
}

TEST_CASE("Upscaler arguments: target sizes, quality change, switching the upscaler off")
{
    Rig rig;
    OpenRig(rig, "caelix_dlss_test", RigDesc(0, 0, 0));
    constexpr uint32_t kWidth = 640;
    constexpr uint32_t kHeight = 360;
    const CaelixUpscalerQueryResult quality =
        Query(rig.device, CAELIX_UPSCALER_DLSS_SR, CAELIX_UPSCALE_QUALITY, kWidth, kHeight);
    const CaelixUpscalerQueryResult performance =
        Query(rig.device, CAELIX_UPSCALER_DLSS_SR, CAELIX_UPSCALE_PERFORMANCE, kWidth, kHeight);
    if (quality.supported == 0 || performance.supported == 0)
    {
        // An unavailable upscaler: the render call returns CAELIX_ERROR_UNSUPPORTED (one error).
        CaelixRenderer* renderer = nullptr;
        REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
        Target color = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32B32A32_SFLOAT);
        CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 1);
        options.upscaler = CAELIX_UPSCALER_DLSS_SR;
        options.upscale_quality = CAELIX_UPSCALE_QUALITY;
        ExpectLibraryErrors(1, [&] {
            CHECK(RenderFrame(rig, renderer, color, FrontCamera(static_cast<float>(kWidth) / kHeight), options,
                              false) == CAELIX_ERROR_UNSUPPORTED);
        });
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_destroy(renderer);
        DestroyTarget(rig.gpu, color);
        std::printf("DLSS SR unsupported (%s): the render call returned CAELIX_ERROR_UNSUPPORTED; skipped\n",
                    ReasonName(quality.reason));
        CloseRig(rig);
        SKIP("DLSS SR is not available on this machine or in this build");
    }
    UploadFloorAndBlock(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    const CaelixCamera camera = FrontCamera(static_cast<float>(kWidth) / kHeight);
    Target color = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32B32A32_SFLOAT);
    CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 1);
    options.upscaler = CAELIX_UPSCALER_DLSS_SR;
    options.upscale_quality = CAELIX_UPSCALE_QUALITY;

    // A geometry target at the output size (it must have the render size).
    {
        Target depth = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32_SFLOAT);
        ExpectLibraryErrors(1, [&] {
            CHECK(RenderFrame(rig, renderer, color, camera, options, false, &depth) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        // At the render size it is accepted.
        Target render_depth = CreateTarget(rig.gpu, quality.render_width, quality.render_height, VK_FORMAT_R32_SFLOAT);
        CHECK(RenderFrame(rig, renderer, color, camera, options, false, &render_depth) == CAELIX_OK);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroyTarget(rig.gpu, depth);
        DestroyTarget(rig.gpu, render_depth);
    }
    // An output below 64 x 32.
    {
        Target tiny = CreateTarget(rig.gpu, 48, 24, VK_FORMAT_R32G32B32A32_SFLOAT);
        ExpectLibraryErrors(1, [&] {
            CHECK(RenderFrame(rig, renderer, tiny, camera, options, false) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        DestroyTarget(rig.gpu, tiny);
    }
    // An unknown upscaler and an unknown quality.
    ExpectLibraryErrors(2, [&] {
        CaelixRenderOptions bad = options;
        bad.upscaler = 7;
        CHECK(RenderFrame(rig, renderer, color, camera, bad, false) == CAELIX_ERROR_INVALID_ARGUMENT);
        bad = options;
        bad.upscale_quality = 9;
        CHECK(RenderFrame(rig, renderer, color, camera, bad, false) == CAELIX_ERROR_INVALID_ARGUMENT);
    });

    // Quality, then Performance: the info follows the new render size.
    RenderUpscaled(rig, renderer, color, camera, options, quality.render_width, 3);
    CaelixUpscalerInfo info = UpscalerInfo(renderer);
    CHECK(info.render_width == quality.render_width);
    CHECK(info.render_height == quality.render_height);
    options.upscale_quality = CAELIX_UPSCALE_PERFORMANCE;
    RenderUpscaled(rig, renderer, color, camera, options, performance.render_width, 3);
    info = UpscalerInfo(renderer);
    std::printf("quality change: Quality %ux%u -> Performance %ux%u (info %ux%u, quality %u, failed %u)\n",
                quality.render_width, quality.render_height, performance.render_width, performance.render_height,
                info.render_width, info.render_height, info.quality, info.failed);
    CHECK(info.quality == CAELIX_UPSCALE_PERFORMANCE);
    CHECK(info.render_width == performance.render_width);
    CHECK(info.render_height == performance.render_height);
    CHECK(info.failed == 0);
    CHECK(AnyNonZero(color));

    // The upscaler off again: the renderer traces at the colour size as before.
    options.upscaler = CAELIX_UPSCALER_NONE;
    REQUIRE(RenderFrame(rig, renderer, color, camera, options, true) == CAELIX_OK);
    CHECK(AnyNonZero(color));
    CaelixNrdInfo nrd{};
    REQUIRE(caelix_renderer_get_nrd_info(renderer, &nrd) == CAELIX_OK);
    CHECK(nrd.width == kWidth);
    CHECK(nrd.height == kHeight);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("Upscaler cost: DLSS SR and RR at 1920x1080 Quality")
{
    Rig rig;
    OpenRig(rig, "caelix_dlss_test", RigDesc(0, 0, 0));
    constexpr uint32_t kWidth = 1920;
    constexpr uint32_t kHeight = 1080;
    const CaelixUpscalerQueryResult sr = Query(rig.device, CAELIX_UPSCALER_DLSS_SR, CAELIX_UPSCALE_QUALITY, kWidth, kHeight);
    const CaelixUpscalerQueryResult rr = Query(rig.device, CAELIX_UPSCALER_DLSS_RR, CAELIX_UPSCALE_QUALITY, kWidth, kHeight);
    if (sr.supported == 0 && rr.supported == 0)
    {
        std::printf("no DLSS upscaler available: skipped\n");
        CloseRig(rig);
        SKIP("no DLSS upscaler is available on this machine or in this build");
    }
    UploadFloorAndBlock(rig);
    const CaelixCamera camera = FrontCamera(static_cast<float>(kWidth) / kHeight);
    Target color = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32B32A32_SFLOAT);
    for (const uint32_t upscaler : {CAELIX_UPSCALER_DLSS_SR, CAELIX_UPSCALER_DLSS_RR})
    {
        const CaelixUpscalerQueryResult& query = upscaler == CAELIX_UPSCALER_DLSS_SR ? sr : rr;
        if (query.supported == 0)
        {
            continue;
        }
        CaelixRenderer* renderer = nullptr;
        REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
        CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 1);
        options.upscaler = upscaler;
        options.upscale_quality = CAELIX_UPSCALE_QUALITY;
        // 24 frames; the timings of the newest resolved frames, averaged over the last 8.
        const uint32_t phases = caelix_upscaler_jitter_phases(upscaler, query.render_width, kWidth);
        double upscale_sum = 0.0;
        double trace_sum = 0.0;
        double denoise_sum = 0.0;
        uint32_t samples = 0;
        CaelixCamera jittered = camera;
        for (uint32_t i = 0; i < 24; ++i)
        {
            caelix_jitter_halton(i, phases, jittered.jitter);
            options.frame_seed = i;
            REQUIRE(RenderFrame(rig, renderer, color, jittered, options, false) == CAELIX_OK);
            if (i >= 16)
            {
                CaelixGpuTimings timings{};
                caelix_renderer_get_gpu_timings(renderer, &timings);
                upscale_sum += timings.upscale_ms;
                trace_sum += timings.trace_ms;
                denoise_sum += timings.denoise_ms;
                ++samples;
            }
        }
        std::printf("%s Quality %ux%u -> %ux%u: upscale %.3f ms (evaluate + encode), trace %.3f ms, denoise %.3f "
                    "ms (mean of %u frames)\n",
                    UpscalerName(upscaler), query.render_width, query.render_height, kWidth, kHeight,
                    upscale_sum / samples, trace_sum / samples, denoise_sum / samples, samples);
        CHECK(upscale_sum > 0.0);
        CHECK(UpscalerInfo(renderer).failed == 0);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        caelix_renderer_destroy(renderer);
    }
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}
