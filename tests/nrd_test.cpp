// NRD denoising (CaelixRenderOptions::denoise in CAELIX_MODE_PATH_TRACE): the
// renderer's NrdExecutor is created, resized and run through the public C ABI, and
// the denoised image of a sunlit floor with a floating block is compared with a
// 1024-frame reference of the same kernel. Uses the public C ABI only; any Vulkan
// validation message fails the test (CloseRig).
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

// An RGBA32F colour target (STORAGE | TRANSFER_SRC) and its last readback.
struct ColorTarget
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::vector<float> pixels;  // RGBA, row 0 = top

    const float* at(uint32_t x, uint32_t y) const
    {
        return pixels.data() + (static_cast<size_t>(y) * width + x) * 4;
    }
};

ColorTarget CreateColorTarget(Gpu& gpu, uint32_t width, uint32_t height)
{
    ColorTarget out;
    out.width = width;
    out.height = height;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
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

void DestroyColorTarget(Gpu& gpu, ColorTarget& target)
{
    vkDestroyImage(gpu.device.device, target.image, nullptr);
    vkFreeMemory(gpu.device.device, target.memory, nullptr);
}

// One frame: flush, render into `target` (the render call must succeed), optionally
// read it back.
void Render(Rig& rig, CaelixRenderer* renderer, ColorTarget& target, const CaelixCamera& camera,
            const CaelixRenderOptions& options, bool readback)
{
    Readback buffer;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(target.width) * target.height * 16;
    if (readback)
    {
        CreateReadback(rig.gpu, bytes, buffer);
    }
    const uint64_t frame = rig.frames.next++;
    SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
        CaelixFrameDesc desc{};
        desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
        desc.frame_number = frame;
        desc.safe_frame_number = frame - 1;
        REQUIRE(caelix_scene_flush(rig.scene, &desc) == CAELIX_OK);
        CaelixRenderTargets targets{};
        targets.color.image = reinterpret_cast<uint64_t>(target.image);
        targets.color.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        targets.color.width = target.width;
        targets.color.height = target.height;
        targets.color.layout = static_cast<uint32_t>(target.layout);
        REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &options) == CAELIX_OK);
        if (!readback)
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
        barrier.image = target.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dependency);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {target.width, target.height, 1};
        vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.buffer,
                               1, &region);
        CopyToHostBarrier(cmd);
    });
    REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
    if (!readback)
    {
        target.layout = VK_IMAGE_LAYOUT_GENERAL;
        return;
    }
    target.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    const float* data = reinterpret_cast<const float*>(buffer.words);
    target.pixels.assign(data, data + static_cast<size_t>(target.width) * target.height * 4);
    DestroyReadback(rig.gpu, buffer);
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

// The sun NEE test scene of caelix_trace_test: a grey floor of 64 x 64 x 8 blocks
// (x, z in [0, 64), y in [-8, 0)) and a 16^3 grey block 8 blocks above its centre
// (x, z in [24, 40), y in [8, 24)).
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

// Camera at (32, 28, -6) looking at the block from the front and above (yaw 0, pitch
// 33.7 degrees down), vertical field of view 60, normal-Z projection (column-major).
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

// Path-traced options: 4 bounces, sun NEE, the sky of caelix_trace_test, a sun of
// radiance 1000 and radius 2 degrees from the upper right front, exposure 1.
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

CaelixNrdInfo NrdInfo(const CaelixRenderer* renderer)
{
    CaelixNrdInfo info{};
    REQUIRE(caelix_renderer_get_nrd_info(renderer, &info) == CAELIX_OK);
    return info;
}

struct Error
{
    double rmse = 0.0;
    double psnr = 0.0;  // peak 1 (linear radiance)
    double mean = 0.0;  // mean of r, g, b over the hit pixels
};

// RMSE of `image` against `reference` over the reference's hit pixels (alpha 1),
// all three channels.
Error Compare(const ColorTarget& image, const ColorTarget& reference)
{
    double sum = 0.0;
    double total = 0.0;
    size_t count = 0;
    for (uint32_t y = 0; y < reference.height; ++y)
    {
        for (uint32_t x = 0; x < reference.width; ++x)
        {
            const float* r = reference.at(x, y);
            if (r[3] != 1.0f)
            {
                continue;
            }
            const float* p = image.at(x, y);
            for (int c = 0; c < 3; ++c)
            {
                const double d = static_cast<double>(p[c]) - r[c];
                sum += d * d;
                total += p[c];
            }
            ++count;
        }
    }
    REQUIRE(count > 0);
    Error out;
    out.rmse = std::sqrt(sum / (3.0 * static_cast<double>(count)));
    out.psnr = 20.0 * std::log10(1.0 / std::max(out.rmse, 1e-12));
    out.mean = total / (3.0 * static_cast<double>(count));
    return out;
}

double HitMean(const ColorTarget& image)
{
    double total = 0.0;
    size_t count = 0;
    for (uint32_t y = 0; y < image.height; ++y)
    {
        for (uint32_t x = 0; x < image.width; ++x)
        {
            const float* p = image.at(x, y);
            if (p[3] == 1.0f)
            {
                total += p[0] + p[1] + p[2];
                ++count;
            }
        }
    }
    REQUIRE(count > 0);
    return total / (3.0 * static_cast<double>(count));
}

// Mean |a - b| over the hit pixels of `reference`, all three channels.
double MeanAbsDifference(const std::vector<float>& a, const std::vector<float>& b,
                         const ColorTarget& reference)
{
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < reference.pixels.size(); i += 4)
    {
        if (reference.pixels[i + 3] != 1.0f)
        {
            continue;
        }
        for (size_t c = 0; c < 3; ++c)
        {
            sum += std::abs(static_cast<double>(a[i + c]) - b[i + c]);
        }
        ++count;
    }
    REQUIRE(count > 0);
    return sum / (3.0 * static_cast<double>(count));
}

}  // namespace

TEST_CASE("NRD executor: instance, pools and pipelines; recreated on a size change")
{
    Rig rig;
    OpenRig(rig, "caelix_nrd_test", RigDesc(0, 0, 0));
    UploadFloorAndBlock(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);

    // No instance before the first denoised render; the library version is known.
    CaelixNrdInfo info = NrdInfo(renderer);
    std::printf("NRD library %u.%u.%u\n", info.version_major, info.version_minor, info.version_build);
    CHECK(info.version_major == 4);
    CHECK(info.version_minor == 18);
    CHECK(info.width == 0);
    CHECK(info.failed == 0);

    ColorTarget small = CreateColorTarget(rig.gpu, 320, 240);
    for (uint32_t i = 0; i < 3; ++i)
    {
        Render(rig, renderer, small, FrontCamera(320.0f / 240.0f),
               LightOptions(CAELIX_MODE_PATH_TRACE, 100 + i, 1), i == 2);
    }
    info = NrdInfo(renderer);
    std::printf("NRD instance %ux%u: %u pipelines, %u permanent textures (%.2f MiB), %u transient "
                "textures (%.2f MiB), %u dispatches\n",
                info.width, info.height, info.pipeline_count, info.permanent_texture_count,
                static_cast<double>(info.permanent_bytes) / (1024.0 * 1024.0),
                info.transient_texture_count,
                static_cast<double>(info.transient_bytes) / (1024.0 * 1024.0), info.dispatch_count);
    CHECK(info.width == 320);
    CHECK(info.height == 240);
    CHECK(info.pipeline_count == 14);
    CHECK(info.permanent_texture_count == 13);
    CHECK(info.transient_texture_count == 8);
    CHECK(info.permanent_bytes > 0);
    CHECK(info.transient_bytes > 0);
    CHECK(info.dispatch_count > 0);
    CHECK(info.failed == 0);
    CaelixGpuTimings timings{};
    caelix_renderer_get_gpu_timings(renderer, &timings);
    std::printf("timings at 320x240: trace %.4f ms, denoise %.4f ms, composite %.4f ms\n",
                timings.trace_ms, timings.denoise_ms, timings.composite_ms);
    CHECK(timings.denoise_ms > 0.0f);

    // A second renderer over the same scene: its NRD instance borrows the device's shared NRD
    // kernels (the first instance built them; this one's desc must match).
    {
        CaelixRenderer* second = nullptr;
        REQUIRE(caelix_renderer_create(rig.device, rig.scene, &second) == CAELIX_OK);
        ColorTarget second_target = CreateColorTarget(rig.gpu, 320, 240);
        Render(rig, second, second_target, FrontCamera(320.0f / 240.0f),
               LightOptions(CAELIX_MODE_PATH_TRACE, 150, 1), true);
        const CaelixNrdInfo second_info = NrdInfo(second);
        std::printf("second renderer: NRD instance %ux%u, %u pipelines, failed %u, %u dispatches\n",
                    second_info.width, second_info.height, second_info.pipeline_count, second_info.failed,
                    second_info.dispatch_count);
        CHECK(second_info.pipeline_count == 14);
        CHECK(second_info.failed == 0);
        CHECK(second_info.width == 320);
        CHECK(second_info.dispatch_count > 0);
        CHECK(HitMean(second_target) > 0.0);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(second);
        caelix_renderer_destroy(second);
        DestroyColorTarget(rig.gpu, second_target);
    }

    // Another size: the instance is replaced (the old one is destroyed once its frame is safe).
    ColorTarget other = CreateColorTarget(rig.gpu, 256, 192);
    for (uint32_t i = 0; i < 2; ++i)
    {
        Render(rig, renderer, other, FrontCamera(256.0f / 192.0f),
               LightOptions(CAELIX_MODE_PATH_TRACE, 200 + i, 1), false);
    }
    info = NrdInfo(renderer);
    CHECK(info.width == 256);
    CHECK(info.height == 192);
    CHECK(info.pipeline_count == 14);
    // Back to the first size: recreated once more.
    Render(rig, renderer, small, FrontCamera(320.0f / 240.0f),
           LightOptions(CAELIX_MODE_PATH_TRACE, 300, 1), true);
    info = NrdInfo(renderer);
    CHECK(info.width == 320);
    CHECK(info.failed == 0);

    // A null argument is rejected.
    ExpectLibraryErrors(1, [&] { CHECK(caelix_renderer_get_nrd_info(renderer, nullptr) ==
                                       CAELIX_ERROR_INVALID_ARGUMENT); });

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyColorTarget(rig.gpu, small);
    DestroyColorTarget(rig.gpu, other);
    CloseRig(rig);
}

TEST_CASE("NRD denoising: quality against a 1024-frame reference, reset, debug views")
{
    Rig rig;
    OpenRig(rig, "caelix_nrd_test", RigDesc(0, 0, 0));
    UploadFloorAndBlock(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    // The host's resolution: NRD's blur radii are in pixels (maxBlurRadius 30,
    // diffusePrepassBlurRadius 30), so at 320x240 the same radii cover four times the
    // screen fraction, and the blur of the sun's hard shadow dominates the error (measured
    // 2026-10-14: RMSE ratio raw / NRD 2.1 at 320x240, 3.1 at 640x480, 3.9 at 1280x720).
    constexpr uint32_t kWidth = 1280;
    constexpr uint32_t kHeight = 720;
    const CaelixCamera camera = FrontCamera(static_cast<float>(kWidth) / kHeight);
    ColorTarget reference = CreateColorTarget(rig.gpu, kWidth, kHeight);
    ColorTarget image = CreateColorTarget(rig.gpu, kWidth, kHeight);

    // Reference: the same kernel accumulated over 1024 frames.
    constexpr uint32_t kReferenceFrames = 1024;
    for (uint32_t i = 0; i < kReferenceFrames; ++i)
    {
        CaelixRenderOptions options = LightOptions(CAELIX_MODE_REFERENCE, 10000 + i, 0);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, reference, camera, options, i + 1 == kReferenceFrames);
    }
    const double reference_mean = HitMean(reference);

    // One raw 1 spp frame.
    Render(rig, renderer, image, camera, LightOptions(CAELIX_MODE_PATH_TRACE, 77, 0), true);
    const Error raw = Compare(image, reference);

    // 64 denoised frames with a static camera and a new seed each; the last two are read
    // back (their difference is the frame-to-frame change of a converged history).
    constexpr uint32_t kDenoisedFrames = 64;
    std::vector<float> before_last;
    for (uint32_t i = 0; i < kDenoisedFrames; ++i)
    {
        Render(rig, renderer, image, camera, LightOptions(CAELIX_MODE_PATH_TRACE, 500 + i, 1),
               i + 2 >= kDenoisedFrames);
        if (i + 2 == kDenoisedFrames)
        {
            before_last = image.pixels;
        }
    }
    const Error denoised = Compare(image, reference);
    const std::vector<float> converged = image.pixels;
    const double steady_change = MeanAbsDifference(converged, before_last, reference);
    std::printf("reference (%u frames): mean %.5f over the hit pixels\n", kReferenceFrames,
                reference_mean);
    std::printf("raw 1 spp:           RMSE %.5f, PSNR %.2f dB, mean %.5f (ratio %.4f)\n", raw.rmse,
                raw.psnr, raw.mean, raw.mean / reference_mean);
    std::printf("NRD after %u frames: RMSE %.5f, PSNR %.2f dB, mean %.5f (ratio %.4f); RMSE ratio "
                "raw / NRD %.2f\n",
                kDenoisedFrames, denoised.rmse, denoised.psnr, denoised.mean,
                denoised.mean / reference_mean, raw.rmse / denoised.rmse);
    CHECK(denoised.rmse < raw.rmse / 3.0);
    CHECK(std::abs(denoised.mean / reference_mean - 1.0) < 0.05);

    CaelixGpuTimings timings{};
    caelix_renderer_get_gpu_timings(renderer, &timings);
    std::printf("timings at %ux%u: trace %.4f ms, denoise %.4f ms, composite %.4f ms\n", kWidth,
                kHeight, timings.trace_ms, timings.denoise_ms, timings.composite_ms);
    CHECK(timings.denoise_ms > 0.0f);

    // Reset: the first frame after reset_accumulation is denoised from one frame only, so it
    // changes much more against the converged frame than one converged frame against the
    // next; then the history converges again.
    {
        CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 900, 1);
        options.reset_accumulation = 1;
        Render(rig, renderer, image, camera, options, true);
    }
    const Error after_reset = Compare(image, reference);
    const double reset_change = MeanAbsDifference(image.pixels, converged, reference);
    for (uint32_t i = 0; i < kDenoisedFrames; ++i)
    {
        Render(rig, renderer, image, camera, LightOptions(CAELIX_MODE_PATH_TRACE, 1000 + i, 1),
               i + 1 == kDenoisedFrames);
    }
    const Error reconverged = Compare(image, reference);
    std::printf("reset: mean |change| against the converged frame %.6f (converged frame to frame "
                "%.6f, ratio %.1f); first frame RMSE %.5f (PSNR %.2f dB), after %u more frames "
                "RMSE %.5f (PSNR %.2f dB)\n",
                reset_change, steady_change, reset_change / steady_change, after_reset.rmse,
                after_reset.psnr, kDenoisedFrames, reconverged.rmse, reconverged.psnr);
    CHECK(reset_change > 2.0 * steady_change);
    CHECK(reconverged.rmse < raw.rmse / 3.0);
    CHECK(std::abs(reconverged.rmse / denoised.rmse - 1.0) < 0.1);

    // Debug views: NRD's validation layer and the denoised diffuse radiance. Both render
    // without validation messages; the diffuse view is the colour without the albedo, so it is
    // brighter than the colour on the grey (albedo 0.48) scene.
    {
        CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 2000, 1);
        options.debug_view = CAELIX_VIEW_NRD_VALIDATION;
        Render(rig, renderer, image, camera, options, true);
        const double layer_difference = MeanAbsDifference(image.pixels, converged, reference);
        std::printf("validation view: mean |difference| to the colour %.4f\n", layer_difference);
        CHECK(layer_difference > 0.05);
        options.debug_view = CAELIX_VIEW_NRD_DIFF;
        options.frame_seed = 2001;
        Render(rig, renderer, image, camera, options, true);
        const double diffuse_mean = HitMean(image);
        std::printf("diffuse view: mean %.5f (colour %.5f)\n", diffuse_mean, reconverged.mean);
        CHECK(diffuse_mean > reconverged.mean * 1.5);
    }
    CHECK(NrdInfo(renderer).failed == 0);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyColorTarget(rig.gpu, reference);
    DestroyColorTarget(rig.gpu, image);
    CloseRig(rig);
}
