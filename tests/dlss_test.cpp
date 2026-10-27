// The DLSS upscalers (CaelixRenderOptions::upscaler, docs/DLSS.md) through the public C ABI:
// the availability query, the jitter helpers, DLSS Super Resolution and Ray Reconstruction on the
// sunlit floor + floating block scene of caelix_nrd_test (smoke tests against an accumulated
// reference), argument checks, quality changes, the voxel id target at the output size (the dedicated
// unjittered pass against a render without an upscaler), the host background under SR and the depth
// target at the output size, and the measured upscale cost at 1920x1080. Any
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

// One frame: flush, render into `color` (and `view_z` when given), optionally read the RGBA32F
// colour back. Returns the render call's result; nothing is read back unless it is CAELIX_OK.
CaelixResult RenderFrame(Rig& rig, CaelixRenderer* renderer, Target& color, const CaelixCamera& camera,
                         const CaelixRenderOptions& options, bool readback, Target* view_z = nullptr)
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
        if (view_z != nullptr)
        {
            targets.view_z = view_z->desc();
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
        if (view_z != nullptr)
        {
            view_z->layout = VK_IMAGE_LAYOUT_GENERAL;
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

// One frame with a colour target (not read back) and a voxel id target (R32G32_UINT; read back into
// `ids` as two words per pixel when the call succeeds and `ids` is not null). Returns the render
// call's result.
CaelixResult RenderVoxelIds(Rig& rig, CaelixRenderer* renderer, Target& color, Target& id_target,
                            const CaelixCamera& camera, const CaelixRenderOptions& options,
                            std::vector<uint32_t>* ids)
{
    REQUIRE(id_target.format == VK_FORMAT_R32G32_UINT);
    Readback buffer;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(id_target.width) * id_target.height * 8;
    if (ids != nullptr)
    {
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
        targets.voxel_id = id_target.desc();
        result = caelix_renderer_render(renderer, &desc, &targets, &camera, &options);
        if (ids == nullptr || result != CAELIX_OK)
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
        barrier.image = id_target.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dependency);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {id_target.width, id_target.height, 1};
        vkCmdCopyImageToBuffer(cmd, id_target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.buffer, 1,
                               &region);
        CopyToHostBarrier(cmd);
    });
    REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
    if (result == CAELIX_OK)
    {
        color.layout = VK_IMAGE_LAYOUT_GENERAL;
        id_target.layout = ids != nullptr ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
        if (ids != nullptr)
        {
            const uint32_t* data = reinterpret_cast<const uint32_t*>(buffer.words);
            ids->assign(data, data + static_cast<size_t>(id_target.width) * id_target.height * 2);
        }
    }
    if (ids != nullptr)
    {
        DestroyReadback(rig.gpu, buffer);
    }
    return result;
}

// A target of RenderTargets: left in GENERAL, or read back into `bytes` (raw texels, row 0 = top,
// `bytes_per_pixel` each) and left in TRANSFER_SRC_OPTIMAL.
struct Written
{
    Target* target = nullptr;
    std::vector<uint8_t>* bytes = nullptr;
    uint32_t bytes_per_pixel = 0;
};

// One frame into `targets` (the caller builds them from Target::desc() and its background); `written`
// lists the Target objects among them. Returns the render call's result; nothing is read back unless it
// is CAELIX_OK.
CaelixResult RenderTargets(Rig& rig, CaelixRenderer* renderer, const CaelixRenderTargets& targets,
                           const std::vector<Written>& written, const CaelixCamera& camera,
                           const CaelixRenderOptions& options)
{
    std::vector<Readback> buffers(written.size());
    for (size_t i = 0; i < written.size(); ++i)
    {
        if (written[i].bytes != nullptr)
        {
            const Target& target = *written[i].target;
            CreateReadback(rig.gpu, static_cast<VkDeviceSize>(target.width) * target.height * written[i].bytes_per_pixel,
                           buffers[i]);
        }
    }
    CaelixResult result = CAELIX_OK;
    const uint64_t frame = rig.frames.next++;
    SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
        CaelixFrameDesc desc{};
        desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
        desc.frame_number = frame;
        desc.safe_frame_number = frame - 1;
        REQUIRE(caelix_scene_flush(rig.scene, &desc) == CAELIX_OK);
        result = caelix_renderer_render(renderer, &desc, &targets, &camera, &options);
        if (result != CAELIX_OK)
        {
            return;
        }
        for (size_t i = 0; i < written.size(); ++i)
        {
            if (written[i].bytes == nullptr)
            {
                continue;
            }
            const Target& target = *written[i].target;
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
            vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffers[i].buffer, 1,
                                   &region);
        }
        CopyToHostBarrier(cmd);
    });
    REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
    for (size_t i = 0; i < written.size(); ++i)
    {
        Target& target = *written[i].target;
        if (result == CAELIX_OK)
        {
            target.layout = written[i].bytes != nullptr ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
            if (written[i].bytes != nullptr)
            {
                const uint8_t* data = reinterpret_cast<const uint8_t*>(buffers[i].words);
                written[i].bytes->assign(data, data + static_cast<size_t>(target.width) * target.height *
                                                          written[i].bytes_per_pixel);
            }
        }
        if (written[i].bytes != nullptr)
        {
            DestroyReadback(rig.gpu, buffers[i]);
        }
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

    // A geometry target at the output size (it must have the render size; the depth target, which may
    // have either, is checked in the output-size depth case).
    {
        Target view_z = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32_SFLOAT);
        ExpectLibraryErrors(1, [&] {
            CHECK(RenderFrame(rig, renderer, color, camera, options, false, &view_z) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        // At the render size it is accepted.
        Target render_view_z = CreateTarget(rig.gpu, quality.render_width, quality.render_height, VK_FORMAT_R32_SFLOAT);
        CHECK(RenderFrame(rig, renderer, color, camera, options, false, &render_view_z) == CAELIX_OK);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroyTarget(rig.gpu, view_z);
        DestroyTarget(rig.gpu, render_view_z);
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

TEST_CASE("voxel id with an upscaler: output size, unjittered dedicated pass")
{
    Rig rig;
    OpenRig(rig, "caelix_dlss_test", RigDesc(0, 0, 0));
    constexpr uint32_t kWidth = 640;
    constexpr uint32_t kHeight = 360;
    const CaelixUpscalerQueryResult sr = Query(rig.device, CAELIX_UPSCALER_DLSS_SR, CAELIX_UPSCALE_QUALITY, kWidth, kHeight);
    const CaelixUpscalerQueryResult rr = Query(rig.device, CAELIX_UPSCALER_DLSS_RR, CAELIX_UPSCALE_QUALITY, kWidth, kHeight);
    if (sr.supported == 0)
    {
        std::printf("DLSS SR unsupported (%s): skipped\n", ReasonName(sr.reason));
        CloseRig(rig);
        SKIP("DLSS SR is not available on this machine or in this build");
    }
    UploadFloorAndBlock(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    const CaelixCamera camera = FrontCamera(static_cast<float>(kWidth) / kHeight);
    Target color = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32B32A32_SFLOAT);
    Target ids = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32_UINT);
    CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 1);
    options.upscaler = CAELIX_UPSCALER_DLSS_SR;
    options.upscale_quality = CAELIX_UPSCALE_QUALITY;

    // 1. A voxel id target at the render size is refused; at the output size it is accepted.
    {
        Target render_ids = CreateTarget(rig.gpu, sr.render_width, sr.render_height, VK_FORMAT_R32G32_UINT);
        ExpectLibraryErrors(1, [&] {
            CHECK(RenderVoxelIds(rig, renderer, color, render_ids, camera, options, nullptr) ==
                  CAELIX_ERROR_INVALID_ARGUMENT);
        });
        CHECK(RenderVoxelIds(rig, renderer, color, ids, camera, options, nullptr) == CAELIX_OK);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroyTarget(rig.gpu, render_ids);
    }

    // 2. Three upscaled frames with the library's Halton jitter, then the same camera without an
    //    upscaler and without jitter at the output size: the two key images are identical (the
    //    dedicated pass traces the unjittered pixel centres). 3. The same with a voxel_id_tmax that
    //    cuts the far floor.
    const auto count_misses = [](const std::vector<uint32_t>& image) {
        size_t misses = 0;
        for (size_t i = 0; i < image.size(); i += 2)
        {
            misses += image[i] == 0xFFFFFFFFu && image[i + 1] == 0xFFFFFFFFu ? 1 : 0;
        }
        return misses;
    };
    const auto compare = [&](uint32_t upscaler, uint32_t render_width, float tmax) {
        CaelixRenderOptions upscaled = options;
        upscaled.upscaler = upscaler;
        upscaled.voxel_id_tmax = tmax;
        const uint32_t phases = caelix_upscaler_jitter_phases(upscaler, render_width, kWidth);
        std::vector<uint32_t> with_upscaler;
        CaelixCamera jittered = camera;
        constexpr uint32_t kFrames = 3;
        for (uint32_t i = 0; i < kFrames; ++i)
        {
            caelix_jitter_halton(i, phases, jittered.jitter);
            upscaled.frame_seed = i;
            REQUIRE(RenderVoxelIds(rig, renderer, color, ids, jittered, upscaled,
                                   i + 1 == kFrames ? &with_upscaler : nullptr) == CAELIX_OK);
        }
        CHECK(UpscalerInfo(renderer).failed == 0);
        CaelixRenderOptions plain = upscaled;
        plain.upscaler = CAELIX_UPSCALER_NONE;
        plain.upscale_quality = 0;
        std::vector<uint32_t> without_upscaler;
        REQUIRE(RenderVoxelIds(rig, renderer, color, ids, camera, plain, &without_upscaler) == CAELIX_OK);
        REQUIRE(with_upscaler.size() == without_upscaler.size());
        size_t differ = 0;
        for (size_t i = 0; i < with_upscaler.size(); i += 2)
        {
            differ += with_upscaler[i] != without_upscaler[i] || with_upscaler[i + 1] != without_upscaler[i + 1] ? 1 : 0;
        }
        const size_t misses = count_misses(with_upscaler);
        std::printf("voxel id %s, voxel_id_tmax %g: %zu of %u pixels differ from the unjittered render without an "
                    "upscaler, %zu miss pixels\n",
                    UpscalerName(upscaler), static_cast<double>(tmax), differ, kWidth * kHeight, misses);
        CHECK(differ == 0);
        return misses;
    };
    const size_t misses_unlimited = compare(CAELIX_UPSCALER_DLSS_SR, sr.render_width, 0.0f);
    // The block's centre is about 50 units from the camera, the floor's far edge about 75.
    const size_t misses_tmax = compare(CAELIX_UPSCALER_DLSS_SR, sr.render_width, 55.0f);
    CHECK(misses_tmax > misses_unlimited);
    CHECK(misses_tmax > 0);
    if (rr.supported != 0)
    {
        compare(CAELIX_UPSCALER_DLSS_RR, rr.render_width, 0.0f);
    }
    else
    {
        std::printf("DLSS RR unsupported (%s): the RR comparison is skipped\n", ReasonName(rr.reason));
    }

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    DestroyTarget(rig.gpu, ids);
    CloseRig(rig);
}

TEST_CASE("background and output-size depth with an upscaler")
{
    Rig rig;
    OpenRig(rig, "caelix_dlss_test", RigDesc(0, 0, 0));
    constexpr uint32_t kWidth = 640;
    constexpr uint32_t kHeight = 360;
    const CaelixUpscalerQueryResult sr = Query(rig.device, CAELIX_UPSCALER_DLSS_SR, CAELIX_UPSCALE_QUALITY, kWidth, kHeight);
    if (sr.supported == 0)
    {
        std::printf("DLSS SR unsupported (%s): skipped\n", ReasonName(sr.reason));
        CloseRig(rig);
        SKIP("DLSS SR is not available on this machine or in this build");
    }
    UploadFloorAndBlock(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    const CaelixCamera camera = FrontCamera(static_cast<float>(kWidth) / kHeight);
    Target color = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32B32A32_SFLOAT);
    Target ids = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32G32_UINT);
    Target output_depth = CreateTarget(rig.gpu, kWidth, kHeight, VK_FORMAT_R32_SFLOAT);
    CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 1);
    options.upscaler = CAELIX_UPSCALER_DLSS_SR;
    options.upscale_quality = CAELIX_UPSCALE_QUALITY;
    const float kBackground[3] = {0.2f, 0.5f, 0.8f};
    const auto constant = [&](uint32_t width, uint32_t height) {
        std::vector<float> rgba(static_cast<size_t>(width) * height * 4);
        for (size_t i = 0; i < rgba.size(); i += 4)
        {
            rgba[i + 0] = kBackground[0];
            rgba[i + 1] = kBackground[1];
            rgba[i + 2] = kBackground[2];
            rgba[i + 3] = 1.0f;
        }
        return rgba;
    };
    BackgroundImage background = CreateBackground(rig.gpu, sr.render_width, sr.render_height);
    UploadBackground(rig.gpu, background, constant(sr.render_width, sr.render_height));
    const uint32_t phases = caelix_upscaler_jitter_phases(CAELIX_UPSCALER_DLSS_SR, sr.render_width, kWidth);

    // 1. A background at the output size is refused (it has the render size); at the render size it is
    //    accepted.
    {
        BackgroundImage output_background = CreateBackground(rig.gpu, kWidth, kHeight);
        UploadBackground(rig.gpu, output_background, constant(kWidth, kHeight));
        CaelixRenderTargets targets{};
        targets.color = color.desc();
        targets.background = output_background.Abi();
        ExpectLibraryErrors(1, [&] {
            CHECK(RenderTargets(rig, renderer, targets, {{&color}}, camera, options) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        targets.background = background.Abi();
        CHECK(RenderTargets(rig, renderer, targets, {{&color}}, camera, options) == CAELIX_OK);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroyBackground(rig.gpu, output_background);
    }

    // 2. SR, the constant background, 3 frames with Halton jitter: every output pixel whose voxel id and
    //    the neighbours' within 2 pixels (5 x 5) are misses is the background colour (DLSS of a constant is
    //    the constant away from the silhouettes). The 3 x 3 neighbourhood is printed too: at 1.5x the
    //    reconstruction still mixes the scene into pixels 2 output pixels from a hit (measured 6.4 %), from
    //    3 pixels on the deviation is below 0.3 %.
    {
        std::vector<uint8_t> colour_bytes;
        std::vector<uint8_t> id_bytes;
        CaelixCamera jittered = camera;
        constexpr uint32_t kFrames = 3;
        for (uint32_t i = 0; i < kFrames; ++i)
        {
            caelix_jitter_halton(i, phases, jittered.jitter);
            CaelixRenderOptions frame_options = options;
            frame_options.frame_seed = i;
            CaelixRenderTargets targets{};
            targets.color = color.desc();
            targets.voxel_id = ids.desc();
            targets.background = background.Abi();
            const bool last = i + 1 == kFrames;
            REQUIRE(RenderTargets(rig, renderer, targets,
                                  {{&color, last ? &colour_bytes : nullptr, 16}, {&ids, last ? &id_bytes : nullptr, 8}},
                                  jittered, frame_options) == CAELIX_OK);
        }
        CHECK(UpscalerInfo(renderer).failed == 0);
        const auto miss = [&](int x, int y) {
            uint32_t words[2];
            std::memcpy(words, id_bytes.data() + (static_cast<size_t>(y) * kWidth + x) * 8, 8);
            return words[0] == 0xFFFFFFFFu && words[1] == 0xFFFFFFFFu;
        };
        // The pixels whose (2 r + 1)^2 neighbourhood is all misses, and their largest relative deviation.
        const auto deviation = [&](int r, size_t& inside) {
            inside = 0;
            double max_deviation = 0.0;
            for (int y = r; y + r < static_cast<int>(kHeight); ++y)
            {
                for (int x = r; x + r < static_cast<int>(kWidth); ++x)
                {
                    bool all_miss = true;
                    for (int dy = -r; dy <= r && all_miss; ++dy)
                    {
                        for (int dx = -r; dx <= r && all_miss; ++dx)
                        {
                            all_miss = miss(x + dx, y + dy);
                        }
                    }
                    if (!all_miss)
                    {
                        continue;
                    }
                    ++inside;
                    float rgba[4];
                    std::memcpy(rgba, colour_bytes.data() + (static_cast<size_t>(y) * kWidth + x) * 16, 16);
                    for (int c = 0; c < 3; ++c)
                    {
                        max_deviation = std::max(
                            max_deviation, std::abs(static_cast<double>(rgba[c]) - kBackground[c]) / kBackground[c]);
                    }
                }
            }
            return max_deviation;
        };
        size_t inside_3 = 0;
        size_t inside_5 = 0;
        const double deviation_3 = deviation(1, inside_3);
        const double deviation_5 = deviation(2, inside_5);
        std::printf("background under DLSS SR: max relative deviation from the background %.4f over %zu output "
                    "pixels with a 5x5 miss neighbourhood (%.4f over %zu with 3x3, not checked)\n",
                    deviation_5, inside_5, deviation_3, inside_3);
        CHECK(inside_5 > 1000);
        CHECK(deviation_5 < 0.02);
    }

    // 3. The depth target at the output size: written by the output-size pass from the unjittered pixel
    //    centres, byte-identical to the depth of an unjittered render without an upscaler at the output
    //    size. A depth target at the render size is still accepted, one at any other size refused.
    {
        std::vector<uint8_t> upscaled_depth;
        CaelixCamera jittered = camera;
        constexpr uint32_t kFrames = 3;
        for (uint32_t i = 0; i < kFrames; ++i)
        {
            caelix_jitter_halton(i, phases, jittered.jitter);
            CaelixRenderOptions frame_options = options;
            frame_options.frame_seed = i;
            CaelixRenderTargets targets{};
            targets.color = color.desc();
            targets.depth = output_depth.desc();
            REQUIRE(RenderTargets(rig, renderer, targets,
                                  {{&color}, {&output_depth, i + 1 == kFrames ? &upscaled_depth : nullptr, 4}},
                                  jittered, frame_options) == CAELIX_OK);
        }
        CaelixRenderOptions plain = options;
        plain.upscaler = CAELIX_UPSCALER_NONE;
        plain.upscale_quality = 0;
        std::vector<uint8_t> plain_depth;
        {
            CaelixRenderTargets targets{};
            targets.color = color.desc();
            targets.depth = output_depth.desc();
            REQUIRE(RenderTargets(rig, renderer, targets, {{&color}, {&output_depth, &plain_depth, 4}}, camera,
                                  plain) == CAELIX_OK);
        }
        REQUIRE(upscaled_depth.size() == plain_depth.size());
        size_t differ = 0;
        size_t misses = 0;
        float max_difference = 0.0f;
        for (size_t i = 0; i < upscaled_depth.size(); i += 4)
        {
            float a = 0.0f;
            float b = 0.0f;
            std::memcpy(&a, upscaled_depth.data() + i, 4);
            std::memcpy(&b, plain_depth.data() + i, 4);
            differ += std::memcmp(upscaled_depth.data() + i, plain_depth.data() + i, 4) != 0 ? 1 : 0;
            misses += a == 1.0f ? 1 : 0;
            max_difference = std::max(max_difference, std::abs(a - b));
        }
        std::printf("output-size depth under DLSS SR: %zu of %u pixels differ from the unjittered render without an "
                    "upscaler (max difference %g), %zu miss pixels\n",
                    differ, kWidth * kHeight, static_cast<double>(max_difference), misses);
        CHECK(differ == 0);
        CHECK(misses > 1000);
        CHECK(misses < static_cast<size_t>(kWidth) * kHeight - 1000);

        Target render_depth = CreateTarget(rig.gpu, sr.render_width, sr.render_height, VK_FORMAT_R32_SFLOAT);
        Target odd_depth = CreateTarget(rig.gpu, sr.render_width + 1, sr.render_height, VK_FORMAT_R32_SFLOAT);
        CaelixRenderTargets targets{};
        targets.color = color.desc();
        targets.depth = render_depth.desc();
        CHECK(RenderTargets(rig, renderer, targets, {{&color}, {&render_depth}}, camera, options) == CAELIX_OK);
        targets.depth = odd_depth.desc();
        ExpectLibraryErrors(1, [&] {
            CHECK(RenderTargets(rig, renderer, targets, {{&color}, {&odd_depth}}, camera, options) ==
                  CAELIX_ERROR_INVALID_ARGUMENT);
        });
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroyTarget(rig.gpu, render_depth);
        DestroyTarget(rig.gpu, odd_depth);
    }

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    DestroyTarget(rig.gpu, ids);
    DestroyTarget(rig.gpu, output_depth);
    DestroyBackground(rig.gpu, background);
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
