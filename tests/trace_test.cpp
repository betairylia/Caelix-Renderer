// The trace and composite kernels (caelix_renderer_render): a one-brick scene and
// the random opaque scene are rendered into offscreen storage images (the colour
// target and the optional geometry targets albedo, normal, depth, motion and
// view_z), which are read back and checked pixel by pixel. The lighting modes are
// checked against a CPU Monte Carlo estimate (an emissive ceiling over a grey
// floor), the procedural sky, primary emission and the reference accumulation;
// sun next-event estimation against the analytic irradiance of the sun cone, the
// sun found by bounce sampling only, and a shadow under a floating block; a host
// sky cubemap (primary miss, SH irradiance of diffuse bounces), the fog of the air
// and the emission scale against closed forms. Two renderers over one flush must
// agree on the motion of a moving entity, and render once per frame each. ReSTIR GI
// without reuse must match the path tracer's reference image, and its reuse must keep
// the mean within a bias bound. The light list: the scene's light table counts its
// emissive bricks, next-event estimation on list lights matches the plain path
// tracer's mean in a closed box lit by one voxel, the path tracer without the flag is
// the same bits whatever the table holds, and the light cells fill and sample.
// Uses the public C ABI only.
#include "test_scenes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace test;

namespace {

constexpr uint32_t kSize = 256;
constexpr float kSky[3] = {0.25f, 0.5f, 0.75f};
constexpr float kNear = 0.1f;
constexpr float kFar = 1000.0f;
constexpr float kMissViewZ = 1e30f;

uint32_t FindMemoryType(Gpu& gpu, uint32_t type_bits, VkMemoryPropertyFlags wanted)
{
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(gpu.device.physical_device.physical_device, &properties);
    uint32_t found = UINT32_MAX;
    for (uint32_t i = 0; i < properties.memoryTypeCount && found == UINT32_MAX; ++i)
    {
        if ((type_bits & (1u << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & wanted) == wanted)
        {
            found = i;
        }
    }
    REQUIRE(found != UINT32_MAX);
    return found;
}

uint32_t BytesPerPixel(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_R16G16_SFLOAT:
            return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32_SFLOAT:
            return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 16;
        default:
            return 0;  // not a format of this test
    }
}

float HalfToFloat(uint16_t h)
{
    const uint32_t exponent = (h >> 10) & 31u;
    const uint32_t mantissa = h & 1023u;
    float value = 0.0f;
    if (exponent == 0)
    {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    }
    else if (exponent == 31)
    {
        value = mantissa != 0 ? NAN : INFINITY;
    }
    else
    {
        value = std::ldexp(static_cast<float>(mantissa | 1024u), static_cast<int>(exponent) - 25);
    }
    return (h & 0x8000u) != 0 ? -value : value;
}

// A storage image that a render call writes, and its last readback.
struct Target
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bytes_per_pixel = 0;
    // UNDEFINED the first time, TRANSFER_SRC_OPTIMAL after a readback.
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::vector<uint8_t> bytes;  // width * height * bytes_per_pixel, row 0 = top

    const uint8_t* at(uint32_t x, uint32_t y) const
    {
        return bytes.data() + (static_cast<size_t>(y) * width + x) * bytes_per_pixel;
    }
    float F32(uint32_t x, uint32_t y, uint32_t c) const
    {
        float value = 0.0f;
        std::memcpy(&value, at(x, y) + 4 * c, 4);
        return value;
    }
    float F16(uint32_t x, uint32_t y, uint32_t c) const
    {
        uint16_t value = 0;
        std::memcpy(&value, at(x, y) + 2 * c, 2);
        return HalfToFloat(value);
    }
};

// An image with the usages a target needs (STORAGE only) plus TRANSFER_SRC for the
// readback. No TRANSFER_DST: the library never clears a target.
Target CreateTarget(Gpu& gpu, VkFormat format, uint32_t width = kSize, uint32_t height = kSize)
{
    Target out;
    out.format = format;
    out.width = width;
    out.height = height;
    out.bytes_per_pixel = BytesPerPixel(format);
    REQUIRE(out.bytes_per_pixel != 0);
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
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex =
        FindMemoryType(gpu, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    REQUIRE(vkAllocateMemory(gpu.device.device, &allocate, nullptr, &out.memory) == VK_SUCCESS);
    REQUIRE(vkBindImageMemory(gpu.device.device, out.image, out.memory, 0) == VK_SUCCESS);
    return out;
}

void DestroyTarget(Gpu& gpu, Target& target)
{
    vkDestroyImage(gpu.device.device, target.image, nullptr);
    vkFreeMemory(gpu.device.device, target.memory, nullptr);
}

// The targets of one render call, in CaelixRenderTargets order; null = absent.
enum TargetIndex
{
    kColor,
    kAlbedo,
    kNormal,
    kDepth,
    kMotion,
    kViewZ,
    kTargetCount
};
struct TargetSet
{
    Target* t[kTargetCount] = {};
};

CaelixTargetImage Abi(const Target* target)
{
    CaelixTargetImage out{};
    if (target != nullptr)
    {
        out.image = reinterpret_cast<uint64_t>(target->image);
        out.format = static_cast<uint32_t>(target->format);
        out.width = target->width;
        out.height = target->height;
        out.layout = static_cast<uint32_t>(target->layout);
    }
    return out;
}

CaelixRenderTargets ToAbi(const TargetSet& set)
{
    CaelixRenderTargets out{};
    out.color = Abi(set.t[kColor]);
    out.albedo = Abi(set.t[kAlbedo]);
    out.normal = Abi(set.t[kNormal]);
    out.depth = Abi(set.t[kDepth]);
    out.motion = Abi(set.t[kMotion]);
    out.view_z = Abi(set.t[kViewZ]);
    return out;
}

// Column-major projections for the camera's view space (+z forward), as a host
// builds them. Element (row r, column c) = m[c * 4 + r].
void NormalZProjection(float m[16], float tan_half_fov_y, float aspect)
{
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = 1.0f / (tan_half_fov_y * aspect);
    m[5] = 1.0f / tan_half_fov_y;
    m[10] = kFar / (kFar - kNear);  // clip.z = far / (far - near) * (z - near)
    m[14] = -kFar * kNear / (kFar - kNear);
    m[11] = 1.0f;  // clip.w = z
}

void ReverseZInfiniteProjection(float m[16], float tan_half_fov_y, float aspect)
{
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = 1.0f / (tan_half_fov_y * aspect);
    m[5] = 1.0f / tan_half_fov_y;
    m[14] = kNear;  // clip.z = near
    m[11] = 1.0f;   // clip.w = z
}

// Device depth clip.z / clip.w of a point at view depth z on the view axis.
float ProjectDepth(const float m[16], float z)
{
    return (m[10] * z + m[14]) / (m[11] * z + m[15]);
}

// Camera at `position` looking along +Z, vertical field of view 60 degrees, with
// the normal-Z projection.
CaelixCamera LookPlusZ(float x, float y, float z)
{
    CaelixCamera camera{};
    camera.position[0] = x;
    camera.position[1] = y;
    camera.position[2] = z;
    camera.right[0] = 1.0f;
    camera.up[1] = 1.0f;
    camera.forward[2] = 1.0f;
    camera.tan_half_fov_y = std::tan(30.0f * 3.14159265f / 180.0f);
    camera.aspect = 1.0f;
    camera.near_z = kNear;
    camera.far_z = kFar;
    NormalZProjection(camera.projection, camera.tan_half_fov_y, camera.aspect);
    return camera;
}

CaelixRenderOptions Options(uint32_t view)
{
    CaelixRenderOptions options{};
    options.debug_view = view;
    options.flags = 0;  // unshaded: the checks compare exact albedo
    options.sun_dir[0] = 0.3f;
    options.sun_dir[1] = 1.0f;
    options.sun_dir[2] = -0.5f;
    options.sky_color[0] = kSky[0];
    options.sky_color[1] = kSky[1];
    options.sky_color[2] = kSky[2];
    options.miss_depth = 1.0f;
    options.miss_view_z = kMissViewZ;
    return options;
}

// One frame: flush, render into the present targets, copy each to a host buffer.
// Afterwards every present target holds its pixels and is in TRANSFER_SRC_OPTIMAL.
// Without `readback` the targets are only rendered and stay in GENERAL.
void Render(Rig& rig, CaelixRenderer* renderer, TargetSet& set, const CaelixCamera& camera,
            const CaelixRenderOptions& options, bool readback = true)
{
    Readback readbacks[kTargetCount];
    for (int i = 0; i < kTargetCount && readback; ++i)
    {
        if (set.t[i] != nullptr)
        {
            const Target& target = *set.t[i];
            CreateReadback(rig.gpu, static_cast<VkDeviceSize>(target.width) * target.height * target.bytes_per_pixel,
                           readbacks[i]);
        }
    }
    const uint64_t frame = rig.frames.next++;
    SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
        CaelixFrameDesc desc{};
        desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
        desc.frame_number = frame;
        desc.safe_frame_number = frame - 1;
        REQUIRE(caelix_scene_flush(rig.scene, &desc) == CAELIX_OK);
        const CaelixRenderTargets targets = ToAbi(set);
        REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &options) == CAELIX_OK);

        for (int i = 0; i < kTargetCount && readback; ++i)
        {
            if (set.t[i] == nullptr)
            {
                continue;
            }
            Target& target = *set.t[i];
            // GENERAL (render contract) -> TRANSFER_SRC, copy.
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
            vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   readbacks[i].buffer, 1, &region);
        }
        if (readback)
        {
            CopyToHostBarrier(cmd);
        }
    });
    REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
    if (!readback)
    {
        for (int i = 0; i < kTargetCount; ++i)
        {
            if (set.t[i] != nullptr)
            {
                set.t[i]->layout = VK_IMAGE_LAYOUT_GENERAL;
            }
        }
        return;
    }
    for (int i = 0; i < kTargetCount; ++i)
    {
        if (set.t[i] == nullptr)
        {
            continue;
        }
        Target& target = *set.t[i];
        target.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(readbacks[i].words);
        target.bytes.assign(bytes, bytes + static_cast<size_t>(target.width) * target.height *
                                               target.bytes_per_pixel);
        DestroyReadback(rig.gpu, readbacks[i]);
    }
}

// Render with the colour target only.
void RenderColor(Rig& rig, CaelixRenderer* renderer, Target& color, const CaelixCamera& camera,
                 const CaelixRenderOptions& options)
{
    TargetSet set;
    set.t[kColor] = &color;
    Render(rig, renderer, set, camera, options);
}

// Calls caelix_renderer_render with a dummy command buffer and requires `expected`
// with one library error: argument checks run before anything is recorded.
void ExpectRenderError(Rig& rig, CaelixRenderer* renderer, const CaelixRenderTargets& targets,
                       const CaelixCamera& camera, CaelixResult expected)
{
    ExpectLibraryErrors(1, [&] {
        CaelixFrameDesc desc{};
        desc.command_buffer = 1;
        desc.frame_number = rig.frames.next;
        const CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
        REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &options) == expected);
    });
}

// Flushes until every acceleration structure is built.
void DrainBuilds(Rig& rig)
{
    int flushes = 0;
    while (Stats(rig.scene).as_builds_pending > 0)
    {
        FlushFrame(rig.gpu, rig.scene, rig.frames);
        REQUIRE(++flushes < 64);
    }
}

bool Near(const uint8_t* pixel, float r, float g, float b, int tolerance)
{
    const float expected[3] = {r, g, b};
    for (int c = 0; c < 3; ++c)
    {
        if (std::abs(static_cast<int>(pixel[c]) - static_cast<int>(std::lround(expected[c] * 255.0f))) >
            tolerance)
        {
            return false;
        }
    }
    return true;
}

bool IsSky(const uint8_t* pixel)
{
    return Near(pixel, kSky[0], kSky[1], kSky[2], 2);
}

void PrintPixel(const char* name, const uint8_t* pixel)
{
    std::printf("%s: %u %u %u %u\n", name, pixel[0], pixel[1], pixel[2], pixel[3]);
}

// One opaque pure-red brick (packed: bit 15 = opaque, red = 31 << 10) at key (0, 0, 0):
// blocks [0, 8)^3 of the entity.
void UploadRedBrick(Rig& rig)
{
    World world;
    Brick brick{};
    brick.fill(static_cast<uint16_t>(0x8000u | (31u << 10)));
    world.bricks[{0, 0, 0}] = brick;
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

}  // namespace

TEST_CASE("one red brick: albedo, normal and brick views; sky before the first TLAS")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target target = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
    const CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);

    // No bricks, no TLAS: the kernel traces nothing and writes the miss values (the
    // TLAS binding is not written; it is PARTIALLY_BOUND).
    {
        Target depth = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT);
        Target albedo = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
        TargetSet set;
        set.t[kColor] = &target;
        set.t[kDepth] = &depth;
        set.t[kAlbedo] = &albedo;
        Render(rig, renderer, set, camera, Options(CAELIX_VIEW_ALBEDO));
        REQUIRE(IsSky(target.at(kSize / 2, kSize / 2)));
        REQUIRE(IsSky(target.at(0, 0)));
        REQUIRE(target.at(kSize / 2, kSize / 2)[3] == 255);
        REQUIRE(depth.F32(kSize / 2, kSize / 2, 0) == 1.0f);
        REQUIRE(albedo.at(kSize / 2, kSize / 2)[3] == 0);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroyTarget(rig.gpu, depth);
        DestroyTarget(rig.gpu, albedo);
    }

    UploadRedBrick(rig);

    RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_ALBEDO));
    const uint8_t* centre = target.at(kSize / 2, kSize / 2);
    PrintPixel("albedo centre", centre);
    PrintPixel("albedo corner", target.at(0, 0));
    CHECK(centre[0] > 200);
    CHECK(centre[1] < 30);
    CHECK(centre[2] < 30);
    CHECK(IsSky(target.at(0, 0)));
    CHECK(IsSky(target.at(kSize - 1, kSize - 1)));

    // The camera sees the -Z face: normal (0, 0, -1) -> (0.5, 0.5, 0).
    RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_NORMAL));
    PrintPixel("normal centre", target.at(kSize / 2, kSize / 2));
    CHECK(Near(target.at(kSize / 2, kSize / 2), 0.5f, 0.5f, 0.0f, 2));

    RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_BRICK));
    const uint8_t* brick_centre = target.at(kSize / 2, kSize / 2);
    PrintPixel("brick centre", brick_centre);
    CHECK((brick_centre[0] | brick_centre[1] | brick_centre[2]) != 0);
    CHECK(!IsSky(brick_centre));

    // The other views run without validation messages.
    RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_DEPTH));
    RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_GROUP));
    RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_STEPS));
    CHECK(!IsSky(target.at(kSize / 2, kSize / 2)));  // one candidate
    CHECK(IsSky(target.at(0, 0)));                   // none

    // GPU timings: every frame was waited for, so the previous render's frame is resolved.
    CaelixGpuTimings timings{};
    caelix_renderer_get_gpu_timings(renderer, &timings);
    std::printf("timings: frame %llu, trace %.4f ms, flush %.4f ms\n",
                static_cast<unsigned long long>(timings.frame_number), timings.trace_ms,
                timings.flush_ms);
    CHECK(timings.frame_number > 0);
    CHECK(timings.frame_number < rig.frames.next);
    CHECK(timings.trace_ms > 0.0f);

    // Invalid arguments are rejected with an error log.
    ExpectLibraryErrors(1, [&] {
        CaelixRenderOptions bad = Options(99);
        CaelixFrameDesc desc{};
        desc.command_buffer = 1;
        desc.frame_number = rig.frames.next;
        CaelixRenderTargets targets{};
        targets.color = Abi(&target);
        REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &bad) ==
                CAELIX_ERROR_INVALID_ARGUMENT);
    });

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, target);
    CloseRig(rig);
}

TEST_CASE("geometry targets: albedo, normal, depth, motion, view_z")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadRedBrick(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);

    Target color = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
    Target albedo = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
    Target normal = CreateTarget(rig.gpu, VK_FORMAT_R16G16B16A16_SFLOAT);
    Target depth = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT);
    Target motion = CreateTarget(rig.gpu, VK_FORMAT_R32G32_SFLOAT);
    Target view_z = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT);
    TargetSet all;
    all.t[kColor] = &color;
    all.t[kAlbedo] = &albedo;
    all.t[kNormal] = &normal;
    all.t[kDepth] = &depth;
    all.t[kMotion] = &motion;
    all.t[kViewZ] = &view_z;

    constexpr uint32_t cx = kSize / 2;
    constexpr uint32_t cy = kSize / 2;
    const auto is_hit = [&](uint32_t x, uint32_t y) { return albedo.at(x, y)[3] == 255; };

    // 1. First render of the renderer, normal-Z projection. The brick covers blocks
    //    [0, 8)^3, so its -Z face lies in the plane z = 0; the camera sits at z = -20
    //    looking +Z, and the centre pixel's ray hits that face near (4, 4, 0):
    //    view_z = 0 - (-20) = 20.
    const CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);
    Render(rig, renderer, all, camera, Options(CAELIX_VIEW_ALBEDO));
    const float view_z0 = view_z.F32(cx, cy, 0);
    std::printf("centre: albedo %u %u %u %u, normal %.4f %.4f %.4f %.4f, view_z %.6f, depth %.8f\n",
                albedo.at(cx, cy)[0], albedo.at(cx, cy)[1], albedo.at(cx, cy)[2],
                albedo.at(cx, cy)[3], normal.F16(cx, cy, 0), normal.F16(cx, cy, 1),
                normal.F16(cx, cy, 2), normal.F16(cx, cy, 3), view_z0, depth.F32(cx, cy, 0));
    CHECK(albedo.at(cx, cy)[0] == 255);
    CHECK(albedo.at(cx, cy)[1] == 0);
    CHECK(albedo.at(cx, cy)[2] == 0);
    CHECK(albedo.at(cx, cy)[3] == 255);
    CHECK(std::abs(normal.F16(cx, cy, 0)) < 1e-3f);
    CHECK(std::abs(normal.F16(cx, cy, 1)) < 1e-3f);
    CHECK(std::abs(normal.F16(cx, cy, 2) + 1.0f) < 1e-3f);
    CHECK(normal.F16(cx, cy, 3) == 1.0f);
    CHECK(std::abs(view_z0 - 20.0f) < 1e-3f);
    CHECK(std::abs(depth.F32(cx, cy, 0) - ProjectDepth(camera.projection, view_z0)) < 1e-5f);
    CHECK(color.at(cx, cy)[0] == 255);
    CHECK(color.at(cx, cy)[3] == 255);
    // The corner sees the sky: miss values everywhere.
    CHECK(albedo.at(0, 0)[0] == 0);
    CHECK(albedo.at(0, 0)[3] == 0);
    CHECK(normal.F16(0, 0, 3) == 0.0f);
    CHECK(depth.F32(0, 0, 0) == 1.0f);
    CHECK(view_z.F32(0, 0, 0) == kMissViewZ);
    CHECK(IsSky(color.at(0, 0)));
    CHECK(color.at(0, 0)[3] == 255);  // the colour target is opaque everywhere
    // First render: no previous camera, so motion is exactly 0 everywhere. Coverage
    // agrees across the targets.
    uint32_t hits = 0;
    uint32_t first_motion_nonzero = 0;
    uint32_t coverage_mismatch = 0;
    for (uint32_t y = 0; y < kSize; ++y)
    {
        for (uint32_t x = 0; x < kSize; ++x)
        {
            const bool hit = is_hit(x, y);
            hits += hit ? 1 : 0;
            if (motion.F32(x, y, 0) != 0.0f || motion.F32(x, y, 1) != 0.0f)
            {
                ++first_motion_nonzero;
            }
            const bool normal_hit = normal.F16(x, y, 3) == 1.0f;
            const bool view_hit = view_z.F32(x, y, 0) != kMissViewZ;
            const bool depth_hit = depth.F32(x, y, 0) != 1.0f;
            if (normal_hit != hit || view_hit != hit || depth_hit != hit ||
                (!hit && albedo.at(x, y)[0] != 0))
            {
                ++coverage_mismatch;
            }
        }
    }
    std::printf("first render: %u hit pixels, %u nonzero motion, %u coverage mismatches\n", hits,
                first_motion_nonzero, coverage_mismatch);
    CHECK(hits > 1000);
    CHECK(first_motion_nonzero == 0);
    CHECK(coverage_mismatch == 0);

    // 2. Same camera with a reverse-Z infinite projection (depth = near / z, miss 0).
    //    The previous camera now exists and equals the current one: motion ~0 on hits.
    {
        CaelixCamera reverse = camera;
        ReverseZInfiniteProjection(reverse.projection, reverse.tan_half_fov_y, reverse.aspect);
        CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
        options.miss_depth = 0.0f;
        Render(rig, renderer, all, reverse, options);
        const float z = view_z.F32(cx, cy, 0);
        std::printf("reverse Z: centre depth %.8f, expected %.8f\n", depth.F32(cx, cy, 0),
                    ProjectDepth(reverse.projection, z));
        CHECK(std::abs(depth.F32(cx, cy, 0) - ProjectDepth(reverse.projection, z)) < 1e-5f);
        CHECK(std::abs(depth.F32(cx, cy, 0) - kNear / 20.0f) < 1e-5f);
        CHECK(depth.F32(0, 0, 0) == 0.0f);
        float max_motion = 0.0f;
        uint32_t miss_motion_nonzero = 0;
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                const float m = std::max(std::abs(motion.F32(x, y, 0)), std::abs(motion.F32(x, y, 1)));
                if (is_hit(x, y))
                {
                    max_motion = std::max(max_motion, m);
                }
                else if (m != 0.0f)
                {
                    ++miss_motion_nonzero;
                }
            }
        }
        std::printf("same camera: max |motion| on hits %.6f px\n", max_motion);
        CHECK(max_motion < 1e-3f);
        CHECK(miss_motion_nonzero == 0);
    }

    // 3. The camera moves +1 unit along `right` (+X). A static point at view depth z
    //    then sits 1 unit further LEFT in the new view than in the previous one, i.e.
    //    the previous camera saw it 1 unit to the right: previous view x = current
    //    view x + 1. One unit at depth z spans width / (2 * tan_half_fov_y * aspect * z)
    //    pixels, so motion.x = previous_pixel.x - current_pixel.x = +256 / (2 * 0.57735
    //    * 1 * 20) = +11.085 px, motion.y = 0.
    const CaelixCamera moved = LookPlusZ(5.0f, 4.0f, -20.0f);
    Render(rig, renderer, all, moved, Options(CAELIX_VIEW_ALBEDO));
    const float view_z1 = view_z.F32(cx, cy, 0);
    const float pixels_per_unit =
        static_cast<float>(kSize) / (2.0f * moved.tan_half_fov_y * moved.aspect * view_z1);
    std::printf("camera +1 right: centre motion %.5f %.5f px, expected %.5f 0, view_z %.6f\n",
                motion.F32(cx, cy, 0), motion.F32(cx, cy, 1), pixels_per_unit, view_z1);
    REQUIRE(is_hit(cx, cy));
    CHECK(std::abs(view_z1 - 20.0f) < 1e-3f);
    CHECK(std::abs(motion.F32(cx, cy, 0) - pixels_per_unit) < 1e-2f);
    CHECK(std::abs(motion.F32(cx, cy, 0) - 11.085f) < 2e-2f);
    CHECK(std::abs(motion.F32(cx, cy, 1)) < 1e-3f);

    // 4. Only the depth target (no colour target): G-buffer style call.
    {
        TargetSet depth_only;
        depth_only.t[kDepth] = &depth;
        Render(rig, renderer, depth_only, moved, Options(CAELIX_VIEW_ALBEDO));
        CHECK(std::abs(depth.F32(cx, cy, 0) - ProjectDepth(moved.projection, view_z1)) < 1e-5f);
        CHECK(depth.F32(0, 0, 0) == 1.0f);
    }
    // Argument errors: no target, sizes differ, one image twice, a wrong format.
    {
        ExpectRenderError(rig, renderer, CaelixRenderTargets{}, moved, CAELIX_ERROR_INVALID_ARGUMENT);
        Target small = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT, kSize / 2, kSize / 2);
        CaelixRenderTargets targets{};
        targets.depth = Abi(&depth);
        targets.view_z = Abi(&small);
        ExpectRenderError(rig, renderer, targets, moved, CAELIX_ERROR_INVALID_ARGUMENT);
        targets.view_z = Abi(&depth);
        ExpectRenderError(rig, renderer, targets, moved, CAELIX_ERROR_INVALID_ARGUMENT);
        targets = CaelixRenderTargets{};
        targets.depth = Abi(&color);  // RGBA8 is not an accepted depth format
        ExpectRenderError(rig, renderer, targets, moved, CAELIX_ERROR_INVALID_ARGUMENT);
        DestroyTarget(rig.gpu, small);
    }

    // 5. The entity moves +2 along X; the camera stays (`moved`, also the previous
    //    camera of this call). The flush of this frame writes the new transform and
    //    keeps the old one as the previous rows. The centre ray now hits world x ~ 5 =
    //    entity-local x ~ 3; under the previous transform that point was at world
    //    x ~ 3, i.e. 2 units LEFT of where it is now, so motion.x = -2 * 11.085 =
    //    -22.17 px.
    {
        const float position[3] = {2.0f, 0.0f, 0.0f};
        const float rotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        REQUIRE(caelix_scene_set_entity_transform(rig.scene, rig.entity, position, rotation) ==
                CAELIX_OK);
    }
    //    This frame renders in the path trace mode (no bounce) with the motion view, so
    //    the colour target (RGBA8) shows the motion: rgb = (motion.xy / 32 * 0.5 + 0.5,
    //    0.5 * coverage), no sRGB encode. Centre: red = (-22.17 / 32) * 0.5 + 0.5 = 0.154.
    CaelixRenderOptions motion_view = Options(CAELIX_VIEW_MOTION);
    motion_view.mode = CAELIX_MODE_PATH_TRACE;
    motion_view.bounce_count = 0;
    Render(rig, renderer, all, moved, motion_view);
    CHECK(Stats(rig.scene).as_builds_pending == 0);
    PrintPixel("motion view centre", color.at(cx, cy));
    PrintPixel("motion view corner", color.at(0, 0));
    CHECK(std::abs(static_cast<int>(color.at(cx, cy)[0]) - 39) <= 1);
    CHECK(std::abs(static_cast<int>(color.at(cx, cy)[1]) - 128) <= 1);
    CHECK(std::abs(static_cast<int>(color.at(cx, cy)[2]) - 128) <= 1);
    CHECK(color.at(cx, cy)[3] == 255);
    CHECK(std::abs(static_cast<int>(color.at(0, 0)[0]) - 128) <= 1);
    CHECK(std::abs(static_cast<int>(color.at(0, 0)[1]) - 128) <= 1);
    CHECK(color.at(0, 0)[2] == 0);
    CHECK(color.at(0, 0)[3] == 0);
    const float view_z2 = view_z.F32(cx, cy, 0);
    const float entity_expected =
        -2.0f * static_cast<float>(kSize) / (2.0f * moved.tan_half_fov_y * moved.aspect * view_z2);
    std::printf("entity +2 x: centre motion %.5f %.5f px, expected %.5f 0\n", motion.F32(cx, cy, 0),
                motion.F32(cx, cy, 1), entity_expected);
    REQUIRE(is_hit(cx, cy));
    CHECK(std::abs(motion.F32(cx, cy, 0) - entity_expected) < 2e-2f);
    CHECK(std::abs(motion.F32(cx, cy, 0) + 22.17f) < 4e-2f);
    CHECK(std::abs(motion.F32(cx, cy, 1)) < 1e-3f);

    // 6. Nothing moves: the next flush makes prev = cur again, motion returns to ~0.
    Render(rig, renderer, all, moved, Options(CAELIX_VIEW_ALBEDO));
    std::printf("at rest again: centre motion %.6f %.6f px\n", motion.F32(cx, cy, 0),
                motion.F32(cx, cy, 1));
    CHECK(std::abs(motion.F32(cx, cy, 0)) < 1e-3f);
    CHECK(std::abs(motion.F32(cx, cy, 1)) < 1e-3f);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    for (int i = 0; i < kTargetCount; ++i)
    {
        DestroyTarget(rig.gpu, *all.t[i]);
    }
    CloseRig(rig);
}

namespace {

// Records a copy of `target` (GENERAL, render contract) into `readback`; the target is
// left in TRANSFER_SRC_OPTIMAL.
void RecordTargetCopy(VkCommandBuffer cmd, const Target& target, const Readback& readback)
{
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
    vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1,
                           &region);
}

}  // namespace

TEST_CASE("two renderers over one flush: both see the entity motion; one render per renderer per frame")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadRedBrick(rig);
    constexpr int kRenderers = 2;  // A and B
    CaelixRenderer* renderers[kRenderers] = {};
    for (CaelixRenderer*& renderer : renderers)
    {
        REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    }
    // Per renderer: albedo (coverage), motion and view_z targets, in this order.
    constexpr int kPerRenderer = 3;
    Target targets[kRenderers][kPerRenderer];
    for (int r = 0; r < kRenderers; ++r)
    {
        targets[r][0] = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
        targets[r][1] = CreateTarget(rig.gpu, VK_FORMAT_R32G32_SFLOAT);
        targets[r][2] = CreateTarget(rig.gpu, VK_FORMAT_R32_SFLOAT);
    }
    const CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);
    const CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
    constexpr uint32_t cx = kSize / 2;
    constexpr uint32_t cy = kSize / 2;

    // One frame: one flush, then A and B render with the same frame desc into the same
    // command buffer; every target is read back. `used` receives the frame desc.
    const auto render_frame = [&](CaelixFrameDesc& used) {
        Readback readbacks[kRenderers][kPerRenderer];
        for (int r = 0; r < kRenderers; ++r)
        {
            for (int t = 0; t < kPerRenderer; ++t)
            {
                const Target& target = targets[r][t];
                CreateReadback(rig.gpu,
                               static_cast<VkDeviceSize>(target.width) * target.height * target.bytes_per_pixel,
                               readbacks[r][t]);
            }
        }
        const uint64_t frame = rig.frames.next;
        SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
            CaelixFrameDesc frame_desc{};
            frame_desc.command_buffer = reinterpret_cast<uint64_t>(cmd);
            frame_desc.frame_number = frame;
            frame_desc.safe_frame_number = frame - 1;
            REQUIRE(caelix_scene_flush(rig.scene, &frame_desc) == CAELIX_OK);
            for (int r = 0; r < kRenderers; ++r)
            {
                CaelixRenderTargets abi{};
                abi.albedo = Abi(&targets[r][0]);
                abi.motion = Abi(&targets[r][1]);
                abi.view_z = Abi(&targets[r][2]);
                REQUIRE(caelix_renderer_render(renderers[r], &frame_desc, &abi, &camera, &options) ==
                        CAELIX_OK);
            }
            for (int r = 0; r < kRenderers; ++r)
            {
                for (int t = 0; t < kPerRenderer; ++t)
                {
                    RecordTargetCopy(cmd, targets[r][t], readbacks[r][t]);
                }
            }
            CopyToHostBarrier(cmd);
            used = frame_desc;
        });
        REQUIRE(caelix_scene_recycle(rig.scene, frame) == CAELIX_OK);
        ++rig.frames.next;
        for (int r = 0; r < kRenderers; ++r)
        {
            for (int t = 0; t < kPerRenderer; ++t)
            {
                Target& target = targets[r][t];
                target.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(readbacks[r][t].words);
                target.bytes.assign(bytes, bytes + static_cast<size_t>(target.width) * target.height *
                                                       target.bytes_per_pixel);
                DestroyReadback(rig.gpu, readbacks[r][t]);
            }
        }
    };
    const auto motion_x = [&](int r) { return targets[r][1].F32(cx, cy, 0); };
    const auto motion_y = [&](int r) { return targets[r][1].F32(cx, cy, 1); };

    // Frame 1: the first render of both renderers: no previous camera, motion 0 everywhere.
    CaelixFrameDesc desc{};
    render_frame(desc);
    for (int r = 0; r < kRenderers; ++r)
    {
        REQUIRE(targets[r][0].at(cx, cy)[3] == 255);  // the centre sees the brick
        uint32_t nonzero = 0;
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                if (targets[r][1].F32(x, y, 0) != 0.0f || targets[r][1].F32(x, y, 1) != 0.0f)
                {
                    ++nonzero;
                }
            }
        }
        std::printf("frame 1, renderer %c: %u pixels with nonzero motion\n", 'A' + r, nonzero);
        CHECK(nonzero == 0);
    }

    // Frame 2: same camera, nothing moves: motion ~0 at the centre.
    render_frame(desc);
    for (int r = 0; r < kRenderers; ++r)
    {
        std::printf("frame 2, renderer %c: centre motion %.6f %.6f px\n", 'A' + r, motion_x(r),
                    motion_y(r));
        CHECK(std::abs(motion_x(r)) < 1e-3f);
        CHECK(std::abs(motion_y(r)) < 1e-3f);
    }

    // Frame 3: the entity moves +2 along X. Both renderers render after the same flush, so
    // both see the entity's previous transform: motion.x = -2 units at the centre's view z.
    {
        const float position[3] = {2.0f, 0.0f, 0.0f};
        const float rotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        REQUIRE(caelix_scene_set_entity_transform(rig.scene, rig.entity, position, rotation) ==
                CAELIX_OK);
    }
    render_frame(desc);
    for (int r = 0; r < kRenderers; ++r)
    {
        REQUIRE(targets[r][0].at(cx, cy)[3] == 255);
        const float view_z = targets[r][2].F32(cx, cy, 0);
        const float expected =
            -2.0f * static_cast<float>(kSize) / (2.0f * camera.tan_half_fov_y * camera.aspect * view_z);
        std::printf("frame 3, renderer %c: centre motion %.5f %.5f px, expected %.5f 0, view_z %.6f\n",
                    'A' + r, motion_x(r), motion_y(r), expected, view_z);
        CHECK(std::abs(motion_x(r) - expected) < 2e-2f);
        CHECK(std::abs(motion_x(r) + 22.17f) < 4e-2f);
        CHECK(std::abs(motion_y(r)) < 1e-3f);
    }
    CHECK(std::abs(motion_x(0) - motion_x(1)) < 1e-4f);
    CHECK(std::abs(motion_y(0) - motion_y(1)) < 1e-4f);

    // A second render of A with frame 3's desc (a fourth command buffer, no new flush): one
    // render per renderer per frame.
    SubmitAndWait(rig.gpu, [&](VkCommandBuffer cmd) {
        CaelixFrameDesc again = desc;
        again.command_buffer = reinterpret_cast<uint64_t>(cmd);
        CaelixRenderTargets abi{};
        abi.albedo = Abi(&targets[0][0]);
        abi.motion = Abi(&targets[0][1]);
        abi.view_z = Abi(&targets[0][2]);
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_renderer_render(renderers[0], &again, &abi, &camera, &options) ==
                  CAELIX_ERROR_INVALID_ARGUMENT);
        });
    });

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    for (int r = 0; r < kRenderers; ++r)
    {
        caelix_renderer_release_target_views(renderers[r]);
        caelix_renderer_destroy(renderers[r]);
        for (Target& target : targets[r])
        {
            DestroyTarget(rig.gpu, target);
        }
    }
    CloseRig(rig);
}

TEST_CASE("random opaque scene: hits, no air boundary")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target target = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);

    std::mt19937 rng(1234);
    World world = RandomOpaqueWorld(rng, 600);
    // Pure magenta (r = b = 31, g = 0) is the air marker: no opaque block may have it.
    for (auto& [key, brick] : world.bricks)
    {
        for (uint16_t& block : brick)
        {
            if (block == 0xFC1Fu)
            {
                block = 0xFC1Eu;
            }
        }
    }
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);

    // Keys span about [-40, 40] x [-20, 20] x [-40, 40] bricks = +-320 blocks.
    const CaelixCamera camera = LookPlusZ(0.0f, 0.0f, -900.0f);
    // Unshaded, so an air hit is exactly magenta.
    RenderColor(rig, renderer, target, camera, Options(CAELIX_VIEW_ALBEDO));
    uint32_t hits = 0;
    uint32_t magenta = 0;
    for (uint32_t y = 0; y < kSize; ++y)
    {
        for (uint32_t x = 0; x < kSize; ++x)
        {
            const uint8_t* p = target.at(x, y);
            if (!IsSky(p))
            {
                ++hits;
            }
            if (p[0] > 250 && p[1] < 5 && p[2] > 250)
            {
                ++magenta;
            }
        }
    }
    std::printf("random scene: %u of %u pixels hit, %u magenta\n", hits, kSize * kSize, magenta);
    CHECK(hits > 0);
    CHECK(magenta == 0);

    CaelixRenderOptions shaded = Options(CAELIX_VIEW_ALBEDO);
    shaded.flags = CAELIX_RENDER_FLAG_SHADING;
    RenderColor(rig, renderer, target, camera, shaded);
    for (uint32_t view = CAELIX_VIEW_NORMAL; view <= CAELIX_VIEW_STEPS; ++view)
    {
        RenderColor(rig, renderer, target, camera, Options(view));
    }

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, target);
    CloseRig(rig);
}

// ---------------------------------------------------------------------------
// Lighting modes (CAELIX_MODE_PATH_TRACE, CAELIX_MODE_REFERENCE)

namespace {

constexpr float kPi = 3.14159265358979f;
constexpr float kZenith[3] = {0.2f, 0.4f, 0.9f};
constexpr float kHorizon[3] = {0.7f, 0.8f, 0.95f};
constexpr float kGround[3] = {0.25f, 0.22f, 0.2f};

void Set3(float out[3], const float in[3])
{
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
}

// Path-traced options: the given mode and bounces, the test sky, no sun radiance,
// exposure 1.
CaelixRenderOptions LightOptions(uint32_t mode, uint32_t bounces, uint32_t seed)
{
    CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
    options.mode = mode;
    options.bounce_count = bounces;
    options.flags = CAELIX_RENDER_FLAG_SKY_ON_MISS;
    options.exposure = 1.0f;
    Set3(options.sky_zenith, kZenith);
    Set3(options.sky_horizon, kHorizon);
    Set3(options.sky_ground, kGround);
    options.frame_seed = seed;
    return options;
}

// CPU copy of SkyRadiance in shaders/caelix_shading.hlsl, without the sun.
void SkyRadiance(const float dir[3], float out[3])
{
    for (int c = 0; c < 3; ++c)
    {
        if (dir[1] >= 0.0f)
        {
            const float t = std::sqrt(std::min(dir[1], 1.0f));
            out[c] = kHorizon[c] + (kZenith[c] - kHorizon[c]) * t;
        }
        else
        {
            const float t = std::sqrt(std::min(-dir[1], 1.0f));
            out[c] = kHorizon[c] + (kGround[c] - kHorizon[c]) * t;
        }
    }
}

// CPU copy of the primary ray direction (CaelixCamera's pinhole mapping).
void PrimaryDirection(const CaelixCamera& camera, uint32_t px, uint32_t py, uint32_t width,
                      uint32_t height, float out[3])
{
    const float nx = (static_cast<float>(px) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f;
    const float ny = 1.0f - (static_cast<float>(py) + 0.5f) / static_cast<float>(height) * 2.0f;
    float length = 0.0f;
    for (int c = 0; c < 3; ++c)
    {
        out[c] = camera.right[c] * (nx * camera.tan_half_fov_y * camera.aspect) +
                 camera.up[c] * (ny * camera.tan_half_fov_y) + camera.forward[c];
        length += out[c] * out[c];
    }
    length = std::sqrt(length);
    for (int c = 0; c < 3; ++c)
    {
        out[c] /= length;
    }
}

float LinearToSrgb(float c)
{
    c = std::clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// Mean of channel `c` of an RGBA32F target over the centre `size` x `size` pixels.
double CentreMean(const Target& target, uint32_t size, uint32_t c)
{
    const uint32_t x0 = (target.width - size) / 2;
    const uint32_t y0 = (target.height - size) / 2;
    double sum = 0.0;
    for (uint32_t y = y0; y < y0 + size; ++y)
    {
        for (uint32_t x = x0; x < x0 + size; ++x)
        {
            sum += target.F32(x, y, c);
        }
    }
    return sum / (static_cast<double>(size) * size);
}

// The energy test scene, packed encoding: a grey floor of 64 x 64 x 8 blocks at
// y in [-8, 0) and an emissive ceiling of 64 x 64 x 8 blocks at y in [24, 32),
// both over x, z in [0, 64).
constexpr uint16_t kGreyFloor = 0x8000u | (15u << 10) | (15u << 5) | 15u;            // rgb5 15/31
constexpr uint16_t kWhiteEmitter = 0x4000u | (15u << 10) | (15u << 6) | (15u << 2);  // rgb4 1, strength 0
constexpr float kCeilingY = 24.0f;
constexpr float kRoomSize = 64.0f;
constexpr float kCameraHeight = 12.0f;

void UploadRoom(Rig& rig)
{
    World world;
    Brick floor{};
    floor.fill(kGreyFloor);
    Brick ceiling{};
    ceiling.fill(kWhiteEmitter);
    for (int32_t z = 0; z < 8; ++z)
    {
        for (int32_t x = 0; x < 8; ++x)
        {
            world.bricks[{x, -1, z}] = floor;
            world.bricks[{x, 3, z}] = ceiling;
        }
    }
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

// Camera at (32, 12, 32) looking straight down at the floor, vertical fov 30.
CaelixCamera LookDown()
{
    CaelixCamera camera{};
    camera.position[0] = 32.0f;
    camera.position[1] = kCameraHeight;
    camera.position[2] = 32.0f;
    camera.right[0] = 1.0f;
    camera.up[2] = 1.0f;
    camera.forward[1] = -1.0f;
    camera.tan_half_fov_y = std::tan(15.0f * kPi / 180.0f);
    camera.aspect = 1.0f;
    camera.near_z = kNear;
    camera.far_z = kFar;
    NormalZProjection(camera.projection, camera.tan_half_fov_y, camera.aspect);
    return camera;
}

// CPU Monte Carlo estimate of the probability that a cosine-distributed direction
// from a uniform point of the floor square [32 - h, 32 + h]^2 (y = 0) reaches the
// bottom face of the ceiling (y = 24, x and z in [0, 64)). With one bounce the GPU
// estimates albedo * emission * this probability.
double CeilingHitProbability(double half_extent, uint32_t samples)
{
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    uint64_t hits = 0;
    for (uint32_t i = 0; i < samples; ++i)
    {
        const double px = 32.0 + (uniform(rng) * 2.0 - 1.0) * half_extent;
        const double pz = 32.0 + (uniform(rng) * 2.0 - 1.0) * half_extent;
        const double r1 = uniform(rng);
        const double phi = 2.0 * 3.14159265358979323846 * uniform(rng);
        const double s = std::sqrt(r1);
        const double dx = s * std::cos(phi);
        const double dz = s * std::sin(phi);
        const double dy = std::sqrt(std::max(0.0, 1.0 - r1));
        if (dy <= 0.0)
        {
            continue;
        }
        const double t = kCeilingY / dy;
        const double x = px + dx * t;
        const double z = pz + dz * t;
        if (x >= 0.0 && x < kRoomSize && z >= 0.0 && z < kRoomSize)
        {
            ++hits;
        }
    }
    return static_cast<double>(hits) / samples;
}

}  // namespace

TEST_CASE("lighting: one bounce from an emissive ceiling matches a CPU Monte Carlo estimate")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadRoom(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LookDown();

    // Black sky, no sun: the only light is the ceiling, seen by the one bounce.
    const auto options_for = [](uint32_t mode, uint32_t seed) {
        CaelixRenderOptions options = LightOptions(mode, 1, seed);
        options.flags = 0;
        const float black[3] = {0.0f, 0.0f, 0.0f};
        Set3(options.sky_zenith, black);
        Set3(options.sky_horizon, black);
        Set3(options.sky_ground, black);
        return options;
    };

    // The floor point of a pixel edge at ndc n is 32 + n * tan_half_fov_y * 12: the
    // centre `size` pixels of the 256-pixel image cover a square of half extent
    // (size / 256) * tan_half_fov_y * 12 around (32, 32).
    const auto half_extent = [&](uint32_t size) {
        return static_cast<double>(size) / kSize * camera.tan_half_fov_y * kCameraHeight;
    };
    // The composite multiplies by the RGBA8 albedo target: 15/31 quantized to 8 bits.
    const double decoded_albedo = 15.0 / 31.0;
    const double albedo = std::round(decoded_albedo * 255.0) / 255.0;

    // 1. Reference mode, 2048 frames with a new seed each.
    constexpr uint32_t kFrames = 2048;
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        CaelixRenderOptions options = options_for(CAELIX_MODE_REFERENCE, 1000 + i);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, set, camera, options, i + 1 == kFrames);
    }
    const double gpu_ref = CentreMean(color, 32, 0);
    const double probability = CeilingHitProbability(half_extent(32), 200000);
    const double cpu_ref = albedo * probability;
    std::printf("energy, reference (%u frames, centre 32x32): GPU %.5f (g %.5f, b %.5f), CPU %.5f "
                "(hit probability %.5f, albedo %.5f, unquantized albedo %.5f -> %.5f), ratio %.4f\n",
                kFrames, gpu_ref, CentreMean(color, 32, 1), CentreMean(color, 32, 2), cpu_ref,
                probability, albedo, decoded_albedo, decoded_albedo * probability, gpu_ref / cpu_ref);
    CHECK(std::abs(gpu_ref / cpu_ref - 1.0) < 0.03);
    CHECK(std::abs(CentreMean(color, 32, 1) - gpu_ref) < 1e-6);
    CHECK(std::abs(CentreMean(color, 32, 2) - gpu_ref) < 1e-6);
    CHECK(color.F32(kSize / 2, kSize / 2, 3) == 1.0f);  // coverage

    // 2. One raw 1 spp frame. Every pixel is 0 or albedo (one Bernoulli sample), so
    //    the mean over 32x32 pixels has a standard deviation of about 2.6 percent; the
    //    5 percent check uses the centre 128x128 pixels (0.65 percent) against a CPU
    //    estimate of that region.
    Render(rig, renderer, set, camera, options_for(CAELIX_MODE_PATH_TRACE, 77));
    const double gpu_raw_32 = CentreMean(color, 32, 0);
    const double gpu_raw_128 = CentreMean(color, 128, 0);
    const double cpu_raw_128 = albedo * CeilingHitProbability(half_extent(128), 200000);
    std::printf("energy, 1 spp: GPU centre 32x32 %.5f (ratio %.4f to the 32x32 CPU estimate), GPU "
                "centre 128x128 %.5f, CPU %.5f, ratio %.4f\n",
                gpu_raw_32, gpu_raw_32 / cpu_ref, gpu_raw_128, cpu_raw_128, gpu_raw_128 / cpu_raw_128);
    CHECK(std::abs(gpu_raw_128 / cpu_raw_128 - 1.0) < 0.05);
    uint32_t other_values = 0;
    for (uint32_t y = 64; y < 192; ++y)
    {
        for (uint32_t x = 64; x < 192; ++x)
        {
            const float v = color.F32(x, y, 0);
            if (v != 0.0f && std::abs(v - static_cast<float>(albedo)) > 1e-6f)
            {
                ++other_values;
            }
        }
    }
    CHECK(other_values == 0);  // 1 spp, one bounce, emission 1: exactly 0 or albedo

    CaelixGpuTimings timings{};
    caelix_renderer_get_gpu_timings(renderer, &timings);
    std::printf("timings: trace %.4f ms, composite %.4f ms\n", timings.trace_ms, timings.composite_ms);
    CHECK(timings.composite_ms > 0.0f);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("lighting: modes, sky on miss, primary emission, sRGB output, argument checks")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadRedBrick(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);
    constexpr uint32_t cx = kSize / 2;
    constexpr uint32_t cy = kSize / 2;
    float corner_dir[3];
    PrimaryDirection(camera, 0, 0, kSize, kSize, corner_dir);
    float corner_sky[3];
    SkyRadiance(corner_dir, corner_sky);

    // 1. No bounce: the centre is the opaque red brick without emission = black,
    //    alpha 1; the corner is the sky radiance in its direction, alpha 0.
    CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 1);
    options.sun_radiance[0] = options.sun_radiance[1] = options.sun_radiance[2] = 50.0f;
    options.sun_dir[0] = 0.0f;  // straight up: not in the corner's direction
    options.sun_dir[1] = 1.0f;
    options.sun_dir[2] = 0.0f;
    Render(rig, renderer, set, camera, options);
    std::printf("pt, 0 bounces: centre %.5f %.5f %.5f %.5f, corner %.5f %.5f %.5f %.5f, expected sky "
                "%.5f %.5f %.5f\n",
                color.F32(cx, cy, 0), color.F32(cx, cy, 1), color.F32(cx, cy, 2), color.F32(cx, cy, 3),
                color.F32(0, 0, 0), color.F32(0, 0, 1), color.F32(0, 0, 2), color.F32(0, 0, 3),
                corner_sky[0], corner_sky[1], corner_sky[2]);
    for (uint32_t c = 0; c < 3; ++c)
    {
        CHECK(color.F32(cx, cy, c) == 0.0f);
        CHECK(std::abs(color.F32(0, 0, c) - corner_sky[c]) < 1e-4f);
    }
    CHECK(color.F32(cx, cy, 3) == 1.0f);
    CHECK(color.F32(0, 0, 3) == 0.0f);

    // The sun disk: aimed at the corner pixel, the corner is sky + sun radiance.
    {
        CaelixRenderOptions sun = options;
        Set3(sun.sun_dir, corner_dir);
        Render(rig, renderer, set, camera, sun);
        for (uint32_t c = 0; c < 3; ++c)
        {
            CHECK(std::abs(color.F32(0, 0, c) - (corner_sky[c] + 50.0f)) < 1e-3f);
        }
        // Pixel (8, 0) is about 1 degree away from the corner: sky only.
        float near_dir[3];
        PrimaryDirection(camera, 8, 0, kSize, kSize, near_dir);
        float near_sky[3];
        SkyRadiance(near_dir, near_sky);
        CHECK(std::abs(color.F32(8, 0, 0) - near_sky[0]) < 1e-4f);
        // With sun NEE the bounce rays lose the disk, but a primary miss still shows it.
        sun.flags |= CAELIX_RENDER_FLAG_SUN_NEE;
        Render(rig, renderer, set, camera, sun);
        for (uint32_t c = 0; c < 3; ++c)
        {
            CHECK(std::abs(color.F32(0, 0, c) - (corner_sky[c] + 50.0f)) < 1e-3f);
        }
    }

    // Exposure scales the colour, the sky included.
    options.exposure = 2.0f;
    Render(rig, renderer, set, camera, options);
    CHECK(std::abs(color.F32(0, 0, 0) - 2.0f * corner_sky[0]) < 2e-4f);
    options.exposure = 1.0f;

    // 2. Without CAELIX_RENDER_FLAG_SKY_ON_MISS a miss is (0, 0, 0, 0).
    options.flags = 0;
    Render(rig, renderer, set, camera, options);
    for (uint32_t c = 0; c < 4; ++c)
    {
        CHECK(color.F32(0, 0, c) == 0.0f);
    }
    CHECK(color.F32(cx, cy, 3) == 1.0f);

    // 3. A UNORM colour target receives sRGB-encoded values.
    {
        Target unorm = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
        TargetSet unorm_set;
        unorm_set.t[kColor] = &unorm;
        options.flags = CAELIX_RENDER_FLAG_SKY_ON_MISS;
        Render(rig, renderer, unorm_set, camera, options);
        const uint8_t* corner = unorm.at(0, 0);
        PrintPixel("pt, RGBA8 corner", corner);
        for (int c = 0; c < 3; ++c)
        {
            CHECK(std::abs(static_cast<int>(corner[c]) -
                           static_cast<int>(std::lround(LinearToSrgb(corner_sky[c]) * 255.0f))) <= 1);
        }
        CHECK(corner[3] == 0);
        CHECK(unorm.at(cx, cy)[3] == 255);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroyTarget(rig.gpu, unorm);
    }

    // 4. G-buffer targets in a lighting mode: the trace still writes them, and a
    //    lighting render without a colour target runs the trace only.
    {
        Target albedo = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
        TargetSet albedo_only;
        albedo_only.t[kAlbedo] = &albedo;
        Render(rig, renderer, albedo_only, camera, options);
        CHECK(albedo.at(cx, cy)[0] == 255);
        CHECK(albedo.at(cx, cy)[3] == 255);
        CHECK(albedo.at(0, 0)[3] == 0);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroyTarget(rig.gpu, albedo);
    }

    // 5. Primary emission: an emissive red brick (rgb4 (15, 0, 0), strength 0) is
    //    deterministic = albedo * 1 = (1, 0, 0) at the centre, with no bounce.
    {
        World world;
        Brick brick{};
        brick.fill(static_cast<uint16_t>(0x4000u | (15u << 10)));
        world.bricks[{0, 0, 0}] = brick;
        Upload(rig, world, KeysOf(world));
        DrainBuilds(rig);
    }
    Render(rig, renderer, set, camera, options);
    std::printf("pt, emissive brick, 0 bounces: centre %.5f %.5f %.5f %.5f\n", color.F32(cx, cy, 0),
                color.F32(cx, cy, 1), color.F32(cx, cy, 2), color.F32(cx, cy, 3));
    CHECK(color.F32(cx, cy, 0) == 1.0f);
    CHECK(color.F32(cx, cy, 1) == 0.0f);
    CHECK(color.F32(cx, cy, 2) == 0.0f);
    CHECK(color.F32(cx, cy, 3) == 1.0f);
    // With bounces the emitter only gains light.
    options.bounce_count = 4;
    Render(rig, renderer, set, camera, options);
    CHECK(color.F32(cx, cy, 0) >= 1.0f);

    // 6. Argument checks: an unknown mode and too many bounces.
    ExpectLibraryErrors(1, [&] {
        CaelixRenderOptions bad = LightOptions(3, 4, 0);
        CaelixFrameDesc desc{};
        desc.command_buffer = 1;
        desc.frame_number = rig.frames.next;
        const CaelixRenderTargets targets = ToAbi(set);
        REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &bad) ==
                CAELIX_ERROR_INVALID_ARGUMENT);
    });
    ExpectLibraryErrors(1, [&] {
        CaelixRenderOptions bad = LightOptions(CAELIX_MODE_PATH_TRACE, CAELIX_MAX_BOUNCES + 1, 0);
        CaelixFrameDesc desc{};
        desc.command_buffer = 1;
        desc.frame_number = rig.frames.next;
        const CaelixRenderTargets targets = ToAbi(set);
        REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &bad) ==
                CAELIX_ERROR_INVALID_ARGUMENT);
    });

    // 7. The primary mode still works on the same renderer (the colour is the debug view, opaque).
    Render(rig, renderer, set, camera, Options(CAELIX_VIEW_ALBEDO));
    CHECK(color.F32(cx, cy, 0) == 1.0f);
    CHECK(color.F32(0, 0, 3) == 1.0f);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("lighting: reference accumulation is the running mean of the raw frames")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadRedBrick(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    TargetSet set;
    set.t[kColor] = &color;
    // Off-centre camera: the brick's -Z and +X faces and the sky are in view.
    const CaelixCamera camera = LookPlusZ(10.0f, 6.0f, -14.0f);
    constexpr uint32_t kSeeds[3] = {7, 8, 9};
    const size_t floats = static_cast<size_t>(kSize) * kSize * 4;

    // Three raw frames.
    std::vector<float> raw[3];
    for (int i = 0; i < 3; ++i)
    {
        Render(rig, renderer, set, camera, LightOptions(CAELIX_MODE_PATH_TRACE, 2, kSeeds[i]));
        raw[i].resize(floats);
        std::memcpy(raw[i].data(), color.bytes.data(), floats * sizeof(float));
    }
    uint32_t differ = 0;
    uint32_t hits = 0;
    for (size_t i = 0; i < floats; i += 4)
    {
        hits += raw[0][i + 3] == 1.0f ? 1 : 0;
        differ += (raw[0][i] != raw[1][i]) ? 1 : 0;
    }
    std::printf("accumulation: %u hit pixels, %u differ between seeds 7 and 8\n", hits, differ);
    REQUIRE(hits > 1000);
    REQUIRE(differ > 1000);

    // The same frames in reference mode: the first after a reset is the raw frame
    // bit for bit, then the running mean.
    float max_error[3] = {0.0f, 0.0f, 0.0f};
    uint32_t first_mismatch = 0;
    for (int n = 0; n < 3; ++n)
    {
        CaelixRenderOptions options = LightOptions(CAELIX_MODE_REFERENCE, 2, kSeeds[n]);
        options.reset_accumulation = n == 0 ? 1u : 0u;
        Render(rig, renderer, set, camera, options);
        for (size_t i = 0; i < floats; ++i)
        {
            const float value = color.F32(static_cast<uint32_t>((i / 4) % kSize),
                                          static_cast<uint32_t>((i / 4) / kSize),
                                          static_cast<uint32_t>(i % 4));
            if (i % 4 == 3)
            {
                continue;  // coverage
            }
            double mean = 0.0;
            for (int k = 0; k <= n; ++k)
            {
                mean += raw[k][i];
            }
            mean /= (n + 1);
            if (n == 0 && value != raw[0][i])
            {
                ++first_mismatch;
            }
            max_error[n] = std::max(max_error[n], static_cast<float>(std::abs(value - mean) /
                                                                     std::max(1.0, std::abs(mean))));
        }
    }
    std::printf("accumulation: first frame %u mismatches, max relative error after 1/2/3 frames "
                "%.3g %.3g %.3g\n",
                first_mismatch, max_error[0], max_error[1], max_error[2]);
    CHECK(first_mismatch == 0);
    CHECK(max_error[1] < 1e-5f);
    CHECK(max_error[2] < 1e-5f);

    // reset_accumulation restarts: the next frame is the raw frame of its seed.
    CaelixRenderOptions options = LightOptions(CAELIX_MODE_REFERENCE, 2, kSeeds[1]);
    options.reset_accumulation = 1;
    Render(rig, renderer, set, camera, options);
    CHECK(std::memcmp(color.bytes.data(), raw[1].data(), floats * sizeof(float)) == 0);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

namespace {

// Sun NEE tests: a grey floor (the energy test's floor, no ceiling), black sky, the
// sun straight up with radiance 1000 and an angular radius of 2 degrees.
constexpr float kNeeSunRadiance = 1000.0f;
constexpr double kNeeSunRadiusDegrees = 2.0;

void UploadFloor(Rig& rig)
{
    World world;
    Brick floor{};
    floor.fill(kGreyFloor);
    for (int32_t z = 0; z < 8; ++z)
    {
        for (int32_t x = 0; x < 8; ++x)
        {
            world.bricks[{x, -1, z}] = floor;
        }
    }
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

// LookDown() moved to (x, height, z).
CaelixCamera LookDownFrom(float x, float height, float z)
{
    CaelixCamera camera = LookDown();
    camera.position[0] = x;
    camera.position[1] = height;
    camera.position[2] = z;
    return camera;
}

CaelixRenderOptions SunOptions(uint32_t mode, uint32_t bounces, uint32_t seed, bool nee)
{
    CaelixRenderOptions options = LightOptions(mode, bounces, seed);
    options.flags = nee ? CAELIX_RENDER_FLAG_SUN_NEE : 0u;
    const float black[3] = {0.0f, 0.0f, 0.0f};
    Set3(options.sky_zenith, black);
    Set3(options.sky_horizon, black);
    Set3(options.sky_ground, black);
    options.sun_dir[0] = 0.0f;
    options.sun_dir[1] = 1.0f;
    options.sun_dir[2] = 0.0f;
    options.sun_radiance[0] = options.sun_radiance[1] = options.sun_radiance[2] = kNeeSunRadiance;
    options.sun_angular_radius = static_cast<float>(kNeeSunRadiusDegrees * 3.14159265358979323846 / 180.0);
    return options;
}

// Reference mode over `frames` frames (seeds seed0 ..), the colour read back after the last.
void RenderReference(Rig& rig, CaelixRenderer* renderer, TargetSet& set, const CaelixCamera& camera,
                     uint32_t bounces, bool nee, uint32_t frames, uint32_t seed0)
{
    for (uint32_t i = 0; i < frames; ++i)
    {
        CaelixRenderOptions options = SunOptions(CAELIX_MODE_REFERENCE, bounces, seed0 + i, nee);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, set, camera, options, i + 1 == frames);
    }
}

// Largest |value - expected| / expected of channel 0 over the centre `size` x `size` pixels.
double CentreMaxRelativeError(const Target& target, uint32_t size, double expected)
{
    const uint32_t x0 = (target.width - size) / 2;
    const uint32_t y0 = (target.height - size) / 2;
    double worst = 0.0;
    for (uint32_t y = y0; y < y0 + size; ++y)
    {
        for (uint32_t x = x0; x < x0 + size; ++x)
        {
            worst = std::max(worst, std::abs(target.F32(x, y, 0) - expected) / expected);
        }
    }
    return worst;
}

// Largest channel-0 value over the centre `size` x `size` pixels.
float CentreMax(const Target& target, uint32_t size)
{
    const uint32_t x0 = (target.width - size) / 2;
    const uint32_t y0 = (target.height - size) / 2;
    float worst = 0.0f;
    for (uint32_t y = y0; y < y0 + size; ++y)
    {
        for (uint32_t x = x0; x < x0 + size; ++x)
        {
            worst = std::max(worst, target.F32(x, y, 0));
        }
    }
    return worst;
}

}  // namespace

TEST_CASE("lighting: sun next-event estimation: energy, no double counting, shadow")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadFloor(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    TargetSet set;
    set.t[kColor] = &color;

    // Expected floor radiance toward the camera: albedo * R * omega * E[cos] / pi, with
    // omega = 2 pi (1 - cos r) and E[cos] = (1 + cos r) / 2 for L uniform in the cone
    // (the floor normal is the sun direction). The composite multiplies by the RGBA8
    // albedo target (15/31 quantized to 123/255).
    const double radius = kNeeSunRadiusDegrees * 3.14159265358979323846 / 180.0;
    const double omega = 2.0 * 3.14159265358979323846 * (1.0 - std::cos(radius));
    const double mean_cos = 0.5 * (1.0 + std::cos(radius));
    const double albedo = std::round(15.0 / 31.0 * 255.0) / 255.0;
    const double irradiance_over_pi = kNeeSunRadiance * omega / 3.14159265358979323846;
    const double expected = albedo * irradiance_over_pi * mean_cos;
    std::printf("sun NEE: omega %.6g sr, R omega / pi %.5f, albedo %.5f, E[cos] %.6f, expected %.5f\n",
                omega, irradiance_over_pi, albedo, mean_cos, expected);
    const CaelixCamera camera = LookDown();

    // 1a. NEE on, 4 bounces (the host default), reference 256 frames: every sample
    //     sees the sun, so the result is almost noise-free.
    RenderReference(rig, renderer, set, camera, 4, true, 256, 5000);
    const double nee_mean = CentreMean(color, 64, 0);
    const double nee_worst = CentreMaxRelativeError(color, 64, expected);
    std::printf("sun NEE, 4 bounces, reference 256 frames: centre 64x64 mean %.5f (ratio %.5f), "
                "largest pixel error %.5f\n",
                nee_mean, nee_mean / expected, nee_worst);
    CHECK(std::abs(nee_mean / expected - 1.0) < 0.01);
    CHECK(nee_worst < 0.01);
    CHECK(std::abs(CentreMean(color, 64, 1) - nee_mean) < 1e-6);
    CHECK(std::abs(CentreMean(color, 64, 2) - nee_mean) < 1e-6);

    // 1b. NEE off, 1 bounce, reference 4096 frames: the sun is found only by the
    //     cosine-sampled bounce (probability sin^2 r = 0.12 percent), the same
    //     expected value with much more noise.
    RenderReference(rig, renderer, set, camera, 1, false, 4096, 20000);
    const double bsdf_mean = CentreMean(color, 64, 0);
    std::printf("sun NEE off, 1 bounce, reference 4096 frames: centre 64x64 mean %.5f (ratio %.5f to "
                "the expected value, %.5f to NEE)\n",
                bsdf_mean, bsdf_mean / expected, bsdf_mean / nee_mean);
    CHECK(std::abs(bsdf_mean / nee_mean - 1.0) < 0.10);

    // 3. Double-count guard: NEE on with 1 bounce. The bounce ray from the NEE vertex
    //    sees the sky without the sun disk (black here), so the result is test 1.
    RenderReference(rig, renderer, set, camera, 1, true, 256, 9000);
    const double one_bounce_mean = CentreMean(color, 64, 0);
    std::printf("sun NEE, 1 bounce, reference 256 frames: centre 64x64 mean %.5f (ratio %.5f to 4 "
                "bounces, %.5f to the expected value)\n",
                one_bounce_mean, one_bounce_mean / nee_mean, one_bounce_mean / expected);
    CHECK(std::abs(one_bounce_mean / nee_mean - 1.0) < 0.01);
    CHECK(std::abs(one_bounce_mean / expected - 1.0) < 0.01);
    // bounce_count 0 is primary emission only: no NEE either.
    Render(rig, renderer, set, camera, SunOptions(CAELIX_MODE_PATH_TRACE, 0, 3, true));
    CHECK(CentreMax(color, 64) == 0.0f);

    // 2. Shadow: a 16x16x16 opaque block 8 voxels above the floor centre (blocks
    //    x, z in [24, 40), y in [8, 24)). One bounce, so that light reflected by the
    //    block's underside cannot reach the floor below it.
    {
        World world;
        Brick block{};
        block.fill(kGreyFloor);
        for (int32_t z = 3; z <= 4; ++z)
        {
            for (int32_t y = 1; y <= 2; ++y)
            {
                for (int32_t x = 3; x <= 4; ++x)
                {
                    world.bricks[{x, y, z}] = block;
                }
            }
        }
        Upload(rig, world, KeysOf(world));
        DrainBuilds(rig);
    }
    // Under the block: camera at height 6 (below the block) over the floor centre;
    // the centre 64x64 pixels cover +-0.4 blocks around (32, 32), deep in the umbra.
    RenderReference(rig, renderer, set, LookDownFrom(32.0f, 6.0f, 32.0f), 1, true, 64, 30000);
    const double shadow_mean = CentreMean(color, 64, 0);
    const float shadow_max = CentreMax(color, 64);
    // Far from the block: around (8, 8), outside the block's footprint and penumbra.
    RenderReference(rig, renderer, set, LookDownFrom(8.0f, 6.0f, 8.0f), 1, true, 64, 31000);
    const double lit_mean = CentreMean(color, 64, 0);
    std::printf("sun NEE shadow: under the block centre 64x64 mean %.6f max %.6f; far from it mean "
                "%.5f (ratio %.5f to test 1)\n",
                shadow_mean, shadow_max, lit_mean, lit_mean / nee_mean);
    CHECK(shadow_max == 0.0f);
    CHECK(std::abs(lit_mean / nee_mean - 1.0) < 0.01);
    CHECK(color.F32(kSize / 2, kSize / 2, 3) == 1.0f);  // coverage

    CaelixGpuTimings timings{};
    caelix_renderer_get_gpu_timings(renderer, &timings);
    std::printf("sun NEE timings: trace %.4f ms, composite %.4f ms\n", timings.trace_ms,
                timings.composite_ms);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

// ---------------------------------------------------------------------------
// Step 6d: the material table on the GPU, the glass delta chain, extinction,
// total internal reflection, and the specular lobe of a host material table.

namespace {

// Packed emitters (rgb4, strength 1: emission = colour * 8).
constexpr uint16_t kRedEmitter8 = 0x4000u | (15u << 10) | 1u;
constexpr uint16_t kGreenEmitter8 = 0x4000u | (15u << 6) | 1u;
constexpr uint16_t kWhiteEmitter8 = 0x4000u | (15u << 10) | (15u << 6) | (15u << 2) | 1u;
constexpr uint16_t kClearGlass = 1;  // palette id 1: tint 1, ior 1.55, smoothness 1
constexpr uint16_t kAmber7 = 35;     // palette id 35: tint (0.98, 0.72, 0.32), extinction 0.032
constexpr double kGlassIor = 1.55;

int32_t FloorDiv8(int32_t v)
{
    return v >= 0 ? v / 8 : -((-v + 7) / 8);
}

// Sets every voxel of the box [lo, hi] (inclusive, entity-local block coordinates) to `value`,
// creating zero (air) bricks where needed.
void FillBox(World& world, int32_t x0, int32_t y0, int32_t z0, int32_t x1, int32_t y1, int32_t z1,
             uint16_t value)
{
    for (int32_t z = z0; z <= z1; ++z)
    {
        for (int32_t y = y0; y <= y1; ++y)
        {
            for (int32_t x = x0; x <= x1; ++x)
            {
                const Key key{FloorDiv8(x), FloorDiv8(y), FloorDiv8(z)};
                auto found = world.bricks.find(key);
                if (found == world.bricks.end())
                {
                    found = world.bricks.emplace(key, Brick{}).first;
                }
                const int32_t lx = x - key[0] * 8;
                const int32_t ly = y - key[1] * 8;
                const int32_t lz = z - key[2] * 8;
                found->second[static_cast<size_t>(lx + 8 * ly + 64 * lz)] = value;
            }
        }
    }
}

// A camera at `position` looking along `forward` (normalized here), right = Y x forward,
// up = forward x right (Unity's basis for a camera without roll).
CaelixCamera LookAlong(const double position[3], const double forward[3], double tan_half_fov_y)
{
    double f[3] = {forward[0], forward[1], forward[2]};
    const double fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    for (double& c : f)
    {
        c /= fl;
    }
    double r[3] = {f[2], 0.0, -f[0]};  // (0, 1, 0) x f
    const double rl = std::sqrt(r[0] * r[0] + r[2] * r[2]);
    r[0] /= rl;
    r[2] /= rl;
    const double u[3] = {f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]};
    CaelixCamera camera{};
    for (int c = 0; c < 3; ++c)
    {
        camera.position[c] = static_cast<float>(position[c]);
        camera.right[c] = static_cast<float>(r[c]);
        camera.up[c] = static_cast<float>(u[c]);
        camera.forward[c] = static_cast<float>(f[c]);
    }
    camera.tan_half_fov_y = static_cast<float>(tan_half_fov_y);
    camera.aspect = 1.0f;
    camera.near_z = kNear;
    camera.far_z = kFar;
    NormalZProjection(camera.projection, camera.tan_half_fov_y, camera.aspect);
    return camera;
}

// Lighting options with a black sky, no sky on miss, no sun.
CaelixRenderOptions GlassOptions(uint32_t mode, uint32_t bounces, uint32_t transparent, uint32_t seed)
{
    CaelixRenderOptions options = LightOptions(mode, bounces, seed);
    options.flags = 0;
    const float black[3] = {0.0f, 0.0f, 0.0f};
    Set3(options.sky_zenith, black);
    Set3(options.sky_horizon, black);
    Set3(options.sky_ground, black);
    options.bounce_count_transparent = transparent;
    return options;
}

void RenderGlassReference(Rig& rig, CaelixRenderer* renderer, TargetSet& set,
                          const CaelixCamera& camera, uint32_t bounces, uint32_t transparent,
                          uint32_t frames, uint32_t seed0)
{
    for (uint32_t i = 0; i < frames; ++i)
    {
        CaelixRenderOptions options = GlassOptions(CAELIX_MODE_REFERENCE, bounces, transparent, seed0 + i);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, set, camera, options, i + 1 == frames);
    }
}

// CPU port of FresnelReflectAmountTransparent (Unity Utils.hlsl, caelix_shading.hlsl).
// `incident` is unit length, `normal` faces against it.
double FresnelTransparent(double n1, double n2, const double incident[3], const double normal[3])
{
    double r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    double cos_x = -(normal[0] * incident[0] + normal[1] * incident[1] + normal[2] * incident[2]);
    if (n1 > n2)
    {
        const double n = n1 / n2;
        const double sin_t2 = n * n * (1.0 - cos_x * cos_x);
        if (sin_t2 >= 1.0)
        {
            return 1.0;
        }
        cos_x = std::sqrt(1.0 - sin_t2);
    }
    const double x = 1.0 - cos_x;
    return r0 + (1.0 - r0) * x * x * x * x * x;
}

// HLSL refract(i, n, eta); false for total internal reflection.
bool Refract(const double i[3], const double n[3], double eta, double out[3])
{
    const double cos_i = -(i[0] * n[0] + i[1] * n[1] + i[2] * n[2]);
    const double k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if (k < 0.0)
    {
        return false;
    }
    for (int c = 0; c < 3; ++c)
    {
        out[c] = eta * i[c] + (eta * cos_i - std::sqrt(k)) * n[c];
    }
    return true;
}

// The slab scene: a red emitter wall (z 16..23) behind a one-voxel clear glass pane (z = 4) that
// sits in air bricks (so the air cells on both sides of the pane exist and report the exit).
void UploadSlab(Rig& rig)
{
    World world;
    FillBox(world, -24, -16, 0, 31, 15, 7, 0);
    FillBox(world, -24, -16, 4, 31, 15, 4, kClearGlass);
    FillBox(world, -48, -24, 16, 47, 23, 23, kRedEmitter8);
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

struct GlassRig
{
    Rig rig;
    CaelixRenderer* renderer = nullptr;
    Target color;
    TargetSet set;
};

void OpenGlassRig(GlassRig& g, uint32_t encoding)
{
    OpenRig(g.rig, "caelix_trace_test", RigDesc(encoding, 0, 0));
    REQUIRE(caelix_renderer_create(g.rig.device, g.rig.scene, &g.renderer) == CAELIX_OK);
    g.color = CreateTarget(g.rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    g.set.t[kColor] = &g.color;
}

void CloseGlassRig(GlassRig& g)
{
    REQUIRE(vkDeviceWaitIdle(g.rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(g.renderer);
    caelix_renderer_destroy(g.renderer);
    DestroyTarget(g.rig.gpu, g.color);
    CloseRig(g.rig);
}

}  // namespace

TEST_CASE("glass: material table on the GPU, slab at normal incidence and at 60 degrees")
{
    GlassRig g;
    OpenGlassRig(g, 0);
    UploadSlab(g.rig);
    const double straight_pos[3] = {0.5, 0.5, -30.0};
    const double straight_dir[3] = {0.0, 0.0, 1.0};
    const CaelixCamera straight = LookAlong(straight_pos, straight_dir, std::tan(4.0 * kPi / 180.0));
    constexpr uint32_t cx = kSize / 2;
    constexpr uint32_t cy = kSize / 2;

    // 1. The material view reads the GPU table: clear glass (albedo 1) mixed with blue.
    {
        CaelixRenderOptions options = Options(CAELIX_VIEW_MATERIAL);
        Render(g.rig, g.renderer, g.set, straight, options);
        std::printf("material view, clear glass: %.5f %.5f %.5f\n", g.color.F32(cx, cy, 0),
                    g.color.F32(cx, cy, 1), g.color.F32(cx, cy, 2));
        CHECK(std::abs(g.color.F32(cx, cy, 0) - 0.6f) < 1e-5f);
        CHECK(std::abs(g.color.F32(cx, cy, 1) - 0.725f) < 1e-5f);
        CHECK(std::abs(g.color.F32(cx, cy, 2) - 1.0f) < 1e-5f);
        // A new palette entry for id 1 reaches the GPU at the next flush.
        CaelixMaterial original{};
        REQUIRE(caelix_scene_get_material(g.rig.scene, kClearGlass, &original) == CAELIX_OK);
        CaelixMaterial entries[2] = {original, original};
        entries[1].albedo[0] = entries[1].albedo[1] = entries[1].albedo[2] = 0.5f;
        REQUIRE(caelix_scene_set_glass_palette(g.rig.scene, entries, 2) == CAELIX_OK);
        Render(g.rig, g.renderer, g.set, straight, options);
        std::printf("material view, palette id 1 set to grey 0.5: %.5f %.5f %.5f\n",
                    g.color.F32(cx, cy, 0), g.color.F32(cx, cy, 1), g.color.F32(cx, cy, 2));
        CHECK(std::abs(g.color.F32(cx, cy, 0) - 0.35f) < 1e-5f);
        CHECK(std::abs(g.color.F32(cx, cy, 1) - 0.475f) < 1e-5f);
        CHECK(std::abs(g.color.F32(cx, cy, 2) - 0.75f) < 1e-5f);
        entries[1] = original;
        REQUIRE(caelix_scene_set_glass_palette(g.rig.scene, entries, 2) == CAELIX_OK);
    }

    // 2. Normal incidence, no stochastic bounce: the wall's emission (8, 0, 0) through two
    //    interfaces. Fresnel at each: F = ((1.55 - 1) / (1.55 + 1))^2. The chain chooses
    //    refraction with probability 1 - F, so the expected value is 8 (1 - F)^2 (1 + F^2 + ...):
    //    paths that reflect twice inside the pane and leave toward the wall add the F^2 term.
    const double f0 = std::pow((kGlassIor - 1.0) / (kGlassIor + 1.0), 2.0);
    {
        RenderGlassReference(g.rig, g.renderer, g.set, straight, 0, 5, 512, 100);
        const double measured = CentreMean(g.color, 32, 0);
        const double brief = 8.0 * (1.0 - f0) * (1.0 - f0);
        const double expected = brief * (1.0 + f0 * f0);
        std::printf("glass slab, normal incidence, reference 512 frames: centre 32x32 red %.5f, "
                    "expected 8 (1 - F)^2 = %.5f (ratio %.5f), with the F^2 term %.5f (ratio %.5f); "
                    "F = %.5f; green %.6f blue %.6f\n",
                    measured, brief, measured / brief, expected, measured / expected, f0,
                    CentreMean(g.color, 32, 1), CentreMean(g.color, 32, 2));
        CHECK(std::abs(measured / brief - 1.0) < 0.02);
        CHECK(std::abs(measured / expected - 1.0) < 0.01);
        CHECK(CentreMean(g.color, 32, 1) == 0.0);
        CHECK(CentreMean(g.color, 32, 2) == 0.0);
        CHECK(g.color.F32(cx, cy, 3) == 1.0f);
        // Without the chain the pane is a rough interface shaded stochastically; with no bounce it
        // contributes nothing (glass has no emission).
        RenderGlassReference(g.rig, g.renderer, g.set, straight, 0, 0, 4, 200);
        CHECK(CentreMax(g.color, 32) == 0.0f);
    }

    // 3. 60 degrees incidence (camera in the x-z plane). Expected 8 (1 - F1)(1 - F2)(1 + F1 F2),
    //    F1 entering (air -> glass) and F2 leaving (glass -> air, Schlick with the transmitted
    //    cosine), both from the shader's formula.
    {
        const double theta = 60.0 * 3.14159265358979323846 / 180.0;
        const double d[3] = {std::sin(theta), 0.0, std::cos(theta)};
        const double target[3] = {0.5, 0.5, 4.0};
        const double pos[3] = {target[0] - 40.0 * d[0], target[1], target[2] - 40.0 * d[2]};
        const CaelixCamera oblique = LookAlong(pos, d, std::tan(4.0 * kPi / 180.0));
        const double n[3] = {0.0, 0.0, -1.0};
        const double f1 = FresnelTransparent(1.0, kGlassIor, d, n);
        double inside[3];
        REQUIRE(Refract(d, n, 1.0 / kGlassIor, inside));
        const double f2 = FresnelTransparent(kGlassIor, 1.0, inside, n);
        RenderGlassReference(g.rig, g.renderer, g.set, oblique, 0, 5, 512, 1000);
        const double measured = CentreMean(g.color, 32, 0);
        const double brief = 8.0 * (1.0 - f1) * (1.0 - f2);
        const double expected = brief * (1.0 + f1 * f2);
        std::printf("glass slab, 60 degrees, reference 512 frames: centre 32x32 red %.5f, expected "
                    "8 (1 - F1)(1 - F2) = %.5f (ratio %.5f), with the F1 F2 term %.5f (ratio %.5f); "
                    "F1 = %.5f, F2 = %.5f\n",
                    measured, brief, measured / brief, expected, measured / expected, f1, f2);
        CHECK(std::abs(measured / brief - 1.0) < 0.03);
    }

    // Argument check: more than CAELIX_MAX_BOUNCES_TRANSPARENT interfaces.
    ExpectLibraryErrors(1, [&] {
        CaelixFrameDesc desc{};
        desc.command_buffer = 1;
        desc.frame_number = g.rig.frames.next;
        const CaelixRenderTargets targets = ToAbi(g.set);
        CaelixRenderOptions bad =
            GlassOptions(CAELIX_MODE_PATH_TRACE, 0, CAELIX_MAX_BOUNCES_TRANSPARENT + 1, 0);
        REQUIRE(caelix_renderer_render(g.renderer, &desc, &targets, &straight, &bad) ==
                CAELIX_ERROR_INVALID_ARGUMENT);
    });
    CloseGlassRig(g);
}

TEST_CASE("glass: extinction through an 8-voxel amber block")
{
    GlassRig g;
    OpenGlassRig(g, 0);
    {
        World world;
        FillBox(world, -16, -16, 0, 15, 15, 15, 0);
        FillBox(world, -16, -16, 4, 15, 15, 11, kAmber7);
        FillBox(world, -32, -32, 24, 31, 31, 31, kWhiteEmitter8);
        Upload(g.rig, world, KeysOf(world));
        DrainBuilds(g.rig);
    }
    const double pos[3] = {0.5, 0.5, -30.0};
    const double dir[3] = {0.0, 0.0, 1.0};
    const CaelixCamera camera = LookAlong(pos, dir, std::tan(4.0 * kPi / 180.0));
    RenderGlassReference(g.rig, g.renderer, g.set, camera, 0, 5, 512, 3000);
    const double f = std::pow((kGlassIor - 1.0) / (kGlassIor + 1.0), 2.0);
    const double tint[3] = {0.98, 0.72, 0.32};
    for (uint32_t c = 0; c < 3; ++c)
    {
        const double transmission = std::exp(-(1.0 - tint[c]) * 8.0 * 0.032);
        const double brief = 8.0 * transmission * (1.0 - f) * (1.0 - f);
        // Two internal reflections add F^2 and two more passes through the block.
        const double expected = brief * (1.0 + f * f * transmission * transmission);
        const double measured = CentreMean(g.color, 32, c);
        std::printf("amber-7, 8 voxels, channel %u: measured %.5f, expected 8 T (1 - F)^2 = %.5f "
                    "(ratio %.5f), with the F^2 term %.5f (ratio %.5f); T = %.5f\n",
                    c, measured, brief, measured / brief, expected, measured / expected, transmission);
        CHECK(std::abs(measured / brief - 1.0) < 0.02);
    }
    CloseGlassRig(g);
}

TEST_CASE("glass: total internal reflection at a side face")
{
    // A 16^3 clear glass block (4..19 on every axis) in air bricks, a red emitter floor below it
    // (y -16..-9) and a green emitter wall beyond its +X face (x 32..39). The camera looks down at
    // 60 degrees from the vertical onto the top face near the +X edge: the refracted ray (34 degrees)
    // meets the +X face at 56 degrees, past the critical angle (40 degrees): it must reflect, leave
    // through the bottom face and reach the red floor. A refraction through the side face would
    // show the green wall.
    GlassRig g;
    OpenGlassRig(g, 0);
    {
        World world;
        FillBox(world, 0, 0, 0, 23, 23, 23, 0);
        FillBox(world, 4, 4, 4, 19, 19, 19, kClearGlass);
        FillBox(world, -24, -16, 0, 23, -9, 23, kRedEmitter8);
        FillBox(world, 32, -16, 0, 39, 23, 23, kGreenEmitter8);
        Upload(g.rig, world, KeysOf(world));
        DrainBuilds(g.rig);
    }
    const double theta = 60.0 * 3.14159265358979323846 / 180.0;
    const double d[3] = {std::sin(theta), -std::cos(theta), 0.0};
    const double target[3] = {17.0, 20.0, 12.0};
    const double pos[3] = {target[0] - 40.0 * d[0], target[1] - 40.0 * d[1], target[2]};
    const CaelixCamera camera = LookAlong(pos, d, std::tan(2.0 * kPi / 180.0));
    RenderGlassReference(g.rig, g.renderer, g.set, camera, 0, 5, 512, 4000);
    // Entering at 60 degrees and leaving the bottom at 34 degrees inside have the same Fresnel term.
    const double n_top[3] = {0.0, 1.0, 0.0};
    const double f = FresnelTransparent(1.0, kGlassIor, d, n_top);
    const double brief = 8.0 * (1.0 - f) * (1.0 - f);
    const double expected = brief * (1.0 + f * f);
    const double red = CentreMean(g.color, 32, 0);
    const double green = CentreMean(g.color, 32, 1);
    std::printf("total internal reflection: centre 32x32 red %.5f, expected 8 (1 - F)^2 = %.5f "
                "(ratio %.5f), with the F^2 term %.5f (ratio %.5f), F = %.5f; green %.6f (no "
                "refraction through the side face)\n",
                red, brief, red / brief, expected, red / expected, f, green);
    CHECK(std::abs(red / brief - 1.0) < 0.03);
    CHECK(green < 0.01 * 8.0);
    CloseGlassRig(g);
}

TEST_CASE("specular: a smooth metal floor from a host table mirrors an emitter; NRD spec channel")
{
    // Block-ID scene: id 1 (code 0x8001) = smooth metal (albedo 0.9, smoothness 1, metallic 1),
    // id 2 (code 0x8002) = white emitter (emission 4). Floor y -8..-1, emitter cube x -8..7,
    // y 12..19, z 16..23. The camera at (0, 16, -16) looks down at 45 degrees at (0, 0, 0): the
    // mirror direction (0, 1, 1) / sqrt 2 meets the cube's -Z face at y = 16.
    GlassRig g;
    OpenGlassRig(g, 1);
    {
        const uint32_t count = 0x8003;
        std::vector<CaelixMaterial> table(count);
        for (uint32_t code = 0; code < count; ++code)
        {
            REQUIRE(caelix_scene_get_material(g.rig.scene, code, &table[code]) == CAELIX_OK);
        }
        CaelixMaterial mirror{};
        mirror.albedo[0] = mirror.albedo[1] = mirror.albedo[2] = 0.9f;
        mirror.smoothness = 1.0f;
        mirror.metallic = 1.0f;
        mirror.ior = 1.5f;
        CaelixMaterial emitter{};
        emitter.albedo[0] = emitter.albedo[1] = emitter.albedo[2] = 1.0f;
        emitter.emission[0] = emitter.emission[1] = emitter.emission[2] = 4.0f;
        emitter.ior = 1.5f;
        table[0x8001] = mirror;
        table[0x8002] = emitter;
        REQUIRE(caelix_scene_set_materials(g.rig.scene, table.data(), count) == CAELIX_OK);
        World world;
        FillBox(world, -32, -8, -32, 31, -1, 31, 0x8001);
        FillBox(world, -8, 12, 16, 7, 19, 23, 0x8002);
        Upload(g.rig, world, KeysOf(world));
        DrainBuilds(g.rig);
    }
    const double pos[3] = {0.0, 16.0, -16.0};
    const double dir[3] = {0.0, -1.0, 1.0};
    const CaelixCamera camera = LookAlong(pos, dir, std::tan(30.0 * kPi / 180.0));
    constexpr uint32_t cx = kSize / 2;
    constexpr uint32_t cy = kSize / 2;
    // The composite multiplies by the RGBA8 albedo target: 0.9 -> 230/255.
    const double expected = std::round(0.9 * 255.0) / 255.0 * 4.0;
    RenderGlassReference(g.rig, g.renderer, g.set, camera, 1, 5, 16, 5000);
    const double centre = CentreMean(g.color, 8, 0);
    const float beside = g.color.F32(4, cy, 0);  // far left: the mirror sees the black sky
    std::printf("mirror floor, reference 16 frames: centre 8x8 %.5f (expected %.5f, ratio %.5f), "
                "far left %.5f\n",
                centre, expected, centre / expected, beside);
    CHECK(std::abs(centre / expected - 1.0) < 0.01);
    CHECK(beside < 0.05f);

    // Path trace mode with NRD: the radiance is in the specular channel, not the diffuse one.
    float spec = 0.0f;
    float diff = 0.0f;
    for (uint32_t view : {static_cast<uint32_t>(CAELIX_VIEW_NRD_SPEC),
                          static_cast<uint32_t>(CAELIX_VIEW_NRD_DIFF)})
    {
        for (uint32_t i = 0; i < 16; ++i)
        {
            CaelixRenderOptions options = GlassOptions(CAELIX_MODE_PATH_TRACE, 1, 5, 6000 + i);
            options.denoise = 1;
            options.debug_view = view;
            Render(g.rig, g.renderer, g.set, camera, options, i + 1 == 16);
        }
        (view == CAELIX_VIEW_NRD_SPEC ? spec : diff) = g.color.F32(cx, cy, 0);
    }
    CaelixNrdInfo info{};
    REQUIRE(caelix_renderer_get_nrd_info(g.renderer, &info) == CAELIX_OK);
    std::printf("mirror floor, NRD after 16 frames: specular view %.5f, diffuse view %.5f (NRD "
                "failed %u)\n",
                spec, diff, info.failed);
    CHECK(info.failed == 0);
    CHECK(spec > 2.0f);
    CHECK(diff < 0.05f * spec);
    CloseGlassRig(g);
}

TEST_CASE("glass: delta checkerboard is stable per pixel; the cross resolve matches the reference")
{
    // (1) A clear glass block (z 4..11) lying directly on a red emitter (z 12..19): the camera ray
    //     has exactly one interface (entering the glass; the glass touches the emitter, so there is
    //     no exit interface), so with the checkerboard nothing in the image is random.
    GlassRig g;
    OpenGlassRig(g, 0);
    {
        World world;
        FillBox(world, -16, -16, 0, 15, 15, 11, 0);
        FillBox(world, -16, -16, 4, 15, 15, 11, kClearGlass);
        FillBox(world, -24, -24, 12, 23, 23, 19, kRedEmitter8);
        Upload(g.rig, world, KeysOf(world));
        DrainBuilds(g.rig);
    }
    Target albedo = CreateTarget(g.rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
    Target motion = CreateTarget(g.rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    TargetSet set = g.set;
    set.t[kAlbedo] = &albedo;
    set.t[kMotion] = &motion;
    const double pos[3] = {0.5, 0.5, -30.0};
    const double dir[3] = {0.0, 0.0, 1.0};
    const CaelixCamera camera = LookAlong(pos, dir, std::tan(4.0 * kPi / 180.0));
    constexpr uint32_t cx = kSize / 2;  // even px + py: refraction
    constexpr uint32_t cy = kSize / 2;
    const double f0 = std::pow((kGlassIor - 1.0) / (kGlassIor + 1.0), 2.0);
    const auto options_for = [](uint32_t seed, bool checkerboard) {
        CaelixRenderOptions options = GlassOptions(CAELIX_MODE_PATH_TRACE, 0, 5, seed);
        if (checkerboard)
        {
            options.flags |= CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD;
        }
        return options;
    };

    // Checkerboard on, NRD off, no stochastic bounce, 10 frames with new seeds: bit-identical.
    std::vector<uint8_t> first;
    uint32_t differing_frames = 0;
    for (uint32_t i = 0; i < 10; ++i)
    {
        Render(g.rig, g.renderer, set, camera, options_for(700 + i, true));
        if (i == 0)
        {
            first = g.color.bytes;
        }
        else if (g.color.bytes != first)
        {
            ++differing_frames;
        }
    }
    // The parities trace different surfaces: even pixels refract into the emitter (albedo target =
    // the emitter, red), odd pixels reflect into the black sky (the chain ends after the glass
    // interface: albedo target = the glass tint). Both are flagged in the motion target's w.
    const uint8_t* even_albedo = albedo.at(cx, cy);
    const uint8_t* odd_albedo = albedo.at(cx + 1, cy);
    const double resolved = g.color.F32(cx, cy, 0);
    const double resolved_odd = g.color.F32(cx + 1, cy, 0);
    const double expected = 8.0 * (1.0 - f0);  // 0.5 * 2 (1 - F) * 8 + 0.5 * 0
    std::printf("checkerboard, 10 frames: %u differ from the first; albedo even (%u %u %u) odd (%u %u "
                "%u); flags %.1f %.1f; resolved centre %.5f, odd neighbour %.5f, expected (1 - F) 8 = "
                "%.5f\n",
                differing_frames, even_albedo[0], even_albedo[1], even_albedo[2], odd_albedo[0],
                odd_albedo[1], odd_albedo[2], motion.F32(cx, cy, 3), motion.F32(cx + 1, cy, 3),
                resolved, resolved_odd, expected);
    CHECK(differing_frames == 0);
    CHECK(Near(even_albedo, 1.0f, 0.0f, 0.0f, 1));
    CHECK(Near(odd_albedo, 0.6f, 0.8f, 1.0f, 1));
    CHECK(motion.F32(cx, cy, 3) == 1.0f);
    CHECK(motion.F32(cx + 1, cy, 3) == 1.0f);
    CHECK(std::abs(resolved / expected - 1.0) < 1e-3);
    CHECK(std::abs(resolved_odd / expected - 1.0) < 1e-3);
    CHECK(CentreMean(g.color, 32, 1) == 0.0);
    // Without the checkerboard the same frames are random (Fresnel choice per frame and pixel), and
    // no pixel is flagged.
    Render(g.rig, g.renderer, set, camera, options_for(700, false));
    const std::vector<uint8_t> random_a = g.color.bytes;
    Render(g.rig, g.renderer, set, camera, options_for(701, false));
    CHECK(g.color.bytes != random_a);
    CHECK(motion.F32(cx, cy, 3) == 0.0f);
    DestroyTarget(g.rig.gpu, albedo);
    DestroyTarget(g.rig.gpu, motion);
    CloseGlassRig(g);

    // (2) The slab (two interfaces, the second one still random): path trace mode with the
    //     checkerboard and the cross resolve, the mean of 128 frames over the centre 32x32, against
    //     the stochastic reference (512 frames).
    GlassRig s;
    OpenGlassRig(s, 0);
    UploadSlab(s.rig);
    RenderGlassReference(s.rig, s.renderer, s.set, camera, 0, 5, 512, 9000);
    const double reference = CentreMean(s.color, 32, 0);
    double sum = 0.0;
    constexpr uint32_t kFrames = 128;
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        CaelixRenderOptions options = GlassOptions(CAELIX_MODE_PATH_TRACE, 0, 5, 9600 + i);
        options.flags |= CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD;
        Render(s.rig, s.renderer, s.set, camera, options);
        sum += CentreMean(s.color, 32, 0);
    }
    const double checkerboard_mean = sum / kFrames;
    std::printf("slab: checkerboard + cross resolve, mean of %u frames, centre 32x32 %.5f; stochastic "
                "reference %.5f; ratio %.5f\n",
                kFrames, checkerboard_mean, reference, checkerboard_mean / reference);
    CHECK(std::abs(checkerboard_mean / reference - 1.0) < 0.02);
    CloseGlassRig(s);
}

namespace {

// The six geometry and colour targets of one renderer for the flip test.
struct FlipTargets
{
    Target color, albedo, normal, depth, motion, view_z;
    TargetSet set;
    void Create(Gpu& gpu, VkFormat color_format)
    {
        color = CreateTarget(gpu, color_format);
        albedo = CreateTarget(gpu, VK_FORMAT_R8G8B8A8_UNORM);
        normal = CreateTarget(gpu, VK_FORMAT_R16G16B16A16_SFLOAT);
        depth = CreateTarget(gpu, VK_FORMAT_R32_SFLOAT);
        motion = CreateTarget(gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
        view_z = CreateTarget(gpu, VK_FORMAT_R32_SFLOAT);
        set.t[kColor] = &color;
        set.t[kAlbedo] = &albedo;
        set.t[kNormal] = &normal;
        set.t[kDepth] = &depth;
        set.t[kMotion] = &motion;
        set.t[kViewZ] = &view_z;
    }
    void Destroy(Gpu& gpu)
    {
        for (Target* t : {&color, &albedo, &normal, &depth, &motion, &view_z})
        {
            DestroyTarget(gpu, *t);
        }
    }
};

// Rows of `flipped` that differ from the rows of `plain` mirrored in y (byte compare).
uint32_t MirrorMismatchRows(const Target& plain, const Target& flipped)
{
    REQUIRE(plain.width == flipped.width);
    REQUIRE(plain.height == flipped.height);
    const size_t row_bytes = static_cast<size_t>(plain.width) * plain.bytes_per_pixel;
    uint32_t rows = 0;
    for (uint32_t y = 0; y < plain.height; ++y)
    {
        if (std::memcmp(plain.at(0, y), flipped.at(0, plain.height - 1 - y), row_bytes) != 0)
        {
            ++rows;
        }
    }
    return rows;
}

// Pixels of an RGBA32F motion target whose flipped value is not (x, -y, z, w) of the plain one.
uint32_t MotionMirrorMismatches(const Target& plain, const Target& flipped)
{
    uint32_t bad = 0;
    for (uint32_t y = 0; y < plain.height; ++y)
    {
        for (uint32_t x = 0; x < plain.width; ++x)
        {
            const uint32_t fy = plain.height - 1 - y;
            if (flipped.F32(x, fy, 0) != plain.F32(x, y, 0) || flipped.F32(x, fy, 1) != -plain.F32(x, y, 1) ||
                flipped.F32(x, fy, 2) != plain.F32(x, y, 2) || flipped.F32(x, fy, 3) != plain.F32(x, y, 3))
            {
                ++bad;
            }
        }
    }
    return bad;
}

// Leftmost hit column of row y, top hit row of column x (albedo coverage); the size when none.
uint32_t FirstHitColumn(const Target& albedo, uint32_t y)
{
    for (uint32_t x = 0; x < albedo.width; ++x)
    {
        if (albedo.at(x, y)[3] == 255) return x;
    }
    return albedo.width;
}
uint32_t FirstHitRow(const Target& albedo, uint32_t x)
{
    for (uint32_t y = 0; y < albedo.height; ++y)
    {
        if (albedo.at(x, y)[3] == 255) return y;
    }
    return albedo.height;
}

}  // namespace

TEST_CASE("flip y: host targets mirrored, motion y negated; jitter moves the sample, not the motion")
{
    constexpr uint32_t cx = kSize / 2;
    constexpr uint32_t cy = kSize / 2;

    // (1) Two renderers on one scene, the same cameras and options; B sets CAELIX_RENDER_FLAG_FLIP_Y.
    {
        Rig rig;
        OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
        UploadRedBrick(rig);
        CaelixRenderer* plain = nullptr;
        CaelixRenderer* flipped = nullptr;
        REQUIRE(caelix_renderer_create(rig.device, rig.scene, &plain) == CAELIX_OK);
        REQUIRE(caelix_renderer_create(rig.device, rig.scene, &flipped) == CAELIX_OK);
        FlipTargets a;
        FlipTargets b;
        a.Create(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
        b.Create(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
        const auto render_both = [&](const CaelixCamera& camera, CaelixRenderOptions options) {
            Render(rig, plain, a.set, camera, options);
            options.flags |= CAELIX_RENDER_FLAG_FLIP_Y;
            Render(rig, flipped, b.set, camera, options);
        };
        // Off-centre camera, so the image is not symmetric in y; then the camera moves +1 up.
        const CaelixCamera c0 = LookPlusZ(2.0f, 5.0f, -20.0f);
        const CaelixCamera c1 = LookPlusZ(2.0f, 6.0f, -20.0f);
        render_both(c0, Options(CAELIX_VIEW_ALBEDO));
        render_both(c1, Options(CAELIX_VIEW_ALBEDO));
        const uint32_t mismatch_color = MirrorMismatchRows(a.color, b.color);
        const uint32_t mismatch_albedo = MirrorMismatchRows(a.albedo, b.albedo);
        const uint32_t mismatch_normal = MirrorMismatchRows(a.normal, b.normal);
        const uint32_t mismatch_depth = MirrorMismatchRows(a.depth, b.depth);
        const uint32_t mismatch_view_z = MirrorMismatchRows(a.view_z, b.view_z);
        const uint32_t mismatch_motion = MotionMirrorMismatches(a.motion, b.motion);
        // A camera that moved +1 up sees a static point LOWER on the screen (larger y, rows go
        // down): the previous pixel is above, motion.y = -pixels per unit. Flipped: +.
        const float view_z = a.view_z.F32(cx, cy, 0);
        const float pixels_per_unit = static_cast<float>(kSize) / (2.0f * c1.tan_half_fov_y * view_z);
        std::printf("flip, camera +1 up: mismatched rows colour %u albedo %u normal %u depth %u view_z %u, "
                    "motion pixels %u; centre motion plain %.5f %.5f, flipped %.5f %.5f, expected 0 %.5f\n",
                    mismatch_color, mismatch_albedo, mismatch_normal, mismatch_depth, mismatch_view_z,
                    mismatch_motion, a.motion.F32(cx, cy, 0), a.motion.F32(cx, cy, 1),
                    b.motion.F32(cx, kSize - 1 - cy, 0), b.motion.F32(cx, kSize - 1 - cy, 1), -pixels_per_unit);
        REQUIRE(a.albedo.at(cx, cy)[3] == 255);
        CHECK(MirrorMismatchRows(a.albedo, a.albedo) > 0);  // the image is not symmetric in y
        CHECK(mismatch_color == 0);
        CHECK(mismatch_albedo == 0);
        CHECK(mismatch_normal == 0);
        CHECK(mismatch_depth == 0);
        CHECK(mismatch_view_z == 0);
        CHECK(mismatch_motion == 0);
        CHECK(std::abs(a.motion.F32(cx, cy, 0)) < 1e-3f);
        CHECK(std::abs(a.motion.F32(cx, cy, 1) + pixels_per_unit) < 1e-2f);
        CHECK(std::abs(b.motion.F32(cx, kSize - 1 - cy, 1) - pixels_per_unit) < 1e-2f);

        // Path trace mode (one bounce, sky on miss, no NRD): the composite's output is mirrored too.
        CaelixRenderOptions pt = LightOptions(CAELIX_MODE_PATH_TRACE, 1, 4242);
        pt.flags |= CAELIX_RENDER_FLAG_SKY_ON_MISS;
        render_both(c1, pt);
        const uint32_t mismatch_pt = MirrorMismatchRows(a.color, b.color);
        // The motion view shows the host motion's convention: green = saturate(motion.y / 64 + 0.5)
        // with motion.y negated when flipped, so the two greens sum to about 255. The camera moves
        // down 1 here: motion.y = +pixels per unit in the plain image.
        CaelixRenderOptions motion_view = pt;
        motion_view.debug_view = CAELIX_VIEW_MOTION;
        render_both(c0, motion_view);
        const int green_sum = a.color.at(cx, cy)[1] + b.color.at(cx, kSize - 1 - cy)[1];
        std::printf("flip, path trace: %u mismatched colour rows; motion view green plain %u flipped %u\n",
                    mismatch_pt, a.color.at(cx, cy)[1], b.color.at(cx, kSize - 1 - cy)[1]);
        CHECK(mismatch_pt == 0);
        CHECK(a.color.at(cx, cy)[1] > 140);
        CHECK(std::abs(green_sum - 255) <= 1);

        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(plain);
        caelix_renderer_release_target_views(flipped);
        caelix_renderer_destroy(plain);
        caelix_renderer_destroy(flipped);
        a.Destroy(rig.gpu);
        b.Destroy(rig.gpu);
        CloseRig(rig);
    }

    // (2) The delta checkerboard and its cross resolve under the flip: the glass slab, path trace
    //     mode, checkerboard on. The resolve taps read unflipped internal images; the output is the
    //     plain image mirrored, bit for bit.
    {
        GlassRig g;
        OpenGlassRig(g, 0);
        UploadSlab(g.rig);
        CaelixRenderer* flipped = nullptr;
        REQUIRE(caelix_renderer_create(g.rig.device, g.rig.scene, &flipped) == CAELIX_OK);
        Target flipped_color = CreateTarget(g.rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
        TargetSet flipped_set;
        flipped_set.t[kColor] = &flipped_color;
        const double pos[3] = {0.5, 0.5, -30.0};
        const double dir[3] = {0.0, 0.0, 1.0};
        const CaelixCamera camera = LookAlong(pos, dir, std::tan(4.0 * kPi / 180.0));
        CaelixRenderOptions options = GlassOptions(CAELIX_MODE_PATH_TRACE, 1, 5, 777);
        options.flags |= CAELIX_RENDER_FLAG_DELTA_CHECKERBOARD;
        Render(g.rig, g.renderer, g.set, camera, options);
        options.flags |= CAELIX_RENDER_FLAG_FLIP_Y;
        Render(g.rig, flipped, flipped_set, camera, options);
        const uint32_t mismatch = MirrorMismatchRows(g.color, flipped_color);
        std::printf("flip, glass checkerboard + cross resolve: %u mismatched rows, centre %.5f\n", mismatch,
                    g.color.F32(cx, cy, 0));
        CHECK(mismatch == 0);
        CHECK(g.color.F32(cx, cy, 0) > 0.0f);
        REQUIRE(vkDeviceWaitIdle(g.rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(flipped);
        caelix_renderer_destroy(flipped);
        DestroyTarget(g.rig.gpu, flipped_color);
        CloseGlassRig(g);
    }

    // (3) Jitter. Camera (4, 4, -20) looking +Z at the red brick [0, 8)^3: the brick's left face
    //     projects to pixel x = (1 - 4 / (20 tan 30)) / 2 * 256 = 83.66, its top face to y = 83.66
    //     (rows down). A pixel hits when its sample position px + 0.5 + jitter.x >= 83.66, so the
    //     leftmost hit column is ceil(83.16 - jitter.x): 84 at jitter 0, 83 at jitter 0.4; likewise
    //     the top hit row with jitter.y (y down). The motion stays the surface motion: 0 for a static
    //     camera whatever the jitter.
    {
        Rig rig;
        OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
        UploadRedBrick(rig);
        CaelixRenderer* renderer = nullptr;
        REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
        FlipTargets t;
        t.Create(rig.gpu, VK_FORMAT_R16G16B16A16_SFLOAT);
        CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);
        Render(rig, renderer, t.set, camera, Options(CAELIX_VIEW_ALBEDO));
        const uint32_t column0 = FirstHitColumn(t.albedo, cy);
        const uint32_t row0 = FirstHitRow(t.albedo, cx);
        camera.jitter[0] = 0.4f;
        camera.jitter[1] = 0.4f;
        Render(rig, renderer, t.set, camera, Options(CAELIX_VIEW_ALBEDO));
        const uint32_t column1 = FirstHitColumn(t.albedo, cy);
        const uint32_t row1 = FirstHitRow(t.albedo, cx);
        float max_motion = 0.0f;
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                if (t.albedo.at(x, y)[3] == 255)
                {
                    max_motion = std::max(max_motion, std::max(std::abs(t.motion.F32(x, y, 0)),
                                                               std::abs(t.motion.F32(x, y, 1))));
                }
            }
        }
        std::printf("jitter 0 -> (0.4, 0.4): leftmost hit column %u -> %u, top hit row %u -> %u, max "
                    "|motion| on hits %.6f px\n",
                    column0, column1, row0, row1, max_motion);
        CHECK(column0 == 84);
        CHECK(column1 == 83);
        CHECK(row0 == 84);
        CHECK(row1 == 83);
        CHECK(max_motion < 1e-3f);
        // NRD accepts a jitter that changes every frame (cameraJitter / cameraJitterPrev).
        for (uint32_t i = 0; i < 4; ++i)
        {
            camera.jitter[0] = (i & 1u) != 0 ? 0.25f : -0.25f;
            camera.jitter[1] = (i & 2u) != 0 ? 0.25f : -0.25f;
            CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 1, 50 + i);
            options.denoise = 1;
            Render(rig, renderer, t.set, camera, options, i == 3);
        }
        CaelixNrdInfo nrd{};
        REQUIRE(caelix_renderer_get_nrd_info(renderer, &nrd) == CAELIX_OK);
        CHECK(nrd.failed == 0);
        CHECK(nrd.dispatch_count > 0);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        caelix_renderer_destroy(renderer);
        t.Destroy(rig.gpu);
        CloseRig(rig);
    }
}

// ---------------------------------------------------------------------------
// Sky cubemap, fog and emission scale (CaelixRenderOptions::sky, fog_*, emission_scale)

namespace {

// IEEE half of a float that is a normal half value (the test's exact radiances).
uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    if (value == 0.0f || exponent <= 0)
    {
        return static_cast<uint16_t>(sign);
    }
    REQUIRE(exponent < 31);
    const uint32_t mantissa = (bits >> 13) & 0x3FFu;
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

// A host sky cubemap: RGBA16F (linear filtering is a required feature of the format),
// `size` x `size` faces, 1 mip, texel = radiance(direction of the texel centre), left in
// SHADER_READ_ONLY_OPTIMAL.
struct SkyCube
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t size = 0;

    CaelixSkyCubemap Abi() const
    {
        CaelixSkyCubemap out{};
        out.image = reinterpret_cast<uint64_t>(image);
        out.format = static_cast<uint32_t>(VK_FORMAT_R16G16B16A16_SFLOAT);
        out.layout = static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        out.size = size;
        out.mip_levels = 1;
        return out;
    }
};

// The world direction (not normalized) of the texel centre (s, t in [-1, 1], t down the face)
// of cube face `face` in the Vulkan face order +X, -X, +Y, -Y, +Z, -Z.
void CubeTexelDirection(uint32_t face, float s, float t, float out[3])
{
    const float directions[6][3] = {{1.0f, -t, -s}, {-1.0f, -t, s}, {s, 1.0f, t}, {s, -1.0f, -t}, {s, -t, 1.0f}, {-s, -t, -1.0f}};
    Set3(out, directions[face]);
}

template <typename Radiance>
SkyCube CreateSkyCube(Gpu& gpu, uint32_t size, Radiance radiance)
{
    const VkDevice device = gpu.device.device;
    SkyCube cube;
    cube.size = size;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    info.extent = {size, size, 1};
    info.mipLevels = 1;
    info.arrayLayers = 6;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    REQUIRE(vkCreateImage(device, &info, nullptr, &cube.image) == VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, cube.image, &requirements);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex =
        FindMemoryType(gpu, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    REQUIRE(vkAllocateMemory(device, &allocate, nullptr, &cube.memory) == VK_SUCCESS);
    REQUIRE(vkBindImageMemory(device, cube.image, cube.memory, 0) == VK_SUCCESS);

    // Staging buffer: 6 faces of size x size RGBA16F texels.
    const VkDeviceSize face_bytes = static_cast<VkDeviceSize>(size) * size * 8;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = face_bytes * 6;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    REQUIRE(vkCreateBuffer(device, &buffer_info, nullptr, &staging) == VK_SUCCESS);
    VkMemoryRequirements buffer_requirements{};
    vkGetBufferMemoryRequirements(device, staging, &buffer_requirements);
    VkMemoryAllocateInfo buffer_allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    buffer_allocate.allocationSize = buffer_requirements.size;
    buffer_allocate.memoryTypeIndex =
        FindMemoryType(gpu, buffer_requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    REQUIRE(vkAllocateMemory(device, &buffer_allocate, nullptr, &staging_memory) == VK_SUCCESS);
    REQUIRE(vkBindBufferMemory(device, staging, staging_memory, 0) == VK_SUCCESS);
    void* mapped = nullptr;
    REQUIRE(vkMapMemory(device, staging_memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS);
    uint16_t* halves = static_cast<uint16_t*>(mapped);
    for (uint32_t face = 0; face < 6; ++face)
    {
        for (uint32_t y = 0; y < size; ++y)
        {
            for (uint32_t x = 0; x < size; ++x)
            {
                float direction[3];
                CubeTexelDirection(face, (static_cast<float>(x) + 0.5f) / static_cast<float>(size) * 2.0f - 1.0f,
                                   (static_cast<float>(y) + 0.5f) / static_cast<float>(size) * 2.0f - 1.0f,
                                   direction);
                const float length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] +
                                               direction[2] * direction[2]);
                for (float& d : direction)
                {
                    d /= length;
                }
                float value[3];
                radiance(direction, value);
                const uint16_t texel[4] = {FloatToHalf(value[0]), FloatToHalf(value[1]),
                                           FloatToHalf(value[2]), FloatToHalf(1.0f)};
                std::memcpy(halves + ((static_cast<size_t>(face) * size + y) * size + x) * 4, texel,
                            sizeof(texel));
            }
        }
    }
    vkUnmapMemory(device, staging_memory);

    SubmitAndWait(gpu, [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = cube.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dependency);
        VkBufferImageCopy regions[6] = {};
        for (uint32_t face = 0; face < 6; ++face)
        {
            regions[face].bufferOffset = face_bytes * face;
            regions[face].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, face, 1};
            regions[face].imageExtent = {size, size, 1};
        }
        vkCmdCopyBufferToImage(cmd, staging, cube.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6,
                               regions);
        // The copy -> the renderer's compute-shader sampled reads (the host's duty).
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier2(cmd, &dependency);
    });
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, staging_memory, nullptr);
    return cube;
}

void DestroySkyCube(Gpu& gpu, SkyCube& cube)
{
    vkDestroyImage(gpu.device.device, cube.image, nullptr);
    vkFreeMemory(gpu.device.device, cube.memory, nullptr);
}

}  // namespace

TEST_CASE("sky cubemap, fog and emission scale")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadRedBrick(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LookPlusZ(4.0f, 4.0f, -20.0f);
    constexpr uint32_t cx = kSize / 2;
    constexpr uint32_t cy = kSize / 2;
    float corner_dir[3];
    PrimaryDirection(camera, 0, 0, kSize, kSize, corner_dir);
    float corner_sky[3];
    SkyRadiance(corner_dir, corner_sky);

    constexpr float kL[3] = {0.5f, 1.0f, 2.0f};
    SkyCube cube = CreateSkyCube(rig.gpu, 8, [&](const float*, float out[3]) { Set3(out, kL); });

    // 1. Uniform cubemap, primary miss: the corner is L (alpha 0), the brick is black.
    CaelixRenderOptions options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 1);
    options.sky = cube.Abi();
    Render(rig, renderer, set, camera, options);
    std::printf("sky cubemap, 0 bounces: corner %.6f %.6f %.6f %.6f (expected %.1f %.1f %.1f), centre "
                "%.6f %.6f %.6f %.6f\n",
                color.F32(0, 0, 0), color.F32(0, 0, 1), color.F32(0, 0, 2), color.F32(0, 0, 3), kL[0],
                kL[1], kL[2], color.F32(cx, cy, 0), color.F32(cx, cy, 1), color.F32(cx, cy, 2),
                color.F32(cx, cy, 3));
    for (uint32_t c = 0; c < 3; ++c)
    {
        CHECK(std::abs(color.F32(0, 0, c) - kL[c]) < 1e-4f);
        CHECK(color.F32(cx, cy, c) == 0.0f);
    }
    CHECK(color.F32(0, 0, 3) == 0.0f);
    CHECK(color.F32(cx, cy, 3) == 1.0f);
    // Without the cubemap the procedural sky is back.
    options.sky = CaelixSkyCubemap{};
    Render(rig, renderer, set, camera, options);
    for (uint32_t c = 0; c < 3; ++c)
    {
        CHECK(std::abs(color.F32(0, 0, c) - corner_sky[c]) < 1e-4f);
    }

    // 2. SH irradiance: one bounce, every diffuse bounce ray from the brick's front face
    //    misses the one-brick scene and receives E(n) / pi = L of the constant sky, so a
    //    pixel on the red brick (albedo (1, 0, 0), RGBA8 255) is albedo * L exactly, with no
    //    variance (YCoCg packing in RGBA16F: tolerance 2e-3).
    const float expected_sh[3] = {kL[0], 0.0f, 0.0f};
    options = LightOptions(CAELIX_MODE_PATH_TRACE, 1, 2);
    options.sky = cube.Abi();
    Render(rig, renderer, set, camera, options);
    const uint32_t probes[5][2] = {{cx, cy}, {cx - 7, cy}, {cx + 7, cy}, {cx, cy - 7}, {cx, cy + 7}};
    float max_sh_error = 0.0f;
    for (const auto& probe : probes)
    {
        std::printf("sky SH, 1 bounce: pixel (%u, %u) = %.6f %.6f %.6f %.6f, expected %.6f %.6f %.6f\n",
                    probe[0], probe[1], color.F32(probe[0], probe[1], 0),
                    color.F32(probe[0], probe[1], 1), color.F32(probe[0], probe[1], 2),
                    color.F32(probe[0], probe[1], 3), expected_sh[0], expected_sh[1], expected_sh[2]);
        CHECK(color.F32(probe[0], probe[1], 3) == 1.0f);
        for (uint32_t c = 0; c < 3; ++c)
        {
            max_sh_error =
                std::max(max_sh_error, std::abs(color.F32(probe[0], probe[1], c) - expected_sh[c]));
        }
    }
    std::printf("sky SH, 1 bounce: max error %.3g\n", max_sh_error);
    CHECK(max_sh_error < 2e-3f);
    // Reference mode: two frames with different seeds give the same value (no variance).
    float reference[2] = {0.0f, 0.0f};
    for (uint32_t n = 0; n < 2; ++n)
    {
        CaelixRenderOptions ref = LightOptions(CAELIX_MODE_REFERENCE, 1, 100 + n);
        ref.reset_accumulation = n == 0 ? 1u : 0u;
        ref.sky = cube.Abi();
        Render(rig, renderer, set, camera, ref);
        reference[n] = color.F32(cx, cy, 0);
    }
    std::printf("sky SH, reference: frame 1 %.6f, frame 2 %.6f\n", reference[0], reference[1]);
    CHECK(std::abs(reference[0] - expected_sh[0]) < 2e-3f);
    CHECK(std::abs(reference[1] - reference[0]) < 1e-6f);
    // A linear sky L(w) = a + b w.z is exactly band 0 + band 1: E(n) / pi = a + (2 / 3) b n.z, so the
    // front face (n = -Z) receives a - 2 b / 3 (checks the band-1 factor and the orientation; the
    // 8 x 8 cubemap interpolates a + b w.z only approximately: tolerance 1e-2).
    {
        constexpr float kA = 1.0f;
        constexpr float kB = 0.6f;
        SkyCube gradient = CreateSkyCube(rig.gpu, 8, [&](const float* direction, float out[3]) {
            out[0] = out[1] = out[2] = kA + kB * direction[2];
        });
        CaelixRenderOptions linear = LightOptions(CAELIX_MODE_PATH_TRACE, 1, 5);
        linear.sky = gradient.Abi();
        Render(rig, renderer, set, camera, linear);
        const float expected_linear = kA - 2.0f * kB / 3.0f;
        std::printf("sky SH, linear sky a + b w.z: centre %.6f, expected %.6f\n", color.F32(cx, cy, 0),
                    expected_linear);
        CHECK(std::abs(color.F32(cx, cy, 0) - expected_linear) < 1e-2f);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        DestroySkyCube(rig.gpu, gradient);
    }

    // 3. Fog: no cubemap, 0 bounces. The centre ray reaches the brick's front face (z = 0)
    //    at t = 20 / dir.z; the brick has no emission, so the centre is the fog's emission
    //    fog_color (1 - exp(-density t)) (RGBA16F deterministic target: 1e-3 relative).
    //    The corner (a miss) is the procedural sky, unchanged.
    float centre_dir[3];
    PrimaryDirection(camera, cx, cy, kSize, kSize, centre_dir);
    const double t_centre = 20.0 / centre_dir[2];
    constexpr float kFogColor[3] = {1.0f, 0.5f, 0.25f};
    constexpr float kFogDensity = 0.05f;
    options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 3);
    options.fog_density = kFogDensity;
    Set3(options.fog_color, kFogColor);
    Render(rig, renderer, set, camera, options);
    const double fog_weight = 1.0 - std::exp(-static_cast<double>(kFogDensity) * t_centre);
    std::printf("fog: t %.5f, centre %.6f %.6f %.6f %.6f, expected %.6f %.6f %.6f; corner %.6f %.6f "
                "%.6f\n",
                t_centre, color.F32(cx, cy, 0), color.F32(cx, cy, 1), color.F32(cx, cy, 2),
                color.F32(cx, cy, 3), kFogColor[0] * fog_weight, kFogColor[1] * fog_weight,
                kFogColor[2] * fog_weight, color.F32(0, 0, 0), color.F32(0, 0, 1), color.F32(0, 0, 2));
    for (uint32_t c = 0; c < 3; ++c)
    {
        const double expected = kFogColor[c] * fog_weight;
        CHECK(std::abs(color.F32(cx, cy, c) - expected) <= 1e-3 * expected);
        CHECK(std::abs(color.F32(0, 0, c) - corner_sky[c]) < 1e-4f);
    }
    CHECK(color.F32(cx, cy, 3) == 1.0f);
    CHECK(color.F32(0, 0, 3) == 0.0f);
    options.fog_density = 0.0f;
    Render(rig, renderer, set, camera, options);
    for (uint32_t c = 0; c < 3; ++c)
    {
        CHECK(color.F32(cx, cy, c) == 0.0f);
    }

    // 4. Emission scale: the emissive red brick (rgb4 (15, 0, 0), strength 0) is (1, 0, 0) at
    //    scale 1 (and at 0, which means 1), (3, 0, 0) at scale 3.
    {
        World world;
        Brick brick{};
        brick.fill(static_cast<uint16_t>(0x4000u | (15u << 10)));
        world.bricks[{0, 0, 0}] = brick;
        Upload(rig, world, KeysOf(world));
        DrainBuilds(rig);
    }
    float emitted[3] = {0.0f, 0.0f, 0.0f};
    const float scales[3] = {1.0f, 3.0f, 0.0f};
    for (uint32_t i = 0; i < 3; ++i)
    {
        options = LightOptions(CAELIX_MODE_PATH_TRACE, 0, 4);
        options.emission_scale = scales[i];
        Render(rig, renderer, set, camera, options);
        emitted[i] = color.F32(cx, cy, 0);
        CHECK(color.F32(cx, cy, 1) == 0.0f);
        CHECK(color.F32(cx, cy, 2) == 0.0f);
    }
    std::printf("emission scale 1 / 3 / 0: centre red %.6f / %.6f / %.6f\n", emitted[0], emitted[1],
                emitted[2]);
    CHECK(std::abs(emitted[0] - 1.0f) < 1e-4f);
    CHECK(std::abs(emitted[1] - 3.0f * emitted[0]) <= 1e-4f * 3.0f * emitted[0]);
    CHECK(emitted[2] == emitted[0]);

    // 5. Argument checks: a sky image without mip levels, a negative fog density, a negative
    //    emission scale.
    const auto expect_invalid = [&](const CaelixRenderOptions& bad) {
        ExpectLibraryErrors(1, [&] {
            CaelixFrameDesc desc{};
            desc.command_buffer = 1;
            desc.frame_number = rig.frames.next;
            const CaelixRenderTargets targets = ToAbi(set);
            REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &bad) ==
                    CAELIX_ERROR_INVALID_ARGUMENT);
        });
    };
    CaelixRenderOptions bad = LightOptions(CAELIX_MODE_PATH_TRACE, 1, 0);
    bad.sky = cube.Abi();
    bad.sky.mip_levels = 0;
    expect_invalid(bad);
    bad = LightOptions(CAELIX_MODE_PATH_TRACE, 1, 0);
    bad.fog_density = -1.0f;
    expect_invalid(bad);
    bad = LightOptions(CAELIX_MODE_PATH_TRACE, 1, 0);
    bad.emission_scale = -1.0f;
    expect_invalid(bad);

    CaelixGpuTimings timings{};
    caelix_renderer_get_gpu_timings(renderer, &timings);
    std::printf("timings: trace %.4f ms (with the sky SH dispatch in the cubemap frames), composite "
                "%.4f ms\n",
                timings.trace_ms, timings.composite_ms);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);  // also the cube view of the sky
    caelix_renderer_destroy(renderer);
    DestroySkyCube(rig.gpu, cube);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

// ReSTIR GI (caelix_renderer_set_gi_options, docs/GI_EXPERIMENTS.md): option validation, the
// unbiasedness of the initial samples without reuse, and the bias of the reuse.

namespace {

// A grey floor (64 x 64 x 8 blocks, y in [-8, 0)), an 8 x 2 x 8 white emitter of emission 8
// floating at y in [14, 16) and a grey wall at x in [40, 42) that casts a shadow; black sky, no
// sun. The camera looks down at the floor from (10, 30, 10) toward (36, 0, 36).
void UploadGiRoom(Rig& rig)
{
    World world;
    FillBox(world, 0, -8, 0, 63, -1, 63, kGreyFloor);
    FillBox(world, 28, 14, 28, 35, 15, 35, kWhiteEmitter8);
    FillBox(world, 40, 0, 16, 41, 9, 47, kGreyFloor);
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

CaelixCamera GiRoomCamera()
{
    const double position[3] = {10.0, 30.0, 10.0};
    const double forward[3] = {26.0, -30.0, 26.0};
    return LookAlong(position, forward, std::tan(30.0 * 3.14159265358979323846 / 180.0));
}

CaelixGiOptions GiOptions(uint32_t method, uint32_t flags)
{
    CaelixGiOptions gi{};
    gi.method = method;
    gi.flags = flags;
    return gi;
}

// Reference mode over `frames` frames (seeds seed0 ..), 3 bounces, black sky; the colour (the
// running mean) is read back after the last frame and returned.
std::vector<float> GiReference(Rig& rig, CaelixRenderer* renderer, TargetSet& set, const CaelixCamera& camera,
                               uint32_t frames, uint32_t seed0)
{
    for (uint32_t i = 0; i < frames; ++i)
    {
        CaelixRenderOptions options = GlassOptions(CAELIX_MODE_REFERENCE, 3, 0, seed0 + i);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, set, camera, options, i + 1 == frames);
    }
    const Target& color = *set.t[kColor];
    std::vector<float> image(static_cast<size_t>(color.width) * color.height * 4);
    std::memcpy(image.data(), color.bytes.data(), image.size() * sizeof(float));
    return image;
}

// Comparison of a test image against a reference image (RGBA32F, kSize x kSize) over the lit
// pixels of the reference: covered, max(rgb) above 1e-4 and below 4 (the emitter's own pixels,
// deterministic emission 8, are left out so that the indirect light dominates the means).
struct GiComparison
{
    uint32_t pixels = 0;
    double ref_mean[3] = {0.0, 0.0, 0.0};
    double test_mean[3] = {0.0, 0.0, 0.0};
    double max_ratio_error = 0.0;  // max over channels of |test / ref - 1|
    double psnr = 0.0;             // dB over the lit pixels, both clamped to [0, 1], peak 1
};

GiComparison CompareGi(const std::vector<float>& ref, const std::vector<float>& test)
{
    GiComparison c;
    double squared = 0.0;
    for (size_t p = 0; p < static_cast<size_t>(kSize) * kSize; ++p)
    {
        const float* r = &ref[p * 4];
        const float* t = &test[p * 4];
        const float peak = std::max(r[0], std::max(r[1], r[2]));
        if (r[3] != 1.0f || !(peak > 1e-4f) || !(peak < 4.0f))
        {
            continue;
        }
        ++c.pixels;
        for (int ch = 0; ch < 3; ++ch)
        {
            c.ref_mean[ch] += r[ch];
            c.test_mean[ch] += t[ch];
            const double d = std::clamp(static_cast<double>(t[ch]), 0.0, 1.0) -
                             std::clamp(static_cast<double>(r[ch]), 0.0, 1.0);
            squared += d * d;
        }
    }
    REQUIRE(c.pixels > 1000);
    for (int ch = 0; ch < 3; ++ch)
    {
        c.ref_mean[ch] /= c.pixels;
        c.test_mean[ch] /= c.pixels;
        c.max_ratio_error = std::max(c.max_ratio_error, std::abs(c.test_mean[ch] / c.ref_mean[ch] - 1.0));
    }
    const double mse = squared / (3.0 * c.pixels);
    c.psnr = mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 99.0;
    return c;
}

void PrintGi(const char* name, const GiComparison& c)
{
    std::printf("%s: %u lit pixels, mean ref %.5f %.5f %.5f, test %.5f %.5f %.5f, max ratio error %.4f, "
                "PSNR %.2f dB\n",
                name, c.pixels, c.ref_mean[0], c.ref_mean[1], c.ref_mean[2], c.test_mean[0], c.test_mean[1],
                c.test_mean[2], c.max_ratio_error, c.psnr);
}

}  // namespace

TEST_CASE("ReSTIR GI: set_gi_options validation")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);

    const auto expect_invalid = [&](const CaelixGiOptions& bad) {
        ExpectLibraryErrors(1, [&] {
            REQUIRE(caelix_renderer_set_gi_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
    };
    CaelixGiOptions bad = GiOptions(2, 0);  // unknown method
    expect_invalid(bad);
    bad = GiOptions(CAELIX_GI_RESTIR, 1u << 31);  // unknown flag (bits 0 .. 8 are defined)
    expect_invalid(bad);
    bad = GiOptions(CAELIX_GI_RESTIR, CAELIX_GI_FLAG_SPATIAL);
    bad.spatial_samples = 9;
    expect_invalid(bad);
    bad = GiOptions(CAELIX_GI_RESTIR, CAELIX_GI_FLAG_SPATIAL);
    bad.spatial_passes = 3;
    expect_invalid(bad);
    bad = GiOptions(CAELIX_GI_RESTIR, CAELIX_GI_FLAG_SPATIAL);
    bad.spatial_radius = NAN;
    expect_invalid(bad);
    bad = GiOptions(CAELIX_GI_RESTIR, CAELIX_GI_FLAG_SPATIAL);
    bad.depth_threshold = -0.5f;
    expect_invalid(bad);
    ExpectLibraryErrors(1, [&] { REQUIRE(caelix_renderer_set_gi_options(renderer, nullptr) == CAELIX_ERROR_INVALID_ARGUMENT); });

    CaelixGiOptions good = GiOptions(CAELIX_GI_RESTIR, CAELIX_GI_FLAG_TEMPORAL | CAELIX_GI_FLAG_SPATIAL |
                                                           CAELIX_GI_FLAG_UNBIASED | CAELIX_GI_FLAG_DIFFUSE_ONLY);
    good.temporal_m_cap = 64;
    good.spatial_samples = 8;
    good.spatial_passes = 2;
    good.spatial_radius = 12.5f;
    good.depth_threshold = 0.05f;
    good.reset = 1;
    CHECK(caelix_renderer_set_gi_options(renderer, &good) == CAELIX_OK);
    const CaelixGiOptions zero{};
    CHECK(caelix_renderer_set_gi_options(renderer, &zero) == CAELIX_OK);
    // No ReSTIR render yet: no reservoir buffer.
    CaelixGiStats stats{};
    caelix_renderer_get_gi_stats(renderer, &stats);
    CHECK(stats.reservoir_bytes == 0);

    caelix_renderer_destroy(renderer);
    CloseRig(rig);
}

TEST_CASE("ReSTIR GI: without reuse it matches the path tracer; with reuse its bias stays bounded")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadGiRoom(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = GiRoomCamera();
    constexpr uint32_t kFrames = 256;

    // 1. The path tracer with the Lambert lobe forced (the reference of the comparisons).
    CaelixGiOptions gi = GiOptions(CAELIX_GI_PATH_TRACE, CAELIX_GI_FLAG_DIFFUSE_ONLY);
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> pt = GiReference(rig, renderer, set, camera, kFrames, 5000);
    CaelixGiStats stats{};
    caelix_renderer_get_gi_stats(renderer, &stats);
    CHECK(stats.reservoir_bytes == 0);
    CHECK(stats.gi_ms == 0.0f);

    // 2. ReSTIR GI without temporal and spatial reuse: every pixel's reservoir is its initial
    //    sample, so the estimate is the path tracer's (other random numbers: the two means differ
    //    by noise only).
    gi = GiOptions(CAELIX_GI_RESTIR, CAELIX_GI_FLAG_DIFFUSE_ONLY);
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> no_reuse = GiReference(rig, renderer, set, camera, kFrames, 9000);
    caelix_renderer_get_gi_stats(renderer, &stats);
    std::printf("ReSTIR GI stats: gi %.4f ms, reservoirs %llu bytes\n", stats.gi_ms,
                static_cast<unsigned long long>(stats.reservoir_bytes));
    CHECK(stats.reservoir_bytes == 4ull * kSize * kSize * 48);
    CHECK(stats.gi_ms > 0.0f);
    // The noise floor: two path tracer runs with other seeds. At 256 frames the pixel noise of
    // this scene limits the PSNR to about 24 dB, so the check is against that floor, not a
    // fixed value.
    gi = GiOptions(CAELIX_GI_PATH_TRACE, CAELIX_GI_FLAG_DIFFUSE_ONLY);
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const GiComparison noise = CompareGi(pt, GiReference(rig, renderer, set, camera, kFrames, 13000));
    PrintGi("path tracer vs path tracer, other seeds (noise floor)", noise);
    const GiComparison unbiased = CompareGi(pt, no_reuse);
    PrintGi("ReSTIR GI without reuse vs path tracer (256 frames)", unbiased);
    CHECK(unbiased.max_ratio_error < 0.03);
    CHECK(unbiased.psnr > noise.psnr - 1.0);

    // 3. Temporal + spatial reuse (1/M weights, no visibility rays): biased; the mean must stay
    //    within 10 percent of the path tracer's. Then with visibility rays and the unbiased
    //    combination (reported, same bound).
    gi = GiOptions(CAELIX_GI_RESTIR, CAELIX_GI_FLAG_DIFFUSE_ONLY | CAELIX_GI_FLAG_TEMPORAL | CAELIX_GI_FLAG_SPATIAL);
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const GiComparison reuse = CompareGi(pt, GiReference(rig, renderer, set, camera, kFrames, 17000));
    PrintGi("ReSTIR GI temporal + spatial (biased) vs path tracer", reuse);
    CHECK(reuse.max_ratio_error < 0.10);

    gi.flags |= CAELIX_GI_FLAG_UNBIASED;
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const GiComparison reuse_unbiased = CompareGi(pt, GiReference(rig, renderer, set, camera, kFrames, 21000));
    PrintGi("ReSTIR GI temporal + spatial (unbiased) vs path tracer", reuse_unbiased);
    CHECK(reuse_unbiased.max_ratio_error < 0.10);

    // 4. Back to the path tracer: the image is the path tracer's again, bit for bit (same seeds).
    gi = GiOptions(CAELIX_GI_PATH_TRACE, CAELIX_GI_FLAG_DIFFUSE_ONLY);
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> pt_again = GiReference(rig, renderer, set, camera, 4, 5000);
    const std::vector<float> pt_short = [&] {
        const CaelixGiOptions zero{};
        REQUIRE(caelix_renderer_set_gi_options(renderer, &zero) == CAELIX_OK);
        return GiReference(rig, renderer, set, camera, 4, 5000);
    }();
    // Packed scenes have smoothness 0: DIFFUSE_ONLY changes nothing there.
    CHECK(std::memcmp(pt_again.data(), pt_short.data(), pt_again.size() * sizeof(float)) == 0);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

// ---------------------------------------------------------------------------
// Light list (docs/LIGHT_LIST.md): the scene's light table, next-event estimation on list lights
// (CAELIX_GI_FLAG_LIGHT_NEE) against the plain path tracer, the path tracer without the flag, and
// the renderer's light cells.

namespace {

// Packed white emitter of strength 3: emission 512 x albedo 1 (a list light at the default
// min_emission 20).
constexpr uint16_t kWhiteEmitter512 = 0x4000u | (15u << 10) | (15u << 6) | (15u << 2) | 3u;
constexpr uint32_t kBoxSize = 128;  // render size of the light NEE cases
constexpr int32_t kBoxInterior = 24;

CaelixLightStats LightStats(const CaelixScene* scene)
{
    CaelixLightStats stats{};
    caelix_scene_get_light_stats(scene, &stats);
    return stats;
}

// A closed grey box: walls 1 voxel thick around the interior [0, 24)^3, one strength-3 white voxel
// in the middle of the ceiling (12, 24, 12) as the only light.
// The light table is opt-in: zero options = the defaults, and the table is built from then on.
void EnableLights(CaelixScene* scene)
{
    const CaelixLightOptions options{};
    REQUIRE(caelix_scene_set_light_options(scene, &options) == CAELIX_OK);
}

// The walls of the light box (grey, 1 voxel thick around the interior [0, 24)^3, no light).
World LightBoxWalls()
{
    World world;
    const int32_t lo = -1;
    const int32_t hi = kBoxInterior;
    FillBox(world, lo, lo, lo, hi, lo, hi, kGreyFloor);  // floor
    FillBox(world, lo, hi, lo, hi, hi, hi, kGreyFloor);  // ceiling
    FillBox(world, lo, lo, lo, lo, hi, hi, kGreyFloor);  // -X wall
    FillBox(world, hi, lo, lo, hi, hi, hi, kGreyFloor);  // +X wall
    FillBox(world, lo, lo, lo, hi, hi, lo, kGreyFloor);  // -Z wall
    FillBox(world, lo, lo, hi, hi, hi, hi, kGreyFloor);  // +Z wall
    return world;
}

void UploadLightBox(Rig& rig)
{
    EnableLights(rig.scene);
    World world = LightBoxWalls();
    FillBox(world, 12, kBoxInterior, 12, 12, kBoxInterior, 12, kWhiteEmitter512);
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

// The light box with an 8 x 8 panel of strength-3 white voxels in the ceiling (y = 24, x and z in
// 8 .. 15: one brick) instead of the single voxel: 25 units above the floor, beyond the near-field
// split, and large on the hemisphere of the floor, so that both the light NEE and the BSDF path
// carry weight under MIS.
void UploadLightPanelBox(Rig& rig)
{
    EnableLights(rig.scene);
    World world = LightBoxWalls();
    FillBox(world, 8, kBoxInterior, 8, 15, kBoxInterior, 15, kWhiteEmitter512);
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

// Inside the box at its centre, looking along +Z: the light is above the view cone (half angle 30
// degrees), so no pixel sees it directly.
CaelixCamera LightBoxCamera()
{
    const double position[3] = {12.0, 12.0, 2.0};
    const double forward[3] = {0.0, 0.0, 1.0};
    return LookAlong(position, forward, std::tan(30.0 * 3.14159265358979323846 / 180.0));
}

// Reference mode over `frames` frames (seeds seed0 ..), 4 bounces, black sky, no sun; the colour
// (the running mean) is read back after the last frame.
std::vector<float> LightReference(Rig& rig, CaelixRenderer* renderer, TargetSet& set, const CaelixCamera& camera,
                                  uint32_t frames, uint32_t seed0)
{
    for (uint32_t i = 0; i < frames; ++i)
    {
        CaelixRenderOptions options = GlassOptions(CAELIX_MODE_REFERENCE, 4, 0, seed0 + i);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, set, camera, options, i + 1 == frames);
    }
    const Target& color = *set.t[kColor];
    std::vector<float> image(static_cast<size_t>(color.width) * color.height * 4);
    std::memcpy(image.data(), color.bytes.data(), image.size() * sizeof(float));
    return image;
}

// Mean linear brightness (rgb mean over every pixel) and the PSNR of 16 x 16 block means (both
// images clamped to [0, 1], peak 1): a 1-voxel light is a rare event for the plain path tracer, so
// its per-pixel noise at a few thousand frames is far above 30 dB; the block means compare the
// light distribution.
struct LightComparison
{
    double ref_mean = 0.0;
    double test_mean = 0.0;
    double ratio = 0.0;
    double block_psnr = 0.0;
};

LightComparison CompareLight(const std::vector<float>& ref, const std::vector<float>& test, uint32_t size)
{
    constexpr uint32_t kBlock = 16;
    LightComparison c;
    double squared = 0.0;
    uint32_t blocks = 0;
    for (uint32_t by = 0; by < size; by += kBlock)
    {
        for (uint32_t bx = 0; bx < size; bx += kBlock)
        {
            double r_block[3] = {0.0, 0.0, 0.0};
            double t_block[3] = {0.0, 0.0, 0.0};
            for (uint32_t y = by; y < by + kBlock; ++y)
            {
                for (uint32_t x = bx; x < bx + kBlock; ++x)
                {
                    const size_t p = (static_cast<size_t>(y) * size + x) * 4;
                    for (int ch = 0; ch < 3; ++ch)
                    {
                        r_block[ch] += ref[p + ch];
                        t_block[ch] += test[p + ch];
                    }
                }
            }
            for (int ch = 0; ch < 3; ++ch)
            {
                c.ref_mean += r_block[ch];
                c.test_mean += t_block[ch];
                const double d = std::clamp(t_block[ch] / (kBlock * kBlock), 0.0, 1.0) -
                                 std::clamp(r_block[ch] / (kBlock * kBlock), 0.0, 1.0);
                squared += d * d;
            }
            ++blocks;
        }
    }
    const double values = 3.0 * static_cast<double>(size) * size;
    c.ref_mean /= values;
    c.test_mean /= values;
    c.ratio = c.test_mean / c.ref_mean;
    const double mse = squared / (3.0 * blocks);
    c.block_psnr = mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 99.0;
    return c;
}

}  // namespace

TEST_CASE("light table counts one emissive brick")
{
    // Brick (0, 0, 0): three strength-3 white voxels and a grey one; brick (1, 0, 0): grey only.
    World world;
    FillBox(world, 1, 1, 1, 3, 1, 1, kWhiteEmitter512);
    FillBox(world, 5, 5, 5, 5, 5, 5, kGreyFloor);
    FillBox(world, 9, 2, 2, 12, 4, 4, kGreyFloor);
    REQUIRE(world.bricks.size() == 2);

    // 1. Default options: one entry.
    {
        Rig rig;
        OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
        CHECK(LightStats(rig.scene).rebuilds == 0);
        EnableLights(rig.scene);
        Upload(rig, world, KeysOf(world));
        const CaelixLightStats stats = LightStats(rig.scene);
        std::printf("light table: %u bricks, %u dropped, %llu bytes, %llu rebuilds\n", stats.bricks,
                    stats.bricks_dropped, static_cast<unsigned long long>(stats.table_bytes),
                    static_cast<unsigned long long>(stats.rebuilds));
        CHECK(stats.bricks == 1);
        CHECK(stats.bricks_dropped == 0);
        CHECK(stats.table_bytes > 64ull + 65536ull * 132ull);
        CHECK(stats.rebuilds >= 1);

        // A threshold above every emission: the next flush rebuilds the table without entries.
        CaelixLightOptions options{};
        options.min_emission = 1e6f;
        REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_OK);
        FlushFrame(rig.gpu, rig.scene, rig.frames);
        CHECK(LightStats(rig.scene).bricks == 0);
        CHECK(LightStats(rig.scene).bricks_dropped == 0);

        // Validation.
        options.min_emission = -1.0f;
        ExpectLibraryErrors(1, [&] {
            REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        options.min_emission = 0.0f;
        options.max_bricks = (1u << 20) + 1u;
        ExpectLibraryErrors(1, [&] {
            REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        CloseRig(rig);
    }

    // 2. The threshold set before the upload: the light words are 0, no entry.
    {
        Rig rig;
        OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
        CaelixLightOptions options{};
        options.min_emission = 1e6f;
        REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_OK);
        Upload(rig, world, KeysOf(world));
        CHECK(LightStats(rig.scene).bricks == 0);
        CloseRig(rig);
    }

    // 3. Two emissive bricks and a capacity of one: one entry, one dropped.
    {
        World two = world;
        FillBox(two, 12, 4, 4, 12, 4, 4, kWhiteEmitter512);  // a corner of the grey block: alive
        Rig rig;
        OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
        CaelixLightOptions options{};
        options.max_bricks = 1;
        REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_OK);
        Upload(rig, two, KeysOf(two));
        const CaelixLightStats stats = LightStats(rig.scene);
        CHECK(stats.bricks == 1);
        CHECK(stats.bricks_dropped == 1);
        CloseRig(rig);
    }
}

TEST_CASE("light NEE is unbiased")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadLightBox(rig);
    REQUIRE(LightStats(rig.scene).bricks == 1);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kBoxSize, kBoxSize);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LightBoxCamera();

    // The plain path tracer finds the 1-voxel light rarely: 4096 frames for its mean.
    CaelixGiOptions gi{};
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> pt = LightReference(rig, renderer, set, camera, 4096, 100000);
    gi.flags = CAELIX_GI_FLAG_LIGHT_NEE;
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> nee = LightReference(rig, renderer, set, camera, 1024, 200000);
    const LightComparison c = CompareLight(pt, nee, kBoxSize);
    std::printf("light NEE vs path tracer: mean %.5f vs %.5f, ratio %.4f, 16x16 block PSNR %.2f dB\n", c.test_mean,
                c.ref_mean, c.ratio, c.block_psnr);
    // The noise floor of the reference (reported only): the path tracer with other seeds.
    gi.flags = 0;
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const LightComparison floor_c =
        CompareLight(pt, LightReference(rig, renderer, set, camera, 4096, 300000), kBoxSize);
    std::printf("path tracer vs path tracer, other seeds: ratio %.4f, 16x16 block PSNR %.2f dB\n", floor_c.ratio,
                floor_c.block_psnr);
    CHECK(c.ref_mean > 0.01);
    CHECK(c.ratio > 0.97);
    CHECK(c.ratio < 1.03);
    CHECK(c.block_psnr > 30.0);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

namespace {

// PSNR over every pixel of two size x size RGBA32F images (both clamped to [0, 1], peak 1).
double PixelPsnr(const std::vector<float>& ref, const std::vector<float>& test, uint32_t size)
{
    double squared = 0.0;
    for (size_t p = 0; p < static_cast<size_t>(size) * size; ++p)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            const double d = std::clamp(static_cast<double>(test[p * 4 + ch]), 0.0, 1.0) -
                             std::clamp(static_cast<double>(ref[p * 4 + ch]), 0.0, 1.0);
            squared += d * d;
        }
    }
    const double mse = squared / (3.0 * size * size);
    return mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 99.0;
}

// Linear RMSE (no clamp) over every pixel and channel.
double PixelRmse(const std::vector<float>& ref, const std::vector<float>& test, uint32_t size)
{
    double squared = 0.0;
    for (size_t p = 0; p < static_cast<size_t>(size) * size; ++p)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            const double d = static_cast<double>(test[p * 4 + ch]) - static_cast<double>(ref[p * 4 + ch]);
            squared += d * d;
        }
    }
    return std::sqrt(squared / (3.0 * size * size));
}

std::vector<float> ColorPixels(const Target& color)
{
    std::vector<float> image(static_cast<size_t>(color.width) * color.height * 4);
    std::memcpy(image.data(), color.bytes.data(), image.size() * sizeof(float));
    return image;
}

}  // namespace

TEST_CASE("light NEE with MIS is unbiased")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadLightBox(rig);
    REQUIRE(LightStats(rig.scene).bricks == 1);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kBoxSize, kBoxSize);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LightBoxCamera();

    CaelixGiOptions gi{};
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> pt = LightReference(rig, renderer, set, camera, 4096, 100000);
    gi.flags = CAELIX_GI_FLAG_LIGHT_NEE | CAELIX_GI_FLAG_LIGHT_MIS;
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> mis = LightReference(rig, renderer, set, camera, 1024, 200000);
    const LightComparison c = CompareLight(pt, mis, kBoxSize);
    std::printf("light NEE + MIS vs path tracer: mean %.5f vs %.5f, ratio %.4f, 16x16 block PSNR %.2f dB\n",
                c.test_mean, c.ref_mean, c.ratio, c.block_psnr);
    CHECK(c.ref_mean > 0.01);
    CHECK(c.ratio > 0.97);
    CHECK(c.ratio < 1.03);
    CHECK(c.block_psnr > 30.0);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("MIS on a large emitter")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadLightPanelBox(rig);
    REQUIRE(LightStats(rig.scene).bricks == 1);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kBoxSize, kBoxSize);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LightBoxCamera();

    CaelixGiOptions gi{};
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> pt = LightReference(rig, renderer, set, camera, 4096, 100000);
    const uint32_t modes[2] = {CAELIX_GI_FLAG_LIGHT_NEE, CAELIX_GI_FLAG_LIGHT_NEE | CAELIX_GI_FLAG_LIGHT_MIS};
    const char* names[2] = {"light NEE", "light NEE + MIS"};
    double rmse[2] = {0.0, 0.0};
    for (int m = 0; m < 2; ++m)
    {
        gi.flags = modes[m];
        REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
        const std::vector<float> converged = LightReference(rig, renderer, set, camera, 1024, 200000);
        const LightComparison c = CompareLight(pt, converged, kBoxSize);
        // The noise of a short accumulation: 64 frames against the 4096-frame path tracer.
        const std::vector<float> short_run = LightReference(rig, renderer, set, camera, 64, 500000);
        rmse[m] = PixelRmse(pt, short_run, kBoxSize);
        std::printf("panel, %s vs path tracer: mean %.5f vs %.5f, ratio %.4f, 16x16 block PSNR %.2f dB, "
                    "64-frame RMSE %.5f\n",
                    names[m], c.test_mean, c.ref_mean, c.ratio, c.block_psnr, rmse[m]);
        CHECK(c.ref_mean > 0.01);
        CHECK(c.ratio > 0.97);
        CHECK(c.ratio < 1.03);
    }
    std::printf("panel: 64-frame RMSE without MIS %.5f, with MIS %.5f (ratio %.3f)\n", rmse[0], rmse[1],
                rmse[1] / rmse[0]);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("dropped bricks keep their emission")
{
    // Two single-voxel lights in two ceiling bricks and a table capacity of one brick: the table
    // drops one of them, the light NEE never samples it, and the BSDF path must keep its emission
    // (docs/LIGHT_MIS.md section 2: the brick -> entry map has no entry for it).
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    CaelixLightOptions options{};
    options.max_bricks = 1;
    REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_OK);
    World world = LightBoxWalls();
    FillBox(world, 4, kBoxInterior, 4, 4, kBoxInterior, 4, kWhiteEmitter512);
    FillBox(world, 20, kBoxInterior, 20, 20, kBoxInterior, 20, kWhiteEmitter512);
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
    const CaelixLightStats stats = LightStats(rig.scene);
    std::printf("dropped bricks: %u bricks, %u dropped\n", stats.bricks, stats.bricks_dropped);
    REQUIRE(stats.bricks == 1);
    REQUIRE(stats.bricks_dropped == 1);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kBoxSize, kBoxSize);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LightBoxCamera();

    CaelixGiOptions gi{};
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> pt = LightReference(rig, renderer, set, camera, 4096, 100000);
    const uint32_t modes[2] = {CAELIX_GI_FLAG_LIGHT_NEE, CAELIX_GI_FLAG_LIGHT_NEE | CAELIX_GI_FLAG_LIGHT_MIS};
    const char* names[2] = {"light NEE", "light NEE + MIS"};
    for (int m = 0; m < 2; ++m)
    {
        gi.flags = modes[m];
        REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
        const LightComparison c =
            CompareLight(pt, LightReference(rig, renderer, set, camera, 1024, 200000), kBoxSize);
        std::printf("dropped bricks, %s vs path tracer: mean %.5f vs %.5f, ratio %.4f, 16x16 block PSNR %.2f dB\n",
                    names[m], c.test_mean, c.ref_mean, c.ratio, c.block_psnr);
        CHECK(c.ref_mean > 0.01);
        CHECK(c.ratio > 0.97);
        CHECK(c.ratio < 1.03);
    }

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("path tracer unchanged without the flag")
{
    // The ReSTIR GI room (an emitter of strength 1: luminance 8). Scene A sets min_emission 1, so its
    // emitter is a list light and its table has entries; scene B keeps the defaults (no list light).
    // One path tracer frame without CAELIX_GI_FLAG_LIGHT_NEE must be the same bits in both.
    const CaelixCamera camera = GiRoomCamera();
    std::vector<uint8_t> images[2];
    for (int scene = 0; scene < 2; ++scene)
    {
        Rig rig;
        OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
        if (scene == 0)
        {
            CaelixLightOptions options{};
            options.min_emission = 1.0f;
            REQUIRE(caelix_scene_set_light_options(rig.scene, &options) == CAELIX_OK);
        }
        UploadGiRoom(rig);
        const CaelixLightStats stats = LightStats(rig.scene);
        std::printf("scene %c: %u light bricks\n", scene == 0 ? 'A' : 'B', stats.bricks);
        CHECK((scene == 0 ? stats.bricks > 0 : stats.bricks == 0));
        CaelixRenderer* renderer = nullptr;
        REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
        Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
        TargetSet set;
        set.t[kColor] = &color;
        Render(rig, renderer, set, camera, GlassOptions(CAELIX_MODE_PATH_TRACE, 4, 0, 4242));
        images[scene] = color.bytes;
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        caelix_renderer_release_target_views(renderer);
        caelix_renderer_destroy(renderer);
        DestroyTarget(rig.gpu, color);
        CloseRig(rig);
    }
    REQUIRE(images[0].size() == images[1].size());
    CHECK(std::memcmp(images[0].data(), images[1].data(), images[0].size()) == 0);
}

TEST_CASE("cells fill and sample")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadLightBox(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);

    CaelixGiOptions bad{};
    bad.flags = CAELIX_GI_FLAG_LIGHT_NEE;
    bad.light_cell_capacity = 1000;  // not a power of two
    ExpectLibraryErrors(1, [&] {
        REQUIRE(caelix_renderer_set_gi_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
    bad.light_cell_capacity = 1u << 21;
    ExpectLibraryErrors(1, [&] {
        REQUIRE(caelix_renderer_set_gi_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
    bad.light_cell_capacity = 0;
    bad.light_candidates = 65;
    ExpectLibraryErrors(1, [&] {
        REQUIRE(caelix_renderer_set_gi_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
    bad.light_candidates = 0;
    bad.light_level_footprint = -1.0f;
    ExpectLibraryErrors(1, [&] {
        REQUIRE(caelix_renderer_set_gi_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    });

    CaelixGiOptions gi{};
    gi.flags = CAELIX_GI_FLAG_LIGHT_NEE;
    gi.light_cell_capacity = 1024;
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    CaelixGiStats stats{};
    caelix_renderer_get_gi_stats(renderer, &stats);
    CHECK(stats.light_cell_bytes == 0);  // created by the first render

    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kBoxSize, kBoxSize);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LightBoxCamera();
    for (uint32_t i = 0; i < 4; ++i)
    {
        Render(rig, renderer, set, camera, GlassOptions(CAELIX_MODE_PATH_TRACE, 4, 0, 777 + i), i == 3);
    }
    caelix_renderer_get_gi_stats(renderer, &stats);
    CaelixGpuTimings timings{};
    caelix_renderer_get_gpu_timings(renderer, &timings);
    std::printf("light cells: %llu bytes, fill %.4f ms (frame %llu), trace %.4f ms\n",
                static_cast<unsigned long long>(stats.light_cell_bytes), stats.light_fill_ms,
                static_cast<unsigned long long>(timings.frame_number), timings.trace_ms);
    CHECK(stats.light_cell_bytes == 1024ull * 296ull);
    CHECK(timings.frame_number != 0);
    CHECK(stats.light_fill_ms > 0.0f);

    // Not black: the last frame's lit pixels (1 spp with light NEE: almost every pixel).
    const float* pixels = reinterpret_cast<const float*>(color.bytes.data());
    uint32_t lit = 0;
    double sum = 0.0;
    for (size_t p = 0; p < static_cast<size_t>(kBoxSize) * kBoxSize; ++p)
    {
        const double v = pixels[p * 4] + pixels[p * 4 + 1] + pixels[p * 4 + 2];
        sum += v;
        lit += v > 0.0 ? 1u : 0u;
    }
    std::printf("light cells: %u of %u pixels lit, mean %.4f\n", lit, kBoxSize * kBoxSize,
                sum / (3.0 * kBoxSize * kBoxSize));
    CHECK(lit > kBoxSize * kBoxSize / 2);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("ReSTIR DI unbiased")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadLightBox(rig);
    REQUIRE(LightStats(rig.scene).bricks == 1);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kBoxSize, kBoxSize);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LightBoxCamera();

    CaelixGiOptions gi{};
    gi.flags = CAELIX_GI_FLAG_LIGHT_NEE;
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> nee = LightReference(rig, renderer, set, camera, 1024, 200000);
    // The noise floor (reported only): light NEE against itself with other seeds.
    const std::vector<float> nee2 = LightReference(rig, renderer, set, camera, 1024, 700000);
    gi.flags = CAELIX_GI_FLAG_LIGHT_NEE | CAELIX_GI_FLAG_RESTIR_DI | CAELIX_GI_FLAG_TEMPORAL |
               CAELIX_GI_FLAG_SPATIAL | CAELIX_GI_FLAG_UNBIASED;
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> di = LightReference(rig, renderer, set, camera, 1024, 400000);

    CaelixGiStats stats{};
    caelix_renderer_get_gi_stats(renderer, &stats);
    const LightComparison c = CompareLight(nee, di, kBoxSize);
    const double psnr = PixelPsnr(nee, di, kBoxSize);
    const double floor_psnr = PixelPsnr(nee, nee2, kBoxSize);
    const LightComparison floor_c = CompareLight(nee, nee2, kBoxSize);
    std::printf("ReSTIR DI (unbiased) vs light NEE, 1024 frames each: mean %.5f vs %.5f, ratio %.4f, "
                "16x16 block PSNR %.2f dB, pixel PSNR %.2f dB; light NEE vs light NEE: ratio %.4f, pixel PSNR "
                "%.2f dB; reservoir bytes %llu, gi %.4f ms\n",
                c.test_mean, c.ref_mean, c.ratio, c.block_psnr, psnr, floor_c.ratio, floor_psnr,
                static_cast<unsigned long long>(stats.reservoir_bytes), stats.gi_ms);
    CHECK(stats.reservoir_bytes == 4ull * 32ull * kBoxSize * kBoxSize);
    CHECK(c.ref_mean > 0.01);
    CHECK(c.ratio > 0.97);
    CHECK(c.ratio < 1.03);
    CHECK(psnr > 30.0);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("ReSTIR DI lowers the error")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadLightBox(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kBoxSize, kBoxSize);
    TargetSet set;
    set.t[kColor] = &color;
    const CaelixCamera camera = LightBoxCamera();

    CaelixGiOptions gi{};
    gi.flags = CAELIX_GI_FLAG_LIGHT_NEE;
    REQUIRE(caelix_renderer_set_gi_options(renderer, &gi) == CAELIX_OK);
    const std::vector<float> reference = LightReference(rig, renderer, set, camera, 1024, 500000);

    // One path trace frame after 64 warm-up frames (the same seeds for each variant).
    const auto single_frame_rmse = [&](uint32_t flags) {
        CaelixGiOptions options{};
        options.flags = flags;
        options.reset = 1;
        REQUIRE(caelix_renderer_set_gi_options(renderer, &options) == CAELIX_OK);
        for (uint32_t i = 0; i <= 64; ++i)
        {
            Render(rig, renderer, set, camera, GlassOptions(CAELIX_MODE_PATH_TRACE, 4, 0, 600000 + i), i == 64);
        }
        return PixelRmse(reference, ColorPixels(color), kBoxSize);
    };
    const double nee = single_frame_rmse(CAELIX_GI_FLAG_LIGHT_NEE);
    const double di = single_frame_rmse(CAELIX_GI_FLAG_LIGHT_NEE | CAELIX_GI_FLAG_RESTIR_DI |
                                        CAELIX_GI_FLAG_TEMPORAL | CAELIX_GI_FLAG_SPATIAL);
    const double di_unbiased =
        single_frame_rmse(CAELIX_GI_FLAG_LIGHT_NEE | CAELIX_GI_FLAG_RESTIR_DI | CAELIX_GI_FLAG_TEMPORAL |
                          CAELIX_GI_FLAG_SPATIAL | CAELIX_GI_FLAG_UNBIASED);
    std::printf("single frame RMSE against the 1024-frame light NEE reference: light NEE %.5f, ReSTIR DI %.5f "
                "(%.3fx), ReSTIR DI unbiased %.5f (%.3fx)\n",
                nee, di, di / nee, di_unbiased, di_unbiased / nee);
    CHECK(di < nee);
    CHECK(di_unbiased < nee);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

TEST_CASE("RESTIR_DI needs LIGHT_NEE")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);

    CaelixGiOptions bad{};
    bad.flags = CAELIX_GI_FLAG_RESTIR_DI | CAELIX_GI_FLAG_TEMPORAL;
    ExpectLibraryErrors(1, [&] {
        REQUIRE(caelix_renderer_set_gi_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
    bad.method = CAELIX_GI_RESTIR;
    ExpectLibraryErrors(1, [&] {
        REQUIRE(caelix_renderer_set_gi_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
    bad.method = CAELIX_GI_PATH_TRACE;
    bad.flags = CAELIX_GI_FLAG_RESTIR_DI | CAELIX_GI_FLAG_LIGHT_NEE;
    bad.di_candidates = 33;
    ExpectLibraryErrors(1, [&] {
        REQUIRE(caelix_renderer_set_gi_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
    CaelixGiOptions good{};
    good.flags = CAELIX_GI_FLAG_RESTIR_DI | CAELIX_GI_FLAG_LIGHT_NEE;
    good.di_candidates = 32;
    CHECK(caelix_renderer_set_gi_options(renderer, &good) == CAELIX_OK);

    caelix_renderer_destroy(renderer);
    CloseRig(rig);
}
