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
// agree on the motion of a moving entity, and render once per frame each. The voxel id
// target holds one key per voxel (two entities, both formats, voxel_id_tmax, flip y, the
// primary-mode voxel id view). The host background replaces the camera-ray misses of the
// lighting modes exactly (no exposure, outside the reference mean) and nothing else. Shadow
// transmission: the primary vertex's sun and lamp shadow rays through a tinted glass slab carry its
// transmittance, the claim keeps the refracted BSDF path from counting the sun twice, an SSS slab
// occludes like any opaque voxel. The light NEE visibility hint
// converges to the sun visibility of a lit and a shadowed floor pixel and follows a turning camera.
// Uses the public C ABI only.
#include "test_scenes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <tuple>
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
        case VK_FORMAT_R32_UINT:
            return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32_SFLOAT:
        case VK_FORMAT_R32G32_UINT:
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
    uint32_t U32(uint32_t x, uint32_t y, uint32_t c) const
    {
        uint32_t value = 0;
        std::memcpy(&value, at(x, y) + 4 * c, 4);
        return value;
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
    kVoxelId,
    kTargetCount
};
struct TargetSet
{
    Target* t[kTargetCount] = {};
    const BackgroundImage* background = nullptr;  // CaelixRenderTargets::background; null = none
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
    out.voxel_id = Abi(set.t[kVoxelId]);
    if (set.background != nullptr)
    {
        out.background = set.background->Abi();
    }
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
        if (all.t[i] != nullptr)
        {
            DestroyTarget(rig.gpu, *all.t[i]);
        }
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

// ---------------------------------------------------------------------------
// Light NEE (docs/LIGHT_NEE.md): the lamp NEE with the sun / lamp choice and the MIS against the plain
// path tracer in a closed box with lamps, a sun opening and glass.

namespace {

constexpr uint16_t kWhiteLamp = 0x4000u | (15u << 10) | (15u << 6) | (15u << 2) | 3u;  // rgb4 1, strength 3
constexpr int32_t kLampBoxInterior = 24;
constexpr uint32_t kLampBoxImage = 128;
// The library default of CaelixLightNeeOptions::hint_patch (kDefaultHintPatch in renderer_light.cpp; chosen by
// the measurement of docs/LIGHT_NEE.md section 12.3).
constexpr uint32_t kDefaultHintPatchExpected = 4;

// A closed grey box around the interior [0, 24)^3 (walls 1 voxel thick) with: an opening of 6 x 6 in
// the ceiling at x, z in [3, 8], its x <= 5 half covered by clear glass (the sun reaches the floor
// directly and through the glass); a grey pillar; single-voxel lamps (strength 3) in the ceiling, in
// the +X wall, in the floor and on top of the pillar (less than 8 units from parts of the pillar: the
// near field), so the NEE sees lamps at every distance and from both sides.
World LampBoxWorld()
{
    World world;
    const int32_t lo = -1;
    const int32_t hi = kLampBoxInterior;
    FillBox(world, lo, lo, lo, hi, lo, hi, kGreyFloor);  // floor
    FillBox(world, lo, hi, lo, hi, hi, hi, kGreyFloor);  // ceiling
    FillBox(world, lo, lo, lo, lo, hi, hi, kGreyFloor);  // -X wall
    FillBox(world, hi, lo, lo, hi, hi, hi, kGreyFloor);  // +X wall
    FillBox(world, lo, lo, lo, hi, hi, lo, kGreyFloor);  // -Z wall
    FillBox(world, lo, lo, hi, hi, hi, hi, kGreyFloor);  // +Z wall
    FillBox(world, 3, hi, 3, 8, hi, 8, 0);               // the ceiling opening (air)
    FillBox(world, 3, hi, 3, 5, hi, 8, kClearGlass);     // half of it glass
    FillBox(world, 15, 0, 9, 17, 13, 11, kGreyFloor);    // the pillar
    FillBox(world, 16, 14, 10, 16, 14, 10, kWhiteLamp);  // on the pillar
    FillBox(world, 18, hi, 18, 18, hi, 18, kWhiteLamp);  // ceiling
    FillBox(world, 10, hi, 20, 10, hi, 20, kWhiteLamp);  // ceiling
    FillBox(world, hi, 6, 12, hi, 6, 12, kWhiteLamp);    // +X wall
    FillBox(world, 6, lo, 18, 6, lo, 18, kWhiteLamp);    // floor
    return world;
}

void UploadLampBox(Rig& rig)
{
    const CaelixLightOptions light{1, 0.0f, 0};  // the light tree with the default threshold
    REQUIRE(caelix_scene_set_light_options(rig.scene, &light) == CAELIX_OK);
    const World world = LampBoxWorld();
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
    // The light tree is built at the flushes of the upload; one more flush settles the BVH.
    FlushFrame(rig.gpu, rig.scene, rig.frames);
}

// Inside the box near the -Z wall, looking along +Z and a little down at the pillar and the floor.
CaelixCamera LampBoxCamera()
{
    const double position[3] = {12.0, 14.0, 1.5};
    const double forward[3] = {0.0, -0.35, 1.0};
    return LookAlong(position, forward, std::tan(40.0 * 3.14159265358979323846 / 180.0));
}

// The lamp box's lighting options: `mode`, `bounces` bounces, 4 transparent bounces, black sky, the sun
// (`sun_radiance` x a 2-degree disk; 0 = no sun) tilted into the opening, sun NEE on, the bounce NEE
// budget `q`.
CaelixRenderOptions LampBoxOptions(uint32_t mode, uint32_t bounces, uint32_t seed, float sun_radiance, float q)
{
    CaelixRenderOptions options = GlassOptions(mode, bounces, 4, seed);
    options.flags = CAELIX_RENDER_FLAG_SUN_NEE;
    const double sun[3] = {0.0, 1.0, -0.4};  // the opening lights the floor around z = 15, in view
    const double length = std::sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]);
    for (int a = 0; a < 3; ++a)
    {
        options.sun_dir[a] = static_cast<float>(sun[a] / length);
    }
    options.sun_radiance[0] = options.sun_radiance[1] = options.sun_radiance[2] = sun_radiance;
    options.sun_angular_radius = static_cast<float>(kNeeSunRadiusDegrees * 3.14159265358979323846 / 180.0);
    options.bounce_nee_probability = q;
    return options;
}

// Reference mode, 4 bounces (LampBoxOptions, bounce NEE budget `q`); the colour (the running mean) after
// `frames` frames.
std::vector<float> LampBoxAccumulate(Rig& rig, CaelixRenderer* renderer, TargetSet& set, uint32_t frames,
                                     uint32_t seed0, float sun_radiance, float q = 1.0f)
{
    const CaelixCamera camera = LampBoxCamera();
    for (uint32_t i = 0; i < frames; ++i)
    {
        CaelixRenderOptions options = LampBoxOptions(CAELIX_MODE_REFERENCE, 4, seed0 + i, sun_radiance, q);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, set, camera, options, i + 1 == frames);
    }
    const Target& color = *set.t[kColor];
    std::vector<float> image(static_cast<size_t>(color.width) * color.height * 4);
    std::memcpy(image.data(), color.bytes.data(), image.size() * sizeof(float));
    return image;
}

// Mean linear brightness (rgb mean over the pixels that do not see a lamp directly), their ratio, and
// the PSNR of 16 x 16 block means (both clamped to [0, 1], peak 1): the block means compare the light
// distribution, the per-pixel noise of the path tracer at a few thousand frames is far above the test's
// tolerance. A pixel whose reference is above kDirectLamp in a channel sees a lamp (emission 512): its
// primary emission is deterministic, equal in both images, and would dominate the means.
constexpr float kDirectLamp = 32.0f;

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
    double values = 0.0;
    uint32_t blocks = 0;
    for (uint32_t by = 0; by < size; by += kBlock)
    {
        for (uint32_t bx = 0; bx < size; bx += kBlock)
        {
            double r_block[3] = {0.0, 0.0, 0.0};
            double t_block[3] = {0.0, 0.0, 0.0};
            uint32_t used = 0;
            for (uint32_t y = by; y < by + kBlock; ++y)
            {
                for (uint32_t x = bx; x < bx + kBlock; ++x)
                {
                    const size_t p = (static_cast<size_t>(y) * size + x) * 4;
                    if (ref[p] > kDirectLamp || ref[p + 1] > kDirectLamp || ref[p + 2] > kDirectLamp)
                    {
                        continue;
                    }
                    ++used;
                    for (int ch = 0; ch < 3; ++ch)
                    {
                        r_block[ch] += ref[p + ch];
                        t_block[ch] += test[p + ch];
                    }
                }
            }
            if (used == 0)
            {
                continue;
            }
            for (int ch = 0; ch < 3; ++ch)
            {
                c.ref_mean += r_block[ch];
                c.test_mean += t_block[ch];
                const double d =
                    std::clamp(t_block[ch] / used, 0.0, 1.0) - std::clamp(r_block[ch] / used, 0.0, 1.0);
                squared += d * d;
            }
            values += 3.0 * used;
            ++blocks;
        }
    }
    c.ref_mean /= values;
    c.test_mean /= values;
    c.ratio = c.test_mean / c.ref_mean;
    const double mse = squared / (3.0 * blocks);
    c.block_psnr = mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 99.0;
    return c;
}

void SetLightNee(CaelixRenderer* renderer, uint32_t enabled, uint32_t flags, uint32_t hint_patch = 0)
{
    CaelixLightNeeOptions options{};
    options.enabled = enabled;
    options.flags = flags;
    options.hint_patch = hint_patch;
    REQUIRE(caelix_renderer_set_light_nee_options(renderer, &options) == CAELIX_OK);
}

}  // namespace

TEST_CASE("light NEE: unbiased against the path tracer (lamps, sun, glass)")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadLampBox(rig);
    CaelixLightStats light{};
    caelix_scene_get_light_stats(rig.scene, &light);
    REQUIRE(light.light_bricks >= 4);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kLampBoxImage, kLampBoxImage);
    TargetSet set;
    set.t[kColor] = &color;

    // The visibility hint runs in the reference mode too (it changes only the sun / lamp choice weights of
    // the primary vertex, so it is unbiased by construction), and so does the bounce NEE budget.
    struct Variant
    {
        const char* name;
        uint32_t flags;
        uint32_t hint_patch;
        float q;  // CaelixRenderOptions::bounce_nee_probability
    };
    const Variant variants[] = {
        {"defaults (MIS, hint)", 0, 0, 1.0f},
        {"no MIS (exclusive rule)", CAELIX_LIGHT_NEE_FLAG_NO_MIS, 0, 1.0f},
        {"no hint", CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT, 0, 1.0f},
        {"hint patch 4", 0, 4, 1.0f},
        {"bounce NEE 0.25", 0, 0, 0.25f},
    };
    // The lamps alone, then the lamps with the sun (the sun / lamp choice, the sun through glass).
    for (const float sun : {0.0f, kNeeSunRadiance})
    {
        // The plain path tracer (the sun NEE only) finds the single-voxel lamps rarely: 4096 frames. The
        // noise floor of that reference (reported only): the path tracer with other seeds.
        SetLightNee(renderer, 0, 0);
        const std::vector<float> pt = LampBoxAccumulate(rig, renderer, set, 4096, 100000, sun);
        const LightComparison floor_c =
            CompareLight(pt, LampBoxAccumulate(rig, renderer, set, 4096, 300000, sun), kLampBoxImage);
        std::printf("sun %.0f: path tracer vs path tracer, other seeds: mean %.5f, ratio %.4f, 16x16 block "
                    "PSNR %.2f dB\n",
                    static_cast<double>(sun), floor_c.ref_mean, floor_c.ratio, floor_c.block_psnr);
        for (const Variant& v : variants)
        {
            SetLightNee(renderer, 1, v.flags, v.hint_patch);
            const std::vector<float> nee = LampBoxAccumulate(rig, renderer, set, 1024, 200000, sun, v.q);
            CaelixLightNeeStats stats{};
            caelix_renderer_get_light_nee_stats(renderer, &stats);
            const LightComparison c = CompareLight(pt, nee, kLampBoxImage);
            std::printf("sun %.0f: light NEE %s vs path tracer: mean %.5f vs %.5f, ratio %.4f, 16x16 block "
                        "PSNR %.2f dB\n",
                        static_cast<double>(sun), v.name, c.test_mean, c.ref_mean, c.ratio, c.block_psnr);
            INFO(v.name << ", sun " << sun);
            CHECK(stats.enabled == 1);
            CHECK(stats.running == 1);
            const bool hint = (v.flags & CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT) == 0;
            CHECK(stats.hint_patch == (hint ? (v.hint_patch != 0 ? v.hint_patch : kDefaultHintPatchExpected) : 0u));
            CHECK(c.ref_mean > 0.01);
            CHECK(c.ratio > 0.97);
            CHECK(c.ratio < 1.03);
            CHECK(c.block_psnr > 30.0);
        }
    }

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

namespace {

// 2000 random bricks scattered in a cube of 48 bricks, a lamp voxel in about every sixth brick;
// `offset_x` shifts the keys.
World LampBlobWorld(uint32_t seed, int32_t offset_x)
{
    std::mt19937 rng(seed);
    World world;
    for (int i = 0; i < 2000; ++i)
    {
        const Key key{static_cast<int32_t>(rng() % 48) - 24 + offset_x, static_cast<int32_t>(rng() % 48) - 24,
                      static_cast<int32_t>(rng() % 48) - 24};
        Brick brick = RandomBrick(rng);
        if (rng() % 6 == 0)
        {
            brick[rng() % 512] = kWhiteLamp;
        }
        world.bricks[key] = brick;
    }
    return world;
}

bool Finite(const Target& color)
{
    for (uint32_t y = 0; y < color.height; ++y)
    {
        for (uint32_t x = 0; x < color.width; ++x)
        {
            for (uint32_t c = 0; c < 4; ++c)
            {
                if (!std::isfinite(color.F32(x, y, c)))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

CaelixLightNeeStats NeeStats(CaelixRenderer* renderer)
{
    CaelixLightNeeStats stats{};
    caelix_renderer_get_light_nee_stats(renderer, &stats);
    return stats;
}

}  // namespace

TEST_CASE("light NEE: lifecycle (entity and group reuse, a paused renderer, modes, no tree, options)")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    const CaelixLightOptions light{1, 0.0f, 0};
    REQUIRE(caelix_scene_set_light_options(rig.scene, &light) == CAELIX_OK);
    World world = LampBlobWorld(5, 0);
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    constexpr uint32_t kImage = 256;
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kImage, kImage);
    TargetSet set;
    set.t[kColor] = &color;

    // Argument checks: nothing changes on an invalid call.
    const auto expect_invalid = [&](const CaelixLightNeeOptions& o) {
        ExpectLibraryErrors(1, [&] {
            REQUIRE(caelix_renderer_set_light_nee_options(renderer, &o) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
    };
    expect_invalid(CaelixLightNeeOptions{2, 0, 0, 0});
    expect_invalid(CaelixLightNeeOptions{1, 1u << 31, 0, 0});
    expect_invalid(CaelixLightNeeOptions{1, CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT << 1, 0, 0});
    expect_invalid(CaelixLightNeeOptions{1, 0, 3, 0});
    expect_invalid(CaelixLightNeeOptions{1, 0, 0, 7});
    REQUIRE(NeeStats(renderer).enabled == 0);

    SetLightNee(renderer, 1, 0);
    const double towards[3] = {0.0, -0.3, 1.0};
    const double position[3] = {8.0, 40.0, -300.0};
    const CaelixCamera look = LookAlong(position, towards, std::tan(35.0 * 3.14159265358979323846 / 180.0));
    uint32_t seed = 1;
    // `count` frames in `mode` with the sun NEE; the stats after the last one.
    const auto frames = [&](uint32_t count, uint32_t mode = CAELIX_MODE_PATH_TRACE) {
        for (uint32_t i = 0; i < count; ++i)
        {
            CaelixRenderOptions o = GlassOptions(mode, 4, 4, seed++);
            o.flags = CAELIX_RENDER_FLAG_SUN_NEE;
            o.sun_dir[1] = 1.0f;
            o.sun_radiance[0] = o.sun_radiance[1] = o.sun_radiance[2] = 50.0f;
            Render(rig, renderer, set, look, o, i + 1 == count);
        }
        REQUIRE(Finite(color));
        return NeeStats(renderer);
    };

    // 1. Enabled: light NEE runs in both lighting modes, not in the primary mode.
    CaelixLightNeeStats s = frames(4);
    CHECK(s.enabled == 1);
    CHECK(s.running == 1);
    CHECK(frames(2, CAELIX_MODE_REFERENCE).running == 1);
    CHECK(frames(2, CAELIX_MODE_PRIMARY).running == 0);
    CHECK(frames(2).running == 1);

    // 2. The entity removed and its id reused in the same flush (a new life), with other bricks: the
    // trace sees only the new light tree's bricks.
    REQUIRE(caelix_scene_remove_entity(rig.scene, rig.entity) == CAELIX_OK);
    CaelixEntityDesc desc{};
    desc.rotation[3] = 1.0f;
    CaelixEntityId reused = CAELIX_INVALID_ENTITY;
    REQUIRE(caelix_scene_add_entity(rig.scene, &desc, &reused) == CAELIX_OK);
    REQUIRE(reused == rig.entity);
    world = LampBlobWorld(6, 1);
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
    CHECK(frames(6).running == 1);

    // 3. Groups released and re-created at once: every brick removed, other lamp bricks uploaded at the
    // same keys in the next flush (new lives of the group indices).
    std::vector<Key> keys = KeysOf(world);
    Remove(rig, keys);
    world = LampBlobWorld(7, 1);
    for (const Key& key : keys)
    {
        if (world.bricks.count(key) == 0)
        {
            world.bricks[key] = Brick{};
        }
    }
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
    CHECK(frames(6).running == 1);

    // 4. A paused renderer: 24 flushes with moves and edits, no render; then rendering goes on.
    for (int i = 0; i < 24; ++i)
    {
        const float p[3] = {static_cast<float>(i % 5), 0.0f, static_cast<float>(i % 3)};
        const float q[4] = {0.0f, std::sin(0.05f * i), 0.0f, std::cos(0.05f * i)};
        REQUIRE(caelix_scene_set_entity_transform(rig.scene, rig.entity, p, q) == CAELIX_OK);
        if (i % 6 == 0)
        {
            std::vector<Key> some(keys.begin(), keys.begin() + std::min<size_t>(keys.size(), 20));
            Remove(rig, some);
        }
        FlushFrame(rig.gpu, rig.scene, rig.frames);
    }
    DrainBuilds(rig);
    CHECK(frames(6).running == 1);

    // 5. The scene's light tree disabled: light NEE does not run (one warning for the whole stretch);
    // enabled again (a new tree), it runs again.
    const CaelixLightOptions no_light{0, 0.0f, 0};
    REQUIRE(caelix_scene_set_light_options(rig.scene, &no_light) == CAELIX_OK);
    DrainBuilds(rig);
    const int warnings_before = g_library_warnings.load();
    s = frames(4);
    CHECK(g_library_warnings.load() - warnings_before == 1);
    g_library_warnings -= g_library_warnings.load() - warnings_before;
    CHECK(s.enabled == 1);
    CHECK(s.running == 0);
    REQUIRE(caelix_scene_set_light_options(rig.scene, &light) == CAELIX_OK);
    DrainBuilds(rig);
    FlushFrame(rig.gpu, rig.scene, rig.frames);
    CHECK(frames(4).running == 1);

    // 6. Disabled, then enabled again without MIS.
    SetLightNee(renderer, 0, 0);
    s = frames(3);
    CHECK(s.enabled == 0);
    CHECK(s.running == 0);
    SetLightNee(renderer, 1, CAELIX_LIGHT_NEE_FLAG_NO_MIS);
    s = frames(3);
    CHECK(s.enabled == 1);
    CHECK(s.running == 1);

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

namespace {

// The pixel (x, y) where the camera `camera` (LookAlong, aspect 1, `size` x `size` pixels) sees the world
// direction `d` (CaelixPrimaryDirection's mapping, pixel centres at + 0.5).
void PixelOfDirection(const CaelixCamera& camera, const double d[3], uint32_t size, double& x, double& y)
{
    double r = 0.0;
    double u = 0.0;
    double f = 0.0;
    for (int c = 0; c < 3; ++c)
    {
        r += d[c] * camera.right[c];
        u += d[c] * camera.up[c];
        f += d[c] * camera.forward[c];
    }
    x = (r / (f * camera.tan_half_fov_y) + 1.0) * 0.5 * size;
    y = (1.0 - u / (f * camera.tan_half_fov_y)) * 0.5 * size;
}

// The world direction of the centre of pixel (x, y) of `camera` (LookAlong, aspect 1).
void DirectionOfPixel(const CaelixCamera& camera, uint32_t x, uint32_t y, uint32_t size, double d[3])
{
    const double nx = (x + 0.5) / size * 2.0 - 1.0;
    const double ny = 1.0 - (y + 0.5) / size * 2.0;
    for (int c = 0; c < 3; ++c)
    {
        d[c] = camera.right[c] * nx * camera.tan_half_fov_y + camera.up[c] * ny * camera.tan_half_fov_y +
               camera.forward[c];
    }
}

}  // namespace

TEST_CASE("light NEE: visibility hint converges and reprojects")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadLampBox(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    constexpr uint32_t kImage = kLampBoxImage;
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT, kImage, kImage);
    Target normal = CreateTarget(rig.gpu, VK_FORMAT_R16G16B16A16_SFLOAT, kImage, kImage);
    TargetSet set;
    set.t[kColor] = &color;
    set.t[kNormal] = &normal;
    const CaelixCamera camera = LampBoxCamera();
    // A small sun (0.25 degrees, so the opening casts a sharp patch of sun on the floor) with the energy of
    // the 2-degree test sun.
    const double sun_radius = 0.25 * 3.14159265358979323846 / 180.0;
    const float sun_radiance = static_cast<float>(kNeeSunRadiance * (kNeeSunRadiusDegrees / 0.25) *
                                                  (kNeeSunRadiusDegrees / 0.25));
    const auto hint_options = [&](uint32_t mode, uint32_t bounces, uint32_t frame_seed, float sun) {
        CaelixRenderOptions options = LampBoxOptions(mode, bounces, frame_seed, sun, 1.0f);
        options.sun_angular_radius = static_cast<float>(sun_radius);
        return options;
    };

    // 1. The sun visibility of the primary vertices without the hint: light NEE off, reference mode, one
    //    bounce, the sun on and off with the same seeds. The lamps (found by the BSDF path only) consume
    //    the same random numbers in both runs, and with one bounce the sun reaches the image only by the
    //    primary vertex's NEE (the bounce segment sees no sun disk), so the difference is the direct sun.
    SetLightNee(renderer, 0, 0);
    const auto accumulate = [&](float sun) {
        constexpr uint32_t kFrames = 64;
        for (uint32_t i = 0; i < kFrames; ++i)
        {
            CaelixRenderOptions options = hint_options(CAELIX_MODE_REFERENCE, 1, 7000 + i, sun);
            options.reset_accumulation = i == 0 ? 1u : 0u;
            Render(rig, renderer, set, camera, options, i + 1 == kFrames);
        }
        std::vector<float> image(static_cast<size_t>(kImage) * kImage * 4);
        std::memcpy(image.data(), color.bytes.data(), image.size() * sizeof(float));
        return image;
    };
    const std::vector<float> lamps_only = accumulate(0.0f);
    const std::vector<float> with_sun = accumulate(sun_radiance);
    const auto direct_sun = [&](uint32_t x, uint32_t y) {
        const size_t p = (static_cast<size_t>(y) * kImage + x) * 4;
        return with_sun[p] - lamps_only[p];
    };
    // A pixel that sees a lamp directly (its emission dominates; the floor lamp lies in the sun).
    const auto sees_lamp = [&](uint32_t x, uint32_t y) {
        return lamps_only[(static_cast<size_t>(y) * kImage + x) * 4] > kDirectLamp;
    };
    float max_sun = 0.0f;
    for (uint32_t y = 0; y < kImage; ++y)
    {
        for (uint32_t x = 0; x < kImage; ++x)
        {
            if (!sees_lamp(x, y))
            {
                max_sun = std::max(max_sun, direct_sun(x, y));
            }
        }
    }
    REQUIRE(max_sun > 0.1f);
    // A lit and a shadowed pixel on upward-facing surfaces (the sun candidate exists there) whose 5 x 5
    // neighbourhood is lit or shadowed alike and sees no lamp directly (so the default hint patch and the
    // reprojection below stay inside it); the candidates closest to the image centre.
    constexpr int kMargin = 2;
    int lit[2] = {-1, -1};
    int shadow[2] = {-1, -1};
    int lit_d = 1 << 30;
    int shadow_d = 1 << 30;
    for (int y = kMargin; y < static_cast<int>(kImage) - kMargin; ++y)
    {
        for (int x = kMargin; x < static_cast<int>(kImage) - kMargin; ++x)
        {
            bool all_lit = true;
            bool all_dark = true;
            for (int dy = -kMargin; dy <= kMargin; ++dy)
            {
                for (int dx = -kMargin; dx <= kMargin; ++dx)
                {
                    const uint32_t nx = static_cast<uint32_t>(x + dx);
                    const uint32_t ny = static_cast<uint32_t>(y + dy);
                    const float s = direct_sun(nx, ny);
                    const bool up = normal.F16(nx, ny, 1) > 0.9f && normal.F16(nx, ny, 3) > 0.5f;
                    const bool lamp = sees_lamp(nx, ny);
                    all_lit = all_lit && up && !lamp && s > 0.9f * max_sun;
                    all_dark = all_dark && up && !lamp && s == 0.0f;
                }
            }
            const int d = std::abs(x - static_cast<int>(kImage) / 2) + std::abs(y - static_cast<int>(kImage) / 2);
            if (all_lit && d < lit_d)
            {
                lit[0] = x;
                lit[1] = y;
                lit_d = d;
            }
            if (all_dark && d < shadow_d)
            {
                shadow[0] = x;
                shadow[1] = y;
                shadow_d = d;
            }
        }
    }
    std::printf("hint test: max direct sun %.4f, lit pixel (%d, %d), shadowed pixel (%d, %d)\n",
                static_cast<double>(max_sun), lit[0], lit[1], shadow[0], shadow[1]);
    REQUIRE(lit[0] >= 0);
    REQUIRE(shadow[0] >= 0);

    // 2. The hint (default patch 2) after 64 path trace frames without the denoiser, through the hint view.
    SetLightNee(renderer, 1, 0);
    uint32_t seed = 9000;
    const auto hint_frames = [&](uint32_t count, const CaelixCamera& view) {
        for (uint32_t i = 0; i < count; ++i)
        {
            CaelixRenderOptions options = hint_options(CAELIX_MODE_PATH_TRACE, 4, seed++, sun_radiance);
            options.debug_view = CAELIX_VIEW_LIGHT_HINT;
            options.exposure = 1.0f;
            Render(rig, renderer, set, view, options, i + 1 == count);
        }
    };
    hint_frames(64, camera);
    CHECK(NeeStats(renderer).hint_patch == kDefaultHintPatchExpected);
    bool g_valid = true;
    for (uint32_t y = 0; y < kImage; ++y)
    {
        for (uint32_t x = 0; x < kImage; ++x)
        {
            const float g = color.F32(x, y, 1);
            g_valid = g_valid && std::isfinite(g) && g >= 0.0f && g <= 1.0f;
        }
    }
    CHECK(g_valid);
    const float lit_r = color.F32(static_cast<uint32_t>(lit[0]), static_cast<uint32_t>(lit[1]), 0);
    const float shadow_r = color.F32(static_cast<uint32_t>(shadow[0]), static_cast<uint32_t>(shadow[1]), 0);
    const float lit_g = color.F32(static_cast<uint32_t>(lit[0]), static_cast<uint32_t>(lit[1]), 1);
    const float shadow_g = color.F32(static_cast<uint32_t>(shadow[0]), static_cast<uint32_t>(shadow[1]), 1);
    std::printf("hint after 64 frames: lit pixel sun %.3f lamp %.3f, shadowed pixel sun %.3f lamp %.3f\n",
                static_cast<double>(lit_r), static_cast<double>(lit_g), static_cast<double>(shadow_r),
                static_cast<double>(shadow_g));
    CHECK(lit_r > 0.8f);
    CHECK(shadow_r < 0.2f);

    // 3. Reprojection: the camera turns by about one pixel per frame (yaw about the world Y axis, no
    //    parallax: a world direction moves to a pixel computed exactly) for 8 frames; the hint at the
    //    pixels where the two points now lie stays within 0.2 of its value.
    double lit_dir[3];
    double shadow_dir[3];
    DirectionOfPixel(camera, static_cast<uint32_t>(lit[0]), static_cast<uint32_t>(lit[1]), kImage, lit_dir);
    DirectionOfPixel(camera, static_cast<uint32_t>(shadow[0]), static_cast<uint32_t>(shadow[1]), kImage,
                     shadow_dir);
    const double pixel_angle = std::atan(2.0 * camera.tan_half_fov_y / kImage);
    CaelixCamera turned = camera;
    for (uint32_t k = 1; k <= 8; ++k)
    {
        const double a = pixel_angle * k;
        const double f[3] = {camera.forward[0] * std::cos(a) + camera.forward[2] * std::sin(a), camera.forward[1],
                             -camera.forward[0] * std::sin(a) + camera.forward[2] * std::cos(a)};
        const double position[3] = {camera.position[0], camera.position[1], camera.position[2]};
        turned = LookAlong(position, f, camera.tan_half_fov_y);
        hint_frames(1, turned);
    }
    double lx = 0.0;
    double ly = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    PixelOfDirection(turned, lit_dir, kImage, lx, ly);
    PixelOfDirection(turned, shadow_dir, kImage, sx, sy);
    const float lit_r_moved = color.F32(static_cast<uint32_t>(lx), static_cast<uint32_t>(ly), 0);
    const float shadow_r_moved = color.F32(static_cast<uint32_t>(sx), static_cast<uint32_t>(sy), 0);
    std::printf("hint after 8 turned frames: lit point at (%.2f, %.2f) sun %.3f, shadowed point at (%.2f, %.2f) "
                "sun %.3f\n",
                lx, ly, static_cast<double>(lit_r_moved), sx, sy, static_cast<double>(shadow_r_moved));
    CHECK(std::abs(lx - lit[0]) > 4.0);  // the points moved by several pixels
    CHECK(std::abs(lit_r_moved - lit_r) < 0.2f);
    CHECK(std::abs(shadow_r_moved - shadow_r) < 0.2f);

    // 4. Every patch size runs (the stats report it; a new patch recreates the internal targets and the
    //    hint starts from 0.5 again), the hint switched off reports 0, an invalid patch is rejected.
    for (const uint32_t patch : {1u, 2u, 4u})
    {
        SetLightNee(renderer, 1, 0, patch);
        hint_frames(4, camera);
        CHECK(NeeStats(renderer).hint_patch == patch);
        CHECK(Finite(color));
    }
    SetLightNee(renderer, 1, CAELIX_LIGHT_NEE_FLAG_NO_VISIBILITY_HINT);
    hint_frames(2, camera);
    CHECK(NeeStats(renderer).hint_patch == 0);
    // The hint view without the hint shows 0.5.
    CHECK(color.F32(static_cast<uint32_t>(lit[0]), static_cast<uint32_t>(lit[1]), 0) == 0.5f);
    ExpectLibraryErrors(1, [&] {
        const CaelixLightNeeOptions bad{1, 0, 3, 0};
        REQUIRE(caelix_renderer_set_light_nee_options(renderer, &bad) == CAELIX_ERROR_INVALID_ARGUMENT);
    });

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    DestroyTarget(rig.gpu, normal);
    CloseRig(rig);
}

namespace {

// The voxel id test: the camera looks straight down (forward -Y, up +Z, right +X) with a
// vertical field of view of 60 degrees.
CaelixCamera LookDown60(float x, float height, float z)
{
    CaelixCamera camera = LookDownFrom(x, height, z);
    camera.tan_half_fov_y = std::tan(30.0f * kPi / 180.0f);
    NormalZProjection(camera.projection, camera.tan_half_fov_y, camera.aspect);
    return camera;
}

// The voxel that the CPU expects at a pixel of the voxel id test: the floor of UploadFloor (entity
// A, voxels x, z in [0, 64), top layer y = -1) and, with `entity_b`, the same bricks as entity B,
// turned 180 degrees about Y with its origin at (0, 0, 64): local (x, y, z) -> world (-x, y, 64 - z).
struct CpuVoxel
{
    enum Kind
    {
        kAmbiguous,  // within 1e-3 of a voxel border: either voxel is right
        kMiss,
        kHit
    };
    Kind kind = kAmbiguous;
    int entity = 0;  // 0 = A, 1 = B
    int x = 0;       // entity-local voxel
    int y = 0;
    int z = 0;

    bool operator<(const CpuVoxel& o) const
    {
        return std::tie(entity, x, y, z) < std::tie(o.entity, o.x, o.y, o.z);
    }
};

CpuVoxel FloorVoxel(const CaelixCamera& camera, uint32_t px, uint32_t py, bool entity_b)
{
    // The unjittered ray of the pixel centre (CaelixCamera's mapping) meets the floor top y = 0 at
    // position + (right * ndc.x * t * aspect + up * ndc.y * t + forward) * height.
    const double ndc_x = (px + 0.5) / kSize * 2.0 - 1.0;
    const double ndc_y = 1.0 - (py + 0.5) / kSize * 2.0;
    const double height = camera.position[1];
    const double wx = camera.position[0] + ndc_x * camera.tan_half_fov_y * camera.aspect * height;
    const double wz = camera.position[2] + ndc_y * camera.tan_half_fov_y * height;
    const auto near_integer = [](double v) { return std::abs(v - std::round(v)) < 1e-3; };
    CpuVoxel out;
    if (near_integer(wx) || near_integer(wz))
    {
        return out;
    }
    REQUIRE(wz > 0.0);
    REQUIRE(wz < 64.0);
    REQUIRE(wx < 64.0);
    out.y = -1;
    if (wx >= 0.0)
    {
        out.kind = CpuVoxel::kHit;
        out.x = static_cast<int>(std::floor(wx));
        out.z = static_cast<int>(std::floor(wz));
        return out;
    }
    if (!entity_b || wx < -64.0)
    {
        out.kind = CpuVoxel::kMiss;
        return out;
    }
    out.kind = CpuVoxel::kHit;
    out.entity = 1;
    out.x = static_cast<int>(std::floor(-wx));
    out.z = static_cast<int>(std::floor(64.0 - wz));
    return out;
}

uint64_t VoxelKey(const Target& target, uint32_t x, uint32_t y)
{
    return (static_cast<uint64_t>(target.U32(x, y, 1)) << 32) | target.U32(x, y, 0);
}

constexpr uint64_t kVoxelMiss = ~0ull;

// Checks a voxel id image (R32G32_UINT) against the CPU voxels: one key per CPU voxel, one CPU
// voxel per key, the miss value exactly on the misses. Returns the number of distinct keys.
size_t CheckVoxelIds(const Target& ids, const CaelixCamera& camera, bool entity_b, const char* label)
{
    std::map<CpuVoxel, std::set<uint64_t>> keys_of_voxel;
    std::map<uint64_t, std::set<CpuVoxel>> voxels_of_key;
    size_t misses = 0;
    size_t bad_misses = 0;
    size_t ambiguous = 0;
    for (uint32_t y = 0; y < kSize; ++y)
    {
        for (uint32_t x = 0; x < kSize; ++x)
        {
            const CpuVoxel voxel = FloorVoxel(camera, x, y, entity_b);
            const uint64_t key = VoxelKey(ids, x, y);
            if (voxel.kind == CpuVoxel::kAmbiguous)
            {
                ++ambiguous;
                continue;
            }
            if (voxel.kind == CpuVoxel::kMiss)
            {
                ++misses;
                bad_misses += key != kVoxelMiss ? 1 : 0;
                continue;
            }
            // Neither word of a hit is the miss value.
            if (ids.U32(x, y, 0) == 0xFFFFFFFFu || ids.U32(x, y, 1) == 0xFFFFFFFFu)
            {
                ++bad_misses;
                continue;
            }
            keys_of_voxel[voxel].insert(key);
            voxels_of_key[key].insert(voxel);
        }
    }
    size_t split_voxels = 0;
    for (const auto& [voxel, keys] : keys_of_voxel)
    {
        split_voxels += keys.size() > 1 ? 1 : 0;
    }
    size_t shared_keys = 0;
    for (const auto& [key, voxels] : voxels_of_key)
    {
        shared_keys += voxels.size() > 1 ? 1 : 0;
    }
    std::printf("voxel id %s: %zu CPU voxels, %zu GPU keys, %zu miss pixels, %zu ambiguous pixels, %zu wrong "
                "miss/hit pixels, %zu voxels with two keys, %zu keys of two voxels\n",
                label, keys_of_voxel.size(), voxels_of_key.size(), misses, ambiguous, bad_misses, split_voxels,
                shared_keys);
    CHECK(bad_misses == 0);
    CHECK(split_voxels == 0);
    CHECK(shared_keys == 0);
    CHECK(keys_of_voxel.size() == voxels_of_key.size());
    return voxels_of_key.size();
}

}  // namespace

TEST_CASE("voxel id target: one key per voxel, miss value, tmax, both formats, entities, flip y, debug view")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target ids = CreateTarget(rig.gpu, VK_FORMAT_R32G32_UINT);
    Target ids_x = CreateTarget(rig.gpu, VK_FORMAT_R32_UINT);
    Target albedo = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R8G8B8A8_UNORM);
    // The camera above the floor's -X edge: the left half of the image misses entity A.
    const CaelixCamera camera = LookDown60(0.0f, 12.0f, 32.0f);
    const auto all_miss = [&](const Target& target) {
        size_t hits = 0;
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                hits += VoxelKey(target, x, y) != kVoxelMiss ? 1 : 0;
            }
        }
        return hits == 0;
    };
    const auto same_image = [](const Target& a, const Target& b) { return a.bytes == b.bytes; };
    TargetSet set;
    set.t[kAlbedo] = &albedo;
    set.t[kVoxelId] = &ids;

    // 0. Before the first TLAS every pixel is the miss value.
    Render(rig, renderer, set, camera, Options(CAELIX_VIEW_ALBEDO));
    CHECK(all_miss(ids));

    // 1. Entity A: one key per voxel, the miss value on the misses, coverage agrees.
    UploadFloor(rig);
    Render(rig, renderer, set, camera, Options(CAELIX_VIEW_ALBEDO));
    const size_t keys_a = CheckVoxelIds(ids, camera, false, "entity A");
    CHECK(keys_a > 50);  // 7 x 14 voxels
    size_t coverage_mismatch = 0;
    for (uint32_t y = 0; y < kSize; ++y)
    {
        for (uint32_t x = 0; x < kSize; ++x)
        {
            const bool covered = albedo.at(x, y)[3] == 255;
            const bool keyed = ids.U32(x, y, 0) != 0xFFFFFFFFu && ids.U32(x, y, 1) != 0xFFFFFFFFu;
            const bool missed = VoxelKey(ids, x, y) == kVoxelMiss;
            coverage_mismatch += (covered ? !keyed : !missed) ? 1 : 0;
        }
    }
    CHECK(coverage_mismatch == 0);
    const Target unflipped = ids;

    // 2. R32_UINT receives word 0 of the same keys; the path trace mode writes the same keys.
    {
        TargetSet single;
        single.t[kVoxelId] = &ids_x;
        Render(rig, renderer, single, camera, Options(CAELIX_VIEW_ALBEDO));
        size_t differ = 0;
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                differ += ids_x.U32(x, y, 0) != unflipped.U32(x, y, 0) ? 1 : 0;
            }
        }
        CHECK(differ == 0);
        CaelixRenderOptions path_trace = Options(CAELIX_VIEW_ALBEDO);
        path_trace.mode = CAELIX_MODE_PATH_TRACE;
        path_trace.bounce_count = 0;
        Render(rig, renderer, set, camera, path_trace);
        CHECK(same_image(ids, unflipped));
    }

    // 3. voxel_id_tmax: half the camera height cuts every hit (the floor is 12 units away), 1000
    //    cuts none.
    {
        CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
        options.voxel_id_tmax = 6.0f;
        Render(rig, renderer, set, camera, options);
        CHECK(all_miss(ids));
        options.voxel_id_tmax = 1000.0f;
        Render(rig, renderer, set, camera, options);
        CHECK(same_image(ids, unflipped));
    }

    // 4. FLIP_Y: the key image is the row mirror of the unflipped one.
    {
        CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
        options.flags = CAELIX_RENDER_FLAG_FLIP_Y;
        Render(rig, renderer, set, camera, options);
        size_t differ = 0;
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                differ += VoxelKey(ids, x, kSize - 1 - y) != VoxelKey(unflipped, x, y) ? 1 : 0;
            }
        }
        CHECK(differ == 0);
    }

    // 5. The primary-mode debug view: one colour per voxel, the sky on a miss and beyond tmax.
    {
        TargetSet view;
        view.t[kColor] = &color;
        view.t[kVoxelId] = &ids;
        Render(rig, renderer, view, camera, Options(CAELIX_VIEW_VOXEL_ID));
        std::map<CpuVoxel, std::set<uint32_t>> colours;
        size_t bad_sky = 0;
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                const CpuVoxel voxel = FloorVoxel(camera, x, y, false);
                if (voxel.kind == CpuVoxel::kMiss)
                {
                    bad_sky += IsSky(color.at(x, y)) ? 0 : 1;
                }
                else if (voxel.kind == CpuVoxel::kHit)
                {
                    uint32_t rgba = 0;
                    std::memcpy(&rgba, color.at(x, y), 4);
                    colours[voxel].insert(rgba);
                }
            }
        }
        size_t split = 0;
        std::set<uint32_t> distinct;
        for (const auto& [voxel, set_of_colours] : colours)
        {
            split += set_of_colours.size() > 1 ? 1 : 0;
            distinct.insert(set_of_colours.begin(), set_of_colours.end());
        }
        std::printf("voxel id view: %zu voxels, %zu distinct colours, %zu voxels with two colours, %zu miss "
                    "pixels not sky\n",
                    colours.size(), distinct.size(), split, bad_sky);
        CHECK(split == 0);
        CHECK(bad_sky == 0);
        CHECK(distinct.size() > colours.size() / 2);
        CaelixRenderOptions options = Options(CAELIX_VIEW_VOXEL_ID);
        options.voxel_id_tmax = 6.0f;
        Render(rig, renderer, view, camera, options);
        size_t not_sky = 0;
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                not_sky += IsSky(color.at(x, y)) ? 0 : 1;
            }
        }
        CHECK(not_sky == 0);
    }

    // 6. Argument checks: a voxel id target in RGBA8, a negative or NaN voxel_id_tmax.
    {
        CaelixRenderTargets targets{};
        targets.voxel_id = Abi(&color);
        ExpectRenderError(rig, renderer, targets, camera, CAELIX_ERROR_INVALID_ARGUMENT);
        targets.voxel_id = Abi(&ids);
        for (const float tmax : {-1.0f, std::numeric_limits<float>::quiet_NaN()})
        {
            ExpectLibraryErrors(1, [&] {
                CaelixFrameDesc desc{};
                desc.command_buffer = 1;
                desc.frame_number = rig.frames.next;
                CaelixRenderOptions options = Options(CAELIX_VIEW_ALBEDO);
                options.voxel_id_tmax = tmax;
                REQUIRE(caelix_renderer_render(renderer, &desc, &targets, &camera, &options) ==
                        CAELIX_ERROR_INVALID_ARGUMENT);
            });
        }
    }

    // 7. Entity B: the same bricks (the same entity-local voxels), turned 180 degrees about Y with its
    //    origin at (0, 0, 64), so it fills the left half of the image with the voxels that entity A
    //    shows on the right. No key of B equals a key of A.
    {
        CaelixEntityDesc desc{};
        desc.position[2] = 64.0f;
        desc.rotation[1] = 1.0f;  // 180 degrees about Y
        CaelixEntityId entity_b = CAELIX_INVALID_ENTITY;
        REQUIRE(caelix_scene_add_entity(rig.scene, &desc, &entity_b) == CAELIX_OK);
        const CaelixEntityId saved = rig.entity;
        rig.entity = entity_b;
        UploadFloor(rig);
        rig.entity = saved;
        Render(rig, renderer, set, camera, Options(CAELIX_VIEW_ALBEDO));
        CheckVoxelIds(ids, camera, true, "entities A and B");
        std::set<uint64_t> keys_of[2];
        std::set<std::array<int, 3>> local_of[2];
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                const CpuVoxel voxel = FloorVoxel(camera, x, y, true);
                if (voxel.kind == CpuVoxel::kHit)
                {
                    keys_of[voxel.entity].insert(VoxelKey(ids, x, y));
                    local_of[voxel.entity].insert({voxel.x, voxel.y, voxel.z});
                }
            }
        }
        size_t shared_local = 0;
        for (const std::array<int, 3>& local : local_of[0])
        {
            shared_local += local_of[1].count(local);
        }
        size_t shared_keys = 0;
        for (const uint64_t key : keys_of[0])
        {
            shared_keys += keys_of[1].count(key);
        }
        std::printf("voxel id entities: A %zu keys, B %zu keys, %zu local voxels seen in both, %zu shared keys\n",
                    keys_of[0].size(), keys_of[1].size(), shared_local, shared_keys);
        CHECK(shared_local > 50);
        CHECK(shared_keys == 0);
    }

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, ids);
    DestroyTarget(rig.gpu, ids_x);
    DestroyTarget(rig.gpu, albedo);
    DestroyTarget(rig.gpu, color);
    CloseRig(rig);
}

namespace {

// The background test: a host image of kSize x kSize, pixel (x, y) = (x / w + offset, y / h, blue, 1).
std::vector<float> GradientBackground(float offset, float blue)
{
    std::vector<float> rgba(static_cast<size_t>(kSize) * kSize * 4);
    for (uint32_t y = 0; y < kSize; ++y)
    {
        for (uint32_t x = 0; x < kSize; ++x)
        {
            float* p = rgba.data() + (static_cast<size_t>(y) * kSize + x) * 4;
            p[0] = static_cast<float>(x) / kSize + offset;
            p[1] = static_cast<float>(y) / kSize;
            p[2] = blue;
            p[3] = 1.0f;
        }
    }
    return rgba;
}

// Counts the pixels of an RGBA32F colour image by coverage (alpha 0 = miss) and the misses whose rgb is
// not the background pixel at the same stored position, or whose alpha is not exactly 0.
struct MissCheck
{
    size_t misses = 0;
    size_t hits = 0;
    size_t wrong = 0;
};
MissCheck CheckMisses(const Target& color, const std::vector<float>& background)
{
    MissCheck out;
    for (uint32_t y = 0; y < kSize; ++y)
    {
        for (uint32_t x = 0; x < kSize; ++x)
        {
            if (color.F32(x, y, 3) != 0.0f)
            {
                ++out.hits;
                continue;
            }
            ++out.misses;
            const float* expected = background.data() + (static_cast<size_t>(y) * kSize + x) * 4;
            for (uint32_t c = 0; c < 3; ++c)
            {
                if (color.F32(x, y, c) != expected[c])
                {
                    ++out.wrong;
                    break;
                }
            }
        }
    }
    return out;
}

// Hit pixels (alpha 1 in `a`) whose 16 bytes differ between two RGBA32F images, plus every pixel whose
// coverage differs.
size_t HitDifferences(const Target& a, const Target& b)
{
    size_t differ = 0;
    for (uint32_t y = 0; y < kSize; ++y)
    {
        for (uint32_t x = 0; x < kSize; ++x)
        {
            const bool hit = a.F32(x, y, 3) != 0.0f;
            if (a.F32(x, y, 3) != b.F32(x, y, 3) || (hit && std::memcmp(a.at(x, y), b.at(x, y), 16) != 0))
            {
                ++differ;
            }
        }
    }
    return differ;
}

}  // namespace

TEST_CASE("background: misses show the host image, hits do not, exposure and accumulation untouched, argument checks")
{
    Rig rig;
    OpenRig(rig, "caelix_trace_test", RigDesc(0, 0, 0));
    UploadFloor(rig);
    CaelixRenderer* renderer = nullptr;
    REQUIRE(caelix_renderer_create(rig.device, rig.scene, &renderer) == CAELIX_OK);
    Target color = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    Target plain = CreateTarget(rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    const std::vector<float> pixels_a = GradientBackground(0.0f, 0.25f);
    const std::vector<float> pixels_b = GradientBackground(0.5f, 2.0f);
    BackgroundImage background_a = CreateBackground(rig.gpu, kSize, kSize);
    BackgroundImage background_b = CreateBackground(rig.gpu, kSize, kSize);
    UploadBackground(rig.gpu, background_a, pixels_a);
    UploadBackground(rig.gpu, background_b, pixels_b);
    // Above the floor's middle, looking along +Z: the upper half and the far band miss.
    const CaelixCamera camera = LookPlusZ(32.0f, 6.0f, 2.0f);
    const auto path_trace = [](uint32_t mode, uint32_t seed, uint32_t flags) {
        CaelixRenderOptions options = LightOptions(mode, 2, seed);
        options.flags = flags;
        options.exposure = 2.0f;
        return options;
    };
    TargetSet with;
    with.t[kColor] = &color;
    with.background = &background_a;
    TargetSet without;
    without.t[kColor] = &plain;

    // 1. Every miss is the background pixel exactly (alpha 0), with and without SKY_ON_MISS; with
    //    FLIP_Y the stored image still pairs pixel by pixel with the (equally stored) background.
    Render(rig, renderer, with, camera, path_trace(CAELIX_MODE_PATH_TRACE, 11, CAELIX_RENDER_FLAG_SKY_ON_MISS));
    const MissCheck sky_flag = CheckMisses(color, pixels_a);
    const Target first = color;  // the readback only (the image stays `color`'s)
    Render(rig, renderer, with, camera, path_trace(CAELIX_MODE_PATH_TRACE, 11, 0));
    const MissCheck no_flag = CheckMisses(color, pixels_a);
    Render(rig, renderer, with, camera,
           path_trace(CAELIX_MODE_PATH_TRACE, 11, CAELIX_RENDER_FLAG_SKY_ON_MISS | CAELIX_RENDER_FLAG_FLIP_Y));
    const MissCheck flipped = CheckMisses(color, pixels_a);
    CHECK(sky_flag.misses > 10000);
    CHECK(sky_flag.hits > 10000);
    CHECK(sky_flag.wrong == 0);
    CHECK(no_flag.misses == sky_flag.misses);
    CHECK(no_flag.wrong == 0);
    CHECK(flipped.misses == sky_flag.misses);
    CHECK(flipped.wrong == 0);

    // 2. The hits are the render without a background, bit for bit (same seed, exposure 2).
    Render(rig, renderer, without, camera, path_trace(CAELIX_MODE_PATH_TRACE, 11, CAELIX_RENDER_FLAG_SKY_ON_MISS));
    const size_t hit_differences = HitDifferences(first, plain);
    CHECK(hit_differences == 0);
    // Without a background the misses are the exposed sky, not the host image.
    const MissCheck sky_misses = CheckMisses(plain, pixels_a);
    CHECK(sky_misses.wrong == sky_misses.misses);

    // 3. Reference mode, 4 frames, the background alternating A, B, A, B: the hits are the 4-frame mean of
    //    the scene (the run without a background), the misses the last background (B).
    constexpr uint32_t kFrames = 4;
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        CaelixRenderOptions options = path_trace(CAELIX_MODE_REFERENCE, 100 + i, CAELIX_RENDER_FLAG_SKY_ON_MISS);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        with.background = i % 2 == 0 ? &background_a : &background_b;
        Render(rig, renderer, with, camera, options, i + 1 == kFrames);
    }
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        CaelixRenderOptions options = path_trace(CAELIX_MODE_REFERENCE, 100 + i, CAELIX_RENDER_FLAG_SKY_ON_MISS);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, without, camera, options, i + 1 == kFrames);
    }
    const MissCheck reference = CheckMisses(color, pixels_b);
    const size_t reference_differences = HitDifferences(color, plain);
    CHECK(reference.misses == sky_flag.misses);
    CHECK(reference.wrong == 0);
    CHECK(reference_differences == 0);
    with.background = &background_a;

    // 4. Argument checks (every mode): a background of the wrong size, in layout UNDEFINED, without a
    //    format, and one that is a target's image.
    {
        BackgroundImage small = CreateBackground(rig.gpu, kSize / 2, kSize / 2);
        UploadBackground(rig.gpu, small, std::vector<float>(static_cast<size_t>(kSize / 2) * (kSize / 2) * 4, 0.5f));
        CaelixRenderTargets targets{};
        targets.color = Abi(&color);
        targets.background = small.Abi();
        ExpectRenderError(rig, renderer, targets, camera, CAELIX_ERROR_INVALID_ARGUMENT);
        targets.background = background_a.Abi();
        targets.background.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        ExpectRenderError(rig, renderer, targets, camera, CAELIX_ERROR_INVALID_ARGUMENT);
        targets.background = background_a.Abi();
        targets.background.format = VK_FORMAT_UNDEFINED;
        ExpectRenderError(rig, renderer, targets, camera, CAELIX_ERROR_INVALID_ARGUMENT);
        targets.background = Abi(&color);
        targets.background.layout = VK_IMAGE_LAYOUT_GENERAL;
        ExpectRenderError(rig, renderer, targets, camera, CAELIX_ERROR_INVALID_ARGUMENT);
        REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
        DestroyBackground(rig.gpu, small);
    }

    // 5. The primary mode ignores the background: the same image as without it.
    Render(rig, renderer, with, camera, Options(CAELIX_VIEW_ALBEDO));
    Render(rig, renderer, without, camera, Options(CAELIX_VIEW_ALBEDO));
    const bool primary_unchanged = color.bytes == plain.bytes;
    CHECK(primary_unchanged);

    std::printf("background: %zu miss / %zu hit pixels; misses not the host image: %zu (sky flag), %zu (no flag), "
                "%zu (flip y); hit pixels differing from the render without it: %zu; reference 4 frames: %zu "
                "misses not the last image, %zu hits differing from the plain mean; primary mode unchanged: %s\n",
                sky_flag.misses, sky_flag.hits, sky_flag.wrong, no_flag.wrong, flipped.wrong, hit_differences,
                reference.wrong, reference_differences, primary_unchanged ? "yes" : "no");

    REQUIRE(vkDeviceWaitIdle(rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(renderer);
    caelix_renderer_destroy(renderer);
    DestroyTarget(rig.gpu, color);
    DestroyTarget(rig.gpu, plain);
    DestroyBackground(rig.gpu, background_a);
    DestroyBackground(rig.gpu, background_b);
    CloseRig(rig);
}

// ---------------------------------------------------------------------------
// Shadow transmission (docs/SHADOW_TRANSMISSION.md): the primary vertex's NEE shadow rays pass
// tinted glass in a straight line, attenuated; the claim keeps the refracted BSDF path
// from counting the sun twice.

namespace {

// Packed codes of the shadow transmission scenes. The floor is white (rgb5 31, albedo exactly 1 in the
// RGBA8 albedo target); the tinted glass, the SSS slab and the plain block get host entries.
constexpr uint16_t kWhiteFloor = 0x8000u | (31u << 10) | (31u << 5) | 31u;
constexpr uint16_t kTintGlass = 200;  // a reserved palette id (vacuum by default)
constexpr uint16_t kSssCode = 0x8000u | (10u << 10) | (20u << 5) | 5u;
constexpr uint16_t kPlainCode = 0x8000u | (20u << 10) | (10u << 5) | 5u;
constexpr uint16_t kBlackWall = 0x8000u;  // rgb5 0: albedo 0
constexpr float kTintAlbedo[3] = {1.0f, 0.5f, 0.25f};
constexpr float kTintExtinction = 0.2f;
constexpr float kTintThickness = 4.0f;

// The tinted glass: albedo (1, 0.5, 0.25), extinction 0.2, ior 1.5, smoothness 1 (a delta interface).
CaelixMaterialEntry TintGlassEntry()
{
    CaelixMaterialEntry e{};
    e.code = kTintGlass;
    Set3(e.material.albedo, kTintAlbedo);
    e.material.smoothness = 1.0f;
    e.material.ior = 1.5f;
    e.material.extinction = kTintExtinction;
    return e;
}

// An opaque SSS entry: albedo 0.5, model HG, sss_albedo 0.8, 16 steps at g = 0.
CaelixMaterialEntry SssEntry(float probability)
{
    CaelixMaterialEntry e{};
    e.code = kSssCode;
    e.material.albedo[0] = e.material.albedo[1] = e.material.albedo[2] = 0.5f;
    e.material.ior = 1.5f;
    e.material.sss_model = CAELIX_SSS_MODEL_HG;
    e.material.sss_probability = probability;
    e.material.sss_albedo[0] = e.material.sss_albedo[1] = e.material.sss_albedo[2] = 0.8f;
    e.material.sss_steps = 16u;
    return e;
}

// The expected transmittance of the tinted slab, channel c: exp(-(1 - albedo) extinction thickness).
double TintTransmittance(uint32_t c)
{
    return std::exp(-(1.0 - kTintAlbedo[c]) * kTintExtinction * kTintThickness);
}

// The sun NEE options of these tests (the sun straight up, 2 degrees, radiance 1000, black sky), with
// 4 transparent bounces and the shadow transmission flag.
CaelixRenderOptions TransmissionOptions(uint32_t mode, uint32_t bounces, uint32_t seed, bool nee,
                                        bool transmission)
{
    CaelixRenderOptions options = SunOptions(mode, bounces, seed, nee);
    options.bounce_count_transparent = 4;
    if (transmission)
    {
        options.flags |= CAELIX_RENDER_FLAG_SHADOW_TRANSMISSION;
    }
    return options;
}

// Reference mode over `frames` frames; the centre 64 x 64 mean of every channel afterwards.
std::array<double, 3> TransmissionReference(Rig& rig, CaelixRenderer* renderer, TargetSet& set,
                                            const CaelixCamera& camera, uint32_t bounces, bool nee,
                                            bool transmission, uint32_t frames, uint32_t seed0)
{
    for (uint32_t i = 0; i < frames; ++i)
    {
        CaelixRenderOptions options = TransmissionOptions(CAELIX_MODE_REFERENCE, bounces, seed0 + i, nee, transmission);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(rig, renderer, set, camera, options, i + 1 == frames);
    }
    const Target& color = *set.t[kColor];
    return {CentreMean(color, 64, 0), CentreMean(color, 64, 1), CentreMean(color, 64, 2)};
}

// A white floor of 64 x 64 blocks at y in [-8, 0) under air bricks up to y = 15 (the air cells next to
// the slabs exist and report the glass exits).
World ShadowFloorWorld()
{
    World world;
    FillBox(world, 0, -8, 0, 63, -1, 63, kWhiteFloor);
    FillBox(world, 0, 0, 0, 63, 15, 63, 0);
    return world;
}

struct ShadowRig
{
    Rig rig;
    CaelixRenderer* renderer = nullptr;
    Target color;
    TargetSet set;
};

void OpenShadowRig(ShadowRig& s)
{
    OpenRig(s.rig, "caelix_trace_test", RigDesc(0, 0, 0));
    REQUIRE(caelix_renderer_create(s.rig.device, s.rig.scene, &s.renderer) == CAELIX_OK);
    s.color = CreateTarget(s.rig.gpu, VK_FORMAT_R32G32B32A32_SFLOAT);
    s.set.t[kColor] = &s.color;
}

void CloseShadowRig(ShadowRig& s)
{
    REQUIRE(vkDeviceWaitIdle(s.rig.gpu.device.device) == VK_SUCCESS);
    caelix_renderer_release_target_views(s.renderer);
    caelix_renderer_destroy(s.renderer);
    DestroyTarget(s.rig.gpu, s.color);
    CloseRig(s.rig);
}

}  // namespace

TEST_CASE("shadow transmission: tinted glass slab")
{
    ShadowRig s;
    OpenShadowRig(s);
    const CaelixMaterialEntry glass = TintGlassEntry();
    REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &glass, 1) == CAELIX_OK);
    {
        // A smooth tinted slab, 4 voxels thick (y in [8, 12)), over the x < 32 half of the floor, and a
        // black wall under its edge (x = 32, y in [0, 8)): a bounce ray from under the slab that the glass
        // reflects cannot reach the lit floor beside it (whose NEE would add the sun there).
        World world = ShadowFloorWorld();
        FillBox(world, 0, 8, 0, 31, 11, 63, kTintGlass);
        FillBox(world, 32, 0, 0, 32, 7, 63, kBlackWall);
        Upload(s.rig, world, KeysOf(world));
        DrainBuilds(s.rig);
    }
    // Below the slab (camera at height 4): the floor under it, and the floor beside it.
    const CaelixCamera under = LookDownFrom(16.0f, 4.0f, 32.0f);
    const CaelixCamera beside = LookDownFrom(48.0f, 4.0f, 32.0f);

    // 1. Flag on, sun NEE on, 1 bounce: the NEE sees the sun through the slab, times its transmittance;
    //    the bounce rays that hit the slab from below cannot add the sun (the claim; 1 bounce ends there).
    const std::array<double, 3> lit = TransmissionReference(s.rig, s.renderer, s.set, beside, 1, true, true, 256, 1000);
    const std::array<double, 3> on = TransmissionReference(s.rig, s.renderer, s.set, under, 1, true, true, 256, 2000);
    for (uint32_t c = 0; c < 3; ++c)
    {
        const double ratio = on[c] / lit[c];
        std::printf("shadow transmission, tinted slab, flag on, channel %u: under %.5f, beside %.5f, ratio %.5f, "
                    "expected exp(-(1 - albedo) 0.2 4) = %.5f (ratio %.5f)\n",
                    c, on[c], lit[c], ratio, TintTransmittance(c), ratio / TintTransmittance(c));
        CHECK(lit[c] > 0.1);
        CHECK(std::abs(ratio / TintTransmittance(c) - 1.0) < 0.05);
    }

    // 2. Flag off: the NEE shadow ray stops at the glass; the sun under the slab comes only from the
    //    refracted BSDF path (floor -> slab bottom -> slab top -> the sun disk: 3 bounces, probability
    //    sin^2 r per floor bounce, so 4096 frames). For a flat slab and a vertical sun that path carries
    //    the same transmittance times the two Fresnel transmissions (1 - F)^2, F = ((1.5 - 1) / 2.5)^2 at
    //    normal incidence (Schlick in the shader).
    const double fresnel = std::pow((1.5 - 1.0) / (1.5 + 1.0), 2.0);
    const std::array<double, 3> off = TransmissionReference(s.rig, s.renderer, s.set, under, 3, true, false, 4096, 10000);
    for (uint32_t c = 0; c < 3; ++c)
    {
        const double ratio = off[c] / lit[c];
        const double expected = TintTransmittance(c) * (1.0 - fresnel) * (1.0 - fresnel);
        std::printf("shadow transmission, tinted slab, flag off (refracted BSDF path only), channel %u: under "
                    "%.5f, ratio %.5f, expected T (1 - F)^2 = %.5f (ratio %.5f; T alone %.5f)\n",
                    c, off[c], ratio, expected, ratio / expected, ratio / TintTransmittance(c));
        CHECK(std::abs(ratio / expected - 1.0) < 0.05);
    }

    // 3. Flag on with 3 bounces: the claim keeps the refracted path from adding the sun a second time.
    const std::array<double, 3> on3 = TransmissionReference(s.rig, s.renderer, s.set, under, 3, true, true, 256, 3000);
    for (uint32_t c = 0; c < 3; ++c)
    {
        std::printf("shadow transmission, tinted slab, flag on, 3 bounces, channel %u: ratio %.5f (1 bounce "
                    "%.5f)\n",
                    c, on3[c] / lit[c], on[c] / lit[c]);
        CHECK(std::abs(on3[c] / lit[c] / TintTransmittance(c) - 1.0) < 0.05);
    }

    // 4. Flag on, sun NEE off: nothing changes (no NEE, no claim): the same image as the flag off, bit for
    //    bit, in the path trace mode over a few frames.
    for (uint32_t frame = 0; frame < 4; ++frame)
    {
        Render(s.rig, s.renderer, s.set, under, TransmissionOptions(CAELIX_MODE_PATH_TRACE, 3, 500 + frame, false, false));
        const std::vector<uint8_t> plain = s.color.bytes;
        Render(s.rig, s.renderer, s.set, under, TransmissionOptions(CAELIX_MODE_PATH_TRACE, 3, 500 + frame, false, true));
        CHECK(s.color.bytes == plain);
    }

    // 5. bounce_nee_probability: 0 and (0, 1] are accepted (no kernel reads it yet); others are rejected.
    {
        CaelixRenderOptions options = TransmissionOptions(CAELIX_MODE_PATH_TRACE, 1, 7, true, true);
        options.bounce_nee_probability = 0.5f;
        Render(s.rig, s.renderer, s.set, under, options);
        for (const float bad : {-0.25f, 1.5f, std::numeric_limits<float>::quiet_NaN()})
        {
            ExpectLibraryErrors(1, [&] {
                CaelixFrameDesc desc{};
                desc.command_buffer = 1;
                desc.frame_number = s.rig.frames.next;
                const CaelixRenderTargets targets = ToAbi(s.set);
                CaelixRenderOptions invalid = options;
                invalid.bounce_nee_probability = bad;
                REQUIRE(caelix_renderer_render(s.renderer, &desc, &targets, &under, &invalid) ==
                        CAELIX_ERROR_INVALID_ARGUMENT);
            });
        }
    }
    CloseShadowRig(s);
}

TEST_CASE("shadow transmission: SSS slab occludes")
{
    ShadowRig s;
    OpenShadowRig(s);
    CaelixMaterialEntry entries[2] = {SssEntry(1.0f), {}};
    entries[1].code = kPlainCode;
    entries[1].material.albedo[0] = entries[1].material.albedo[1] = entries[1].material.albedo[2] = 0.5f;
    entries[1].material.ior = 1.5f;
    REQUIRE(caelix_scene_set_material_entries(s.rig.scene, entries, 2) == CAELIX_OK);
    {
        // From y = 8: an SSS slab 2 voxels thick over x < 32, z < 32; 6 voxels thick over x >= 32, z < 32;
        // a plain opaque block over x < 32, z >= 32; nothing over x >= 32, z >= 32.
        World world = ShadowFloorWorld();
        FillBox(world, 0, 8, 0, 31, 9, 31, kSssCode);
        FillBox(world, 32, 8, 0, 63, 13, 31, kSssCode);
        FillBox(world, 0, 8, 32, 31, 9, 63, kPlainCode);
        Upload(s.rig, world, KeysOf(world));
        DrainBuilds(s.rig);
    }
    const CaelixCamera thin = LookDownFrom(16.0f, 4.0f, 16.0f);
    const CaelixCamera thick = LookDownFrom(48.0f, 4.0f, 16.0f);
    const CaelixCamera plain = LookDownFrom(16.0f, 4.0f, 48.0f);
    const CaelixCamera open_floor = LookDownFrom(48.0f, 4.0f, 48.0f);

    // Flag on, 1 bounce: the transmissive NEE ray stops at the SSS slab (2 or 6 voxels thick) exactly as
    // at the plain opaque block; only the open floor is lit.
    const std::array<double, 3> lit = TransmissionReference(s.rig, s.renderer, s.set, open_floor, 1, true, true, 64, 4000);
    const std::array<double, 3> thin_t = TransmissionReference(s.rig, s.renderer, s.set, thin, 1, true, true, 64, 5000);
    const float thin_max = CentreMax(s.color, 64);
    const std::array<double, 3> thick_t = TransmissionReference(s.rig, s.renderer, s.set, thick, 1, true, true, 64, 6000);
    const float thick_max = CentreMax(s.color, 64);
    const std::array<double, 3> plain_t = TransmissionReference(s.rig, s.renderer, s.set, plain, 1, true, true, 64, 7000);
    const float plain_max = CentreMax(s.color, 64);
    std::printf("shadow transmission, SSS slab occludes: lit %.5f; 2 voxels %.6f (max %.6f); 6 voxels %.6f (max "
                "%.6f); plain block %.6f (max %.6f)\n",
                lit[0], thin_t[0], thin_max, thick_t[0], thick_max, plain_t[0], plain_max);
    CHECK(lit[0] > 0.1);
    for (uint32_t c = 0; c < 3; ++c)
    {
        CHECK(thin_t[c] == 0.0);
        CHECK(thick_t[c] == 0.0);
        CHECK(plain_t[c] == 0.0);
    }
    CHECK(thin_max == 0.0f);
    CHECK(thick_max == 0.0f);
    CHECK(plain_max == 0.0f);

    // Flag off: the SSS slab occludes like any opaque voxel.
    const std::array<double, 3> off = TransmissionReference(s.rig, s.renderer, s.set, thin, 1, true, false, 16, 9000);
    CHECK(off[0] == 0.0);
    CHECK(CentreMax(s.color, 64) == 0.0f);
    CloseShadowRig(s);
}

TEST_CASE("shadow transmission: hard occluder and lamp candidate")
{
    ShadowRig s;
    OpenShadowRig(s);
    const CaelixLightOptions light{1, 0.0f, 0};  // the light tree with the default threshold
    REQUIRE(caelix_scene_set_light_options(s.rig.scene, &light) == CAELIX_OK);
    const CaelixMaterialEntry glass = TintGlassEntry();
    REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &glass, 1) == CAELIX_OK);
    // A white floor (x, z in [0, 32)) under air bricks; a 2 x 2 lamp 30 units above the floor centre;
    // the box x, z in [8, 24), y in [8, 12) between them holds the tinted glass, a plain opaque block, or
    // air.
    auto upload = [&](uint16_t box) {
        World world;
        FillBox(world, 0, -8, 0, 31, -1, 31, kWhiteFloor);
        FillBox(world, 0, 0, 0, 31, 15, 31, 0);
        FillBox(world, 8, 8, 8, 23, 11, 23, box);
        FillBox(world, 15, 30, 15, 16, 30, 16, kWhiteLamp);
        Upload(s.rig, world, KeysOf(world));
        DrainBuilds(s.rig);
        // The light tree is built at the flushes of the upload; one more flush settles the BVH.
        FlushFrame(s.rig.gpu, s.rig.scene, s.rig.frames);
    };
    SetLightNee(s.renderer, 1, 0);
    const CaelixCamera camera = LookDownFrom(16.0f, 4.0f, 16.0f);
    // Lamps only (no sun radiance), light NEE with MIS, 1 bounce, the flag on.
    auto accumulate = [&](uint32_t frames, uint32_t seed0) {
        for (uint32_t i = 0; i < frames; ++i)
        {
            CaelixRenderOptions options = GlassOptions(CAELIX_MODE_REFERENCE, 1, 4, seed0 + i);
            options.flags = CAELIX_RENDER_FLAG_SHADOW_TRANSMISSION;
            options.reset_accumulation = i == 0 ? 1u : 0u;
            Render(s.rig, s.renderer, s.set, camera, options, i + 1 == frames);
        }
        return std::array<double, 3>{CentreMean(s.color, 64, 0), CentreMean(s.color, 64, 1),
                                     CentreMean(s.color, 64, 2)};
    };

    upload(kTintGlass);
    CaelixLightStats light_stats{};
    caelix_scene_get_light_stats(s.rig.scene, &light_stats);
    REQUIRE(light_stats.light_bricks >= 1);
    const std::array<double, 3> through = accumulate(256, 100);
    REQUIRE(NeeStats(s.renderer).running == 1);
    upload(kGreyFloor);
    const std::array<double, 3> blocked = accumulate(64, 200);
    const float blocked_max = CentreMax(s.color, 64);
    upload(0);
    const std::array<double, 3> open = accumulate(256, 300);
    std::printf("shadow transmission, lamp: open %.5f %.5f %.5f; behind the plain block %.6f (max %.6f)\n", open[0],
                open[1], open[2], blocked[0], blocked_max);
    for (uint32_t c = 0; c < 3; ++c)
    {
        const double ratio = through[c] / open[c];
        std::printf("shadow transmission, lamp behind the tinted slab, channel %u: %.5f, ratio %.5f, expected "
                    "%.5f (ratio %.5f)\n",
                    c, through[c], ratio, TintTransmittance(c), ratio / TintTransmittance(c));
        CHECK(open[c] > 0.01);
        CHECK(std::abs(ratio / TintTransmittance(c) - 1.0) < 0.10);
        CHECK(blocked[c] == 0.0);
    }
    CHECK(blocked_max == 0.0f);
    CloseShadowRig(s);
}

namespace {

// The subsurface walk tests (docs/SUBSURFACE.md section 5): a white procedural sky (all three colours 1,
// no sun radiance), so a vertex that sees only the sky receives exactly 1 from every direction.

// An opaque SSS entry of code `code` (albedo 0.5) with the given walk parameters.
CaelixMaterialEntry WalkEntry(uint16_t code, uint32_t model, float probability, float sss_albedo, float anisotropy,
                              uint32_t steps)
{
    CaelixMaterialEntry e{};
    e.code = code;
    e.material.albedo[0] = e.material.albedo[1] = e.material.albedo[2] = 0.5f;
    e.material.ior = 1.5f;
    e.material.sss_model = model;
    e.material.sss_probability = probability;
    e.material.sss_albedo[0] = e.material.sss_albedo[1] = e.material.sss_albedo[2] = sss_albedo;
    e.material.sss_anisotropy = anisotropy;
    e.material.sss_steps = steps;
    return e;
}

// The white-sky options: sky on miss, the subsurface flag when asked, no sun NEE and no sun radiance.
CaelixRenderOptions FurnaceOptions(uint32_t mode, uint32_t bounces, uint32_t seed, bool subsurface)
{
    CaelixRenderOptions options = LightOptions(mode, bounces, seed);
    const float white[3] = {1.0f, 1.0f, 1.0f};
    Set3(options.sky_zenith, white);
    Set3(options.sky_horizon, white);
    Set3(options.sky_ground, white);
    options.flags = CAELIX_RENDER_FLAG_SKY_ON_MISS | (subsurface ? CAELIX_RENDER_FLAG_SUBSURFACE : 0u);
    return options;
}

// Reference mode over `frames` frames with FurnaceOptions; the centre 64 x 64 mean of every channel.
std::array<double, 3> FurnaceReference(ShadowRig& s, const CaelixCamera& camera, uint32_t bounces, bool subsurface,
                                       uint32_t frames, uint32_t seed0)
{
    for (uint32_t i = 0; i < frames; ++i)
    {
        CaelixRenderOptions options = FurnaceOptions(CAELIX_MODE_REFERENCE, bounces, seed0 + i, subsurface);
        options.reset_accumulation = i == 0 ? 1u : 0u;
        Render(s.rig, s.renderer, s.set, camera, options, i + 1 == frames);
    }
    return {CentreMean(s.color, 64, 0), CentreMean(s.color, 64, 1), CentreMean(s.color, 64, 2)};
}

void UploadWorld(Rig& rig, const World& world)
{
    Upload(rig, world, KeysOf(world));
    DrainBuilds(rig);
}

}  // namespace

TEST_CASE("subsurface: furnace")
{
    ShadowRig s;
    OpenShadowRig(s);
    {
        // A 4^3 cube of the SSS code floating in the white sky (x, y, z in [0, 4)).
        World world;
        FillBox(world, 0, 0, 0, 3, 3, 3, kSssCode);
        UploadWorld(s.rig, world);
    }
    // 6 units above the top face: the top face fills the whole image.
    const CaelixCamera camera = LookDownFrom(2.0f, 10.0f, 2.0f);

    // sss_albedo 1, probability 1, g 0: every primary vertex walks; an exit is a diffuse vertex on the
    // cube's surface that sees only the sky (the cube is convex), so the pixel is the sky radiance 1 times
    // the fraction of walks that leave before the step cap. The guide albedo is lerp(0.5, 1, 1) = 1.
    // The dead fraction of a 4^3 cube at 64 steps is about 0.81^64 = 1e-6 (the survival of a lattice walk
    // in 4 cells per axis), far below the 2 % of the spec.
    const CaelixMaterialEntry walk = WalkEntry(kSssCode, CAELIX_SSS_MODEL_HG, 1.0f, 1.0f, 0.0f, 64u);
    REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &walk, 1) == CAELIX_OK);
    const std::array<double, 3> sss = FurnaceReference(s, camera, 4, true, 512, 100);
    std::printf("subsurface furnace, 4^3 cube, steps 64: mean %.6f %.6f %.6f (dead fraction %.2e)\n", sss[0], sss[1],
                sss[2], 1.0 - sss[0]);
    for (uint32_t c = 0; c < 3; ++c)
    {
        CHECK(sss[c] >= 0.95);
        CHECK(sss[c] <= 1.0 + 1e-4);
    }

    // A short step cap: the dead fraction becomes visible (the cap works).
    const CaelixMaterialEntry short_walk = WalkEntry(kSssCode, CAELIX_SSS_MODEL_HG, 1.0f, 1.0f, 0.0f, 2u);
    REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &short_walk, 1) == CAELIX_OK);
    const std::array<double, 3> capped = FurnaceReference(s, camera, 4, true, 256, 200);
    std::printf("subsurface furnace, steps 2: mean %.5f (dead fraction %.4f)\n", capped[0], 1.0 - capped[0]);
    CHECK(capped[0] < sss[0] - 0.05);
    CHECK(capped[0] > 0.1);

    // The flag off (the subsurface kernel, no walk): the plain opaque surface, albedo 0.5 under a sky of 1.
    REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &walk, 1) == CAELIX_OK);
    const std::array<double, 3> flag_off = FurnaceReference(s, camera, 4, false, 64, 300);
    // sss_model 0 (the gate; no SSS entry, so the main kernel): the same plain surface.
    const CaelixMaterialEntry plain = WalkEntry(kSssCode, CAELIX_SSS_MODEL_NONE, 1.0f, 1.0f, 0.0f, 64u);
    REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &plain, 1) == CAELIX_OK);
    const std::array<double, 3> model_none = FurnaceReference(s, camera, 4, true, 64, 300);
    std::printf("subsurface furnace, flag off: %.5f; sss_model 0: %.5f (expected 0.5)\n", flag_off[0], model_none[0]);
    for (uint32_t c = 0; c < 3; ++c)
    {
        CHECK(std::abs(flag_off[c] - 0.5) < 0.01);
        CHECK(std::abs(model_none[c] - 0.5) < 0.01);
    }
    CloseShadowRig(s);
}

TEST_CASE("subsurface: lifecycle")
{
    ShadowRig s;
    OpenShadowRig(s);
    CaelixMaterialEntry plain_block = WalkEntry(kPlainCode, CAELIX_SSS_MODEL_NONE, 0.0f, 0.0f, 0.0f, 0u);
    REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &plain_block, 1) == CAELIX_OK);

    // 1. A scene without SSS materials: a white floor and a plain block. The flag changes nothing (the
    //    renderer binds the main kernel and drops the flag): the same image bit for bit, same seed.
    {
        World world;
        FillBox(world, 0, -8, 0, 63, -1, 63, kWhiteFloor);
        FillBox(world, 24, 0, 24, 39, 15, 39, kPlainCode);
        UploadWorld(s.rig, world);
    }
    const CaelixCamera beside = LookDownFrom(48.0f, 6.0f, 32.0f);
    std::vector<std::vector<uint8_t>> main_images;
    for (uint32_t frame = 0; frame < 3; ++frame)
    {
        Render(s.rig, s.renderer, s.set, beside, FurnaceOptions(CAELIX_MODE_PATH_TRACE, 3, 40 + frame, false));
        const std::vector<uint8_t> off = s.color.bytes;
        Render(s.rig, s.renderer, s.set, beside, FurnaceOptions(CAELIX_MODE_PATH_TRACE, 3, 40 + frame, true));
        CHECK(s.color.bytes == off);
        main_images.push_back(off);
    }
    // The same scene with an SSS entry on a code that no voxel uses: the subsurface kernel draws the same
    // image as the main kernel (no SSS vertex draws a number; the flag on and off).
    {
        const CaelixMaterialEntry unused = WalkEntry(kSssCode, CAELIX_SSS_MODEL_HG, 1.0f, 1.0f, 0.0f, 8u);
        REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &unused, 1) == CAELIX_OK);
        size_t differing = 0;
        for (uint32_t frame = 0; frame < 3; ++frame)
        {
            for (const bool flag : {false, true})
            {
                Render(s.rig, s.renderer, s.set, beside, FurnaceOptions(CAELIX_MODE_PATH_TRACE, 3, 40 + frame, flag));
                differing += s.color.bytes == main_images[frame] ? 0u : 1u;
            }
        }
        std::printf("subsurface lifecycle: subsurface kernel vs main kernel without SSS voxels: %zu of 6 images "
                    "differ\n",
                    differing);
        CHECK(differing == 0);
    }

    // 2. The walk at a bounce vertex: the block becomes SSS; the camera sees only the floor beside it, so
    //    every change of the image comes from walks at bounce vertices. Finite, and not the flag-off image.
    {
        CaelixMaterialEntry sss_block = WalkEntry(kPlainCode, CAELIX_SSS_MODEL_HG, 0.8f, 0.9f, 0.3f, 32u);
        REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &sss_block, 1) == CAELIX_OK);
        const std::array<double, 3> off = FurnaceReference(s, beside, 3, false, 64, 500);
        const std::vector<uint8_t> off_image = s.color.bytes;
        const std::array<double, 3> on = FurnaceReference(s, beside, 3, true, 64, 500);
        std::printf("subsurface lifecycle, floor beside an SSS block: flag off %.5f, flag on %.5f\n", off[0], on[0]);
        CHECK(Finite(s.color));
        CHECK(s.color.bytes != off_image);
        CHECK(on[0] > 0.1);
    }

    // 3. Dead walks: steps 1, g 0.9 (forward, into the slab), probability 1, sss_albedo 1 in a slab 16
    //    voxels thick. A walk survives only when its single step turns back out of the top face, so the lit
    //    top face stays close to black (no emission), against 0.5 for the plain surface.
    {
        World world;
        FillBox(world, 0, 0, 0, 31, 15, 31, kSssCode);
        UploadWorld(s.rig, world);
        const CaelixMaterialEntry dead = WalkEntry(kSssCode, CAELIX_SSS_MODEL_HG, 1.0f, 1.0f, 0.9f, 1u);
        REQUIRE(caelix_scene_set_material_entries(s.rig.scene, &dead, 1) == CAELIX_OK);
        const CaelixCamera top = LookDownFrom(16.0f, 24.0f, 16.0f);
        const std::array<double, 3> slab = FurnaceReference(s, top, 4, true, 128, 700);
        std::printf("subsurface lifecycle, dead walks (steps 1, g 0.9): top face %.5f %.5f %.5f\n", slab[0], slab[1],
                    slab[2]);
        CHECK(Finite(s.color));
        for (uint32_t c = 0; c < 3; ++c)
        {
            CHECK(slab[c] < 0.05);
        }
    }
    CloseShadowRig(s);
}
