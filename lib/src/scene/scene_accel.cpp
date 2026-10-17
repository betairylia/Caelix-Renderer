// Acceleration structures: BLAS per render group, TLAS per scene.
#include "scene/scene.h"

#include "core/log.h"
#include "scene/scene_internal.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <exception>

namespace caelix {
namespace {

// Build infos per vkCmdBuildAccelerationStructuresKHR call.
constexpr size_t kMaxBuildsPerCall = 256;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

VkAccelerationStructureGeometryKHR AabbGeometry(VkDeviceAddress data)
{
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geometry.geometry.aabbs.data.deviceAddress = data;
    geometry.geometry.aabbs.stride = kAabbRowBytes;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    return geometry;
}

VkAccelerationStructureGeometryKHR InstanceGeometry(VkDeviceAddress data)
{
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = data;
    return geometry;
}

// Mode BUILD, PREFER_FAST_TRACE only. Refit (ALLOW_UPDATE) and compaction
// (ALLOW_COMPACTION) come later.
VkAccelerationStructureBuildGeometryInfoKHR BuildInfo(VkAccelerationStructureTypeKHR type,
                                                      const VkAccelerationStructureGeometryKHR* geometry)
{
    VkAccelerationStructureBuildGeometryInfoKHR info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type = type;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount = 1;
    info.pGeometries = geometry;
    return info;
}

}  // namespace

// ---------------------------------------------------------------------------
// Acceleration structures

VkAccelerationStructureBuildSizesInfoKHR Scene::BlasSizes(uint32_t primitive_count)
{
    // The sizes of an AABB BLAS depend only on the primitive count (same type, flags
    // and geometry count), so they are cached.
    const auto found = blas_sizes_.find(primitive_count);
    if (found != blas_sizes_.end())
    {
        return found->second;
    }
    const VkAccelerationStructureGeometryKHR geometry = AabbGeometry(0);
    const VkAccelerationStructureBuildGeometryInfoKHR info =
        BuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &geometry);
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device_.device(),
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info,
                                            &primitive_count, &sizes);
    blas_sizes_.emplace(primitive_count, sizes);
    return sizes;
}

VkAccelerationStructureBuildSizesInfoKHR Scene::TlasSizes(uint32_t instance_count) const
{
    const VkAccelerationStructureGeometryKHR geometry = InstanceGeometry(0);
    const VkAccelerationStructureBuildGeometryInfoKHR info =
        BuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &geometry);
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device_.device(),
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info,
                                            &instance_count, &sizes);
    return sizes;
}

bool Scene::PlanAccelerationStructures(uint64_t frame, bool& out_of_memory)
{
    out_of_memory = false;
    blas_geometries_.clear();
    blas_infos_.clear();
    blas_ranges_.clear();
    tlas_planned_ = false;

    // 1. Candidates: groups whose BLAS is out of date and whose AABB rows are final
    //    after this flush (no unstaged rows, no finalize entries left for later).
    //    The pending list keeps its order, so builds are first come, first served.
    std::vector<uint32_t> candidates;
    {
        size_t kept = 0;
        for (const uint32_t group_index : blas_pending_)
        {
            Group& group = groups_[group_index];
            if (!group.alive || !group.blas_dirty)
            {
                group.in_blas_list = false;
                continue;
            }
            blas_pending_[kept++] = group_index;
            if (!group.in_dirty_list && group.finalize_wait_serial != flush_serial_)
            {
                candidates.push_back(group_index);
            }
        }
        blas_pending_.resize(kept);
    }
    // Every BLAS build changes what the TLAS refers to, so a BLAS is only built in a
    // flush that also builds the TLAS.
    if (candidates.empty() && !tlas_dirty_)
    {
        return false;
    }

    // 2. Sizes. The TLAS is sized for every live group, an upper bound of the
    //    instances after this flush's BLAS builds.
    const uint32_t instance_bound = std::max(1u, groups_alive_);
    const VkAccelerationStructureBuildSizesInfoKHR tlas_sizes = TlasSizes(instance_bound);
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizes(candidates.size());
    VkDeviceSize max_blas_scratch = 0;
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        sizes[i] = BlasSizes(static_cast<uint32_t>(groups_[candidates[i]].slots.size()));
        max_blas_scratch = std::max(max_blas_scratch, sizes[i].buildScratchSize);
    }
    const VkDeviceSize align = std::max<VkDeviceSize>(
        1, device_.as_properties().minAccelerationStructureScratchOffsetAlignment);

    // 3. Allocations, before anything changes: a failure leaves the scene as it was.
    struct NewAs
    {
        AsBlock block;
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
    };
    NewAs new_tlas;
    std::vector<NewAs> new_blas;  // parallel to the selected candidates; no handle = in place
    std::vector<VkDeviceAddress> blas_scratch;
    VkDeviceAddress tlas_scratch = 0;
    StagingRing::Span span;
    const VkDevice device = device_.device();
    try
    {
        // The TLAS and the largest single BLAS must fit together, else the scratch
        // buffer grows (its contents need no copy; the old buffer is kept until this
        // frame is safe, as earlier frames may still use it).
        const VkDeviceSize required = AlignUp(tlas_sizes.buildScratchSize, align) +
                                      AlignUp(max_blas_scratch, align) + align;
        if (required > scratch_.size())
        {
            GpuBuffer grown = GpuBuffer::DeviceLocalWithUsage(
                device_.allocator(), device, std::bit_ceil(required),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            CAELIX_LOGI("acceleration structure scratch grows from %llu to %llu bytes",
                        static_cast<unsigned long long>(scratch_.size()),
                        static_cast<unsigned long long>(grown.size()));
            deferred_buffers_.push_back(DeferredBuffer{frame, std::move(scratch_)});
            scratch_ = std::move(grown);
        }
        instances_.Reserve(static_cast<VkDeviceSize>(instance_bound) * kInstanceBytes);
        if (!staging_.TryAllocate(static_cast<uint64_t>(instance_bound) * kInstanceBytes, 16,
                                  StagingRing::Purpose::kRows, span))
        {
            CAELIX_LOGD("caelix_scene_flush: the staging ring has no room for %u TLAS instances; the "
                        "acceleration structure builds wait for the next flush",
                        instance_bound);
            return false;
        }

        // Scratch: the TLAS first, then BLASes in list order until one does not fit.
        const VkDeviceAddress base = scratch_.address();
        VkDeviceSize cursor = 0;
        const auto take = [&](VkDeviceSize bytes, VkDeviceAddress& out) {
            const VkDeviceAddress aligned = AlignUp(base + cursor, align);
            const VkDeviceSize end = aligned - base + bytes;
            if (end > scratch_.size())
            {
                return false;
            }
            out = aligned;
            cursor = end;
            return true;
        };
        if (!take(tlas_sizes.buildScratchSize, tlas_scratch))
        {
            return false;  // cannot happen: the buffer holds `required`
        }
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            VkDeviceAddress address = 0;
            if (!take(sizes[i].buildScratchSize, address))
            {
                break;
            }
            blas_scratch.push_back(address);
        }

        // Storage. A structure is rebuilt in place when its block is large enough.
        if (tlas_handle_ == VK_NULL_HANDLE || tlas_.size < tlas_sizes.accelerationStructureSize)
        {
            new_tlas.block = as_pool_.Allocate(tlas_sizes.accelerationStructureSize, AsKind::kTlas);
            new_tlas.handle = CreateAccelerationStructure(device, as_pool_.buffer(new_tlas.block),
                                                          new_tlas.block,
                                                          VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
            new_tlas.address = AccelerationStructureAddress(device, new_tlas.handle);
        }
        new_blas.reserve(blas_scratch.size());
        for (size_t i = 0; i < blas_scratch.size(); ++i)
        {
            const Group& group = groups_[candidates[i]];
            new_blas.emplace_back();
            if (group.blas_handle != VK_NULL_HANDLE &&
                group.blas.size >= sizes[i].accelerationStructureSize)
            {
                continue;
            }
            NewAs& created = new_blas.back();
            created.block = as_pool_.Allocate(sizes[i].accelerationStructureSize, AsKind::kBlas);
            created.handle = CreateAccelerationStructure(device, as_pool_.buffer(created.block),
                                                         created.block,
                                                         VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
            created.address = AccelerationStructureAddress(device, created.handle);
        }
    }
    catch (const std::exception& error)
    {
        // Nothing recorded uses the new structures: release them at once.
        new_blas.push_back(new_tlas);
        for (const NewAs& created : new_blas)
        {
            if (created.handle != VK_NULL_HANDLE)
            {
                vkDestroyAccelerationStructureKHR(device, created.handle, nullptr);
            }
            as_pool_.Free(created.block);
        }
        CAELIX_LOGE("caelix_scene_flush: cannot allocate acceleration structures (%s); the builds "
                    "wait",
                    error.what());
        out_of_memory = true;
        return false;
    }

    // 4. Commit: the BLAS builds.
    const VkDeviceAddress aabb_base = aabbs_.address();
    blas_geometries_.reserve(new_blas.size());  // the infos point into it
    for (size_t i = 0; i < new_blas.size(); ++i)
    {
        Group& group = groups_[candidates[i]];
        const NewAs& created = new_blas[i];
        if (created.handle != VK_NULL_HANDLE)
        {
            if (group.blas_handle != VK_NULL_HANDLE)
            {
                // The old BLAS is used until this frame's TLAS replaces it.
                as_pool_.FreeAfter(group.blas, frame, group.blas_handle);
            }
            else
            {
                ++blas_count_;
            }
            group.blas = created.block;
            group.blas_handle = created.handle;
            group.blas_address = created.address;
        }
        group.blas_primitive_count = static_cast<uint32_t>(group.slots.size());
        group.blas_dirty = false;
        blas_geometries_.push_back(
            AabbGeometry(aabb_base + aabbs_.ByteOffset(group.aabb_rows.offset)));
        VkAccelerationStructureBuildGeometryInfoKHR info =
            BuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &blas_geometries_.back());
        info.dstAccelerationStructure = group.blas_handle;
        info.scratchData.deviceAddress = blas_scratch[i];
        blas_infos_.push_back(info);
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = group.blas_primitive_count;
        blas_ranges_.push_back(range);
    }

    // 5. The TLAS: one instance per live group with a BLAS.
    if (new_tlas.handle != VK_NULL_HANDLE)
    {
        if (tlas_handle_ != VK_NULL_HANDLE)
        {
            as_pool_.FreeAfter(tlas_, frame, tlas_handle_);
        }
        tlas_ = new_tlas.block;
        tlas_handle_ = new_tlas.handle;
        tlas_address_ = new_tlas.address;
    }
    uint32_t instance_count = 0;
    for (uint32_t group_index = 0; group_index < groups_.size(); ++group_index)
    {
        const Group& group = groups_[group_index];
        if (!group.alive || group.blas_handle == VK_NULL_HANDLE)
        {
            continue;
        }
        const Entity& entity = entities_[group.entity];
        VkAccelerationStructureInstanceKHR instance{};
        std::memcpy(instance.transform.matrix, entity.cur_rows, sizeof(instance.transform.matrix));
        instance.instanceCustomIndex = group_index;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        instance.accelerationStructureReference = group.blas_address;
        std::memcpy(span.data + static_cast<VkDeviceSize>(instance_count) * kInstanceBytes, &instance,
                    sizeof(instance));
        ++instance_count;
    }
    if (instance_count > 0)
    {
        pending_.push_back(PendingCopy{span.offset, instances_.buffer(), 0,
                                       static_cast<VkDeviceSize>(instance_count) * kInstanceBytes});
    }
    tlas_geometry_ = InstanceGeometry(instances_.address());
    tlas_info_ = BuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &tlas_geometry_);
    tlas_info_.dstAccelerationStructure = tlas_handle_;
    tlas_info_.scratchData.deviceAddress = tlas_scratch;
    tlas_range_ = VkAccelerationStructureBuildRangeInfoKHR{};
    tlas_range_.primitiveCount = instance_count;
    tlas_planned_ = true;
    tlas_dirty_ = false;
    // The new TLAS no longer refers to the BLASes of freed groups.
    for (const RetiredBlas& retired : retired_blas_)
    {
        as_pool_.FreeAfter(retired.block, frame, retired.handle);
    }
    retired_blas_.clear();
    as_builds_total_ += blas_infos_.size() + 1;
    return true;
}

void Scene::RecordAccelerationStructureBuilds(VkCommandBuffer cmd)
{
    if (!blas_infos_.empty())
    {
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> ranges(blas_ranges_.size());
        for (size_t i = 0; i < ranges.size(); ++i)
        {
            ranges[i] = &blas_ranges_[i];
        }
        for (size_t i = 0; i < blas_infos_.size(); i += kMaxBuildsPerCall)
        {
            const size_t n = std::min(kMaxBuildsPerCall, blas_infos_.size() - i);
            vkCmdBuildAccelerationStructuresKHR(cmd, static_cast<uint32_t>(n), blas_infos_.data() + i,
                                                ranges.data() + i);
        }
        // The TLAS build reads the BLASes.
        MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                      VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                      VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                      VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }
    if (tlas_planned_)
    {
        const VkAccelerationStructureBuildRangeInfoKHR* range = &tlas_range_;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_info_, &range);
        // Later commands (the ray query kernels) traverse the structures.
        MemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                      VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    }
}

void Scene::DestroyAccelerationStructures()
{
    const VkDevice device = device_.device();
    for (Group& group : groups_)
    {
        if (group.blas_handle != VK_NULL_HANDLE)
        {
            vkDestroyAccelerationStructureKHR(device, group.blas_handle, nullptr);
            group.blas_handle = VK_NULL_HANDLE;
        }
    }
    for (const RetiredBlas& retired : retired_blas_)
    {
        vkDestroyAccelerationStructureKHR(device, retired.handle, nullptr);
    }
    retired_blas_.clear();
    if (tlas_handle_ != VK_NULL_HANDLE)
    {
        vkDestroyAccelerationStructureKHR(device, tlas_handle_, nullptr);
        tlas_handle_ = VK_NULL_HANDLE;
    }
    // as_pool_ destroys the handles of its pending frees.
}

}  // namespace caelix
