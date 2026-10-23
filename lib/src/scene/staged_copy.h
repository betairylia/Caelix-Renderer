#pragma once
#include "core/vk_common.h"

namespace caelix {

// One copy from the staging ring into a pool or table, recorded by the flush (Scene::RecordCopies).
struct StagedCopy
{
    VkDeviceSize src_offset;
    VkBuffer dst;
    VkDeviceSize dst_offset;
    VkDeviceSize size;
};

}  // namespace caelix
