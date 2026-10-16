//! Sky cubemap upload: an RGBA32F cube image with mips, filled once through a staging buffer and a
//! one-time command buffer, then kept in `SHADER_READ_ONLY_OPTIMAL` until the context is dropped.

use ash::vk;
use ash::vk::Handle;

use crate::{vk_error, Context, Error};

const FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const TEXEL_BYTES: u64 = 16;

/// A sky cubemap on the device: the values of `CaelixSkyCubemap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkyCubemap {
    /// `VkImage`, 6 layers, cube compatible.
    pub image: u64,
    /// `VkFormat` (`R32G32B32A32_SFLOAT`).
    pub format: u32,
    /// `VkImageLayout` (`SHADER_READ_ONLY_OPTIMAL`).
    pub layout: u32,
    /// Face width = height of mip 0 in texels.
    pub size: u32,
    pub mip_levels: u32,
}

/// An image and its memory, destroyed in `Context::drop`.
pub(crate) struct OwnedImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
}

impl OwnedImage {
    pub(crate) fn destroy(&self, device: &ash::Device) {
        // SAFETY: the caller guarantees the GPU no longer uses the image.
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Every object of one upload. Null handles are skipped by Vulkan's destroy calls, so the whole set
/// can be destroyed at any point of the upload.
#[derive(Default)]
struct Upload {
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    staging: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    pool: vk::CommandPool,
    fence: vk::Fence,
}

impl Upload {
    /// Destroys the staging objects (and the image too when `keep_image` is false).
    fn destroy(&self, device: &ash::Device, keep_image: bool) {
        // SAFETY: the caller guarantees the GPU no longer uses the objects; destroying a null
        // handle is a no-op.
        unsafe {
            device.destroy_fence(self.fence, None);
            // Frees the command buffer too.
            device.destroy_command_pool(self.pool, None);
            device.destroy_buffer(self.staging, None);
            device.free_memory(self.staging_memory, None);
            if !keep_image {
                device.destroy_image(self.image, None);
                device.free_memory(self.image_memory, None);
            }
        }
    }
}

fn invalid(message: String) -> Error {
    Error::InvalidArgument(message)
}

/// The index of a memory type allowed by `type_bits` with every flag of `wanted`.
fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|&i| {
        type_bits & (1 << i) != 0 && properties.memory_types[i as usize].property_flags.contains(wanted)
    })
}

impl Context {
    /// Creates a cube image (`R32G32B32A32_SFLOAT`, 6 layers, `mips.len()` mip levels, sampled) and
    /// fills it: `mips[m]` holds the 6 faces of `(size >> m)^2` texels each, face-major in Vulkan
    /// face order (+X, -X, +Y, -Y, +Z, -Z), row-major inside a face. The upload is submitted on the
    /// graphics queue and waited for; the image is left in `SHADER_READ_ONLY_OPTIMAL` and lives until
    /// the context is dropped.
    pub fn create_sky_cubemap(&mut self, size: u32, mips: &[Vec<[f32; 4]>]) -> Result<SkyCubemap, Error> {
        if size == 0 || mips.is_empty() || mips.len() > (32 - size.leading_zeros()) as usize {
            return Err(invalid(format!("sky cubemap: size {size} with {} mip levels", mips.len())));
        }
        for (level, mip) in mips.iter().enumerate() {
            let side = u64::from((size >> level).max(1));
            if mip.len() as u64 != 6 * side * side {
                return Err(invalid(format!(
                    "sky cubemap: mip {level} has {} texels, expected {}",
                    mip.len(),
                    6 * side * side
                )));
            }
        }
        // SAFETY: a query on a live physical device.
        let format_properties =
            unsafe { self.instance.get_physical_device_format_properties(self.physical_device, FORMAT) };
        let needed = vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST;
        if !format_properties.optimal_tiling_features.contains(needed) {
            return Err(Error::NoSuitableDevice(format!("{FORMAT:?} cannot be sampled and copied to")));
        }

        let mut upload = Upload::default();
        let result = self.upload_cubemap(&mut upload, size, mips);
        match result {
            Ok(()) => {
                // The fence wait completed: nothing uses the staging objects any more.
                upload.destroy(&self.device, true);
                self.owned_images.push(OwnedImage { image: upload.image, memory: upload.image_memory });
                Ok(SkyCubemap {
                    image: upload.image.as_raw(),
                    format: FORMAT.as_raw() as u32,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL.as_raw() as u32,
                    size,
                    mip_levels: mips.len() as u32,
                })
            }
            Err(e) => {
                // A submit may have happened before the error: wait before destroying.
                self.wait_idle();
                upload.destroy(&self.device, false);
                Err(e)
            }
        }
    }

    /// Creates the objects of `upload` one by one (so the caller can destroy them on an error),
    /// records and submits the copy, and waits for it.
    fn upload_cubemap(&self, upload: &mut Upload, size: u32, mips: &[Vec<[f32; 4]>]) -> Result<(), Error> {
        let device = &self.device;
        let mip_levels = mips.len() as u32;
        // SAFETY: a query on a live physical device.
        let memory_properties =
            unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) };

        // SAFETY (whole function): object creation, memory mapping and command recording on live
        // handles owned by `upload`; the caller destroys them after the GPU is done with them.
        upload.image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(FORMAT)
                    .extent(vk::Extent3D { width: size, height: size, depth: 1 })
                    .mip_levels(mip_levels)
                    .array_layers(6)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(vk_error("vkCreateImage"))?;
        let requirements = unsafe { device.get_image_memory_requirements(upload.image) };
        let memory_type = find_memory_type(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| Error::Vulkan { call: "sky cubemap", result: "no device-local memory type".into() })?;
        upload.image_memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .map_err(vk_error("vkAllocateMemory"))?;
        unsafe { device.bind_image_memory(upload.image, upload.image_memory, 0) }
            .map_err(vk_error("vkBindImageMemory"))?;

        // Staging buffer: every mip level, its 6 faces contiguous.
        let total_bytes: u64 = mips.iter().map(|mip| mip.len() as u64 * TEXEL_BYTES).sum();
        upload.staging = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(total_bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(vk_error("vkCreateBuffer"))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(upload.staging) };
        let memory_type = find_memory_type(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| Error::Vulkan { call: "sky cubemap", result: "no host-visible coherent memory type".into() })?;
        upload.staging_memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .map_err(vk_error("vkAllocateMemory"))?;
        unsafe { device.bind_buffer_memory(upload.staging, upload.staging_memory, 0) }
            .map_err(vk_error("vkBindBufferMemory"))?;

        let mut regions = Vec::with_capacity(mips.len());
        // SAFETY: the memory is host-visible and coherent; `total_bytes` bytes are mapped and every
        // write stays inside them (the offsets add up to `total_bytes`).
        unsafe {
            let mapped = device
                .map_memory(upload.staging_memory, 0, total_bytes, vk::MemoryMapFlags::empty())
                .map_err(vk_error("vkMapMemory"))?
                .cast::<u8>();
            let mut offset = 0u64;
            for (level, mip) in mips.iter().enumerate() {
                let bytes = mip.len() * TEXEL_BYTES as usize;
                std::ptr::copy_nonoverlapping(mip.as_ptr().cast::<u8>(), mapped.add(offset as usize), bytes);
                let side = (size >> level).max(1);
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(level as u32)
                                .base_array_layer(0)
                                .layer_count(6),
                        )
                        .image_extent(vk::Extent3D { width: side, height: side, depth: 1 }),
                );
                offset += bytes as u64;
            }
            device.unmap_memory(upload.staging_memory);
        }

        upload.pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(self.queue_family),
                None,
            )
        }
        .map_err(vk_error("vkCreateCommandPool"))?;
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(upload.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(vk_error("vkAllocateCommandBuffers"))?[0];
        upload.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vk_error("vkCreateFence"))?;

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(6);
        let to_transfer = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(upload.image)
            .subresource_range(range);
        let to_shader = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(upload.image)
            .subresource_range(range);
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit = vk::SubmitInfo2::default().command_buffer_infos(&command_buffers);
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(vk_error("vkBeginCommandBuffer"))?;
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_transfer)),
            );
            device.cmd_copy_buffer_to_image(
                cmd,
                upload.staging,
                upload.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_shader)),
            );
            device.end_command_buffer(cmd).map_err(vk_error("vkEndCommandBuffer"))?;
            device
                .queue_submit2(self.queue, &[submit], upload.fence)
                .map_err(vk_error("vkQueueSubmit2"))?;
            device.wait_for_fences(&[upload.fence], true, u64::MAX).map_err(vk_error("vkWaitForFences"))?;
        }
        Ok(())
    }
}
