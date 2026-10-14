//! Swapchain image readback for screenshots.

use ash::vk;

use crate::{vk_error, Error};

/// A host-visible buffer that receives one copy of a swapchain image.
pub(crate) struct PendingReadback {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Frame whose command buffer records the copy.
    pub frame: u64,
    pub width: u32,
    pub height: u32,
    /// The image format stores blue in byte 0 (B8G8R8A8_*): swizzle to RGBA.
    pub bgra: bool,
}

/// Whether `format` has 4 bytes per pixel in RGBA or BGRA order; `Some(true)` for BGRA.
pub(crate) fn rgba8_order(format: vk::Format) -> Option<bool> {
    match format {
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => Some(true),
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Some(false),
        _ => None,
    }
}

impl PendingReadback {
    /// Creates a host-visible, coherent buffer of `width * height * 4` bytes.
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
        bgra: bool,
        frame: u64,
    ) -> Result<Self, Error> {
        let size = u64::from(width) * u64::from(height) * 4;
        // SAFETY (whole function): object creation and queries on live handles. On an error the
        // objects created so far are destroyed before returning.
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(vk_error("vkCreateBuffer"))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = (0..properties.memory_type_count).find(|&i| {
            requirements.memory_type_bits & (1 << i) != 0
                && properties.memory_types[i as usize].property_flags.contains(wanted)
        });
        let Some(memory_type) = memory_type else {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(Error::Vulkan {
                call: "readback",
                result: "no host-visible coherent memory type".into(),
            });
        };
        let memory = match unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        } {
            Ok(memory) => memory,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(vk_error("vkAllocateMemory")(e));
            }
        };
        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(vk_error("vkBindBufferMemory")(e));
        }
        Ok(Self { buffer, memory, frame, width, height, bgra })
    }

    /// Records the copy of `image` (in GENERAL layout) into the buffer, and leaves the image in
    /// PRESENT_SRC_KHR. Replaces the GENERAL -> PRESENT barrier of a normal frame.
    pub(crate) fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image) {
        let to_transfer = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(crate::color_range());
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 });
        let to_present = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(crate::color_range());
        let to_host = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        // SAFETY: the command buffer is recording; the image and the buffer are live.
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_transfer)),
            );
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.buffer,
                std::slice::from_ref(&region),
            );
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&to_present))
                    .buffer_memory_barriers(std::slice::from_ref(&to_host)),
            );
        }
    }

    /// Reads the buffer as RGBA8, top row first. The frame must have completed.
    pub(crate) fn read(&self, device: &ash::Device) -> Result<Vec<u8>, Error> {
        let size = self.width as usize * self.height as usize * 4;
        // SAFETY: the copy's frame has completed (checked by the caller); the memory is
        // host-visible and coherent, and `size` bytes are mapped.
        let mut pixels = unsafe {
            let mapped = device
                .map_memory(self.memory, 0, size as u64, vk::MemoryMapFlags::empty())
                .map_err(vk_error("vkMapMemory"))?;
            let pixels = std::slice::from_raw_parts(mapped.cast::<u8>(), size).to_vec();
            device.unmap_memory(self.memory);
            pixels
        };
        if self.bgra {
            for pixel in pixels.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        Ok(pixels)
    }

    pub(crate) fn destroy(&self, device: &ash::Device) {
        // SAFETY: the caller guarantees the GPU no longer uses the buffer.
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}
