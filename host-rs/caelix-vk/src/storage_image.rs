//! Storage images for the caller (for example the renderer's voxel id target): a 2D image with
//! `STORAGE` usage in device-local memory, created in `UNDEFINED` layout and never filled by this
//! crate. It lives until `destroy_storage_image` or until the context is dropped.

use ash::vk;
use ash::vk::Handle;

use crate::cubemap::OwnedImage;
use crate::{vk_error, Context, Error};

/// `VK_FORMAT_R32G32_UINT` as a raw `VkFormat` (the voxel id target's two-word format).
pub const FORMAT_R32G32_UINT: u32 = vk::Format::R32G32_UINT.as_raw() as u32;

/// A storage image on the device: the values of a `CaelixTargetImage` (layout `UNDEFINED`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageImage {
    /// `VkImage`, 2D, one mip level, one layer.
    pub image: u64,
    /// `VkFormat`.
    pub format: u32,
    pub width: u32,
    pub height: u32,
}

impl Context {
    /// Creates a 2D image (`format`, a raw `VkFormat`; optimal tiling, `STORAGE` usage, device-local
    /// memory) in `UNDEFINED` layout. Its contents are undefined until a render call writes it.
    pub fn create_storage_image(&mut self, width: u32, height: u32, format: u32) -> Result<StorageImage, Error> {
        let format = vk::Format::from_raw(format as i32);
        if width == 0 || height == 0 {
            return Err(Error::InvalidArgument(format!("storage image: size {width}x{height}")));
        }
        // SAFETY: a query on a live physical device.
        let format_properties =
            unsafe { self.instance.get_physical_device_format_properties(self.physical_device, format) };
        if !format_properties.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
            return Err(Error::NoSuitableDevice(format!("{format:?} cannot be a storage image")));
        }
        let device = &self.device;
        // SAFETY (whole block): object creation on the live device; on an error every object
        // created so far is destroyed before returning.
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::STORAGE)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(vk_error("vkCreateImage"))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        // SAFETY: a query on a live physical device.
        let memory_properties =
            unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) };
        let memory_type = (0..memory_properties.memory_type_count).find(|&i| {
            requirements.memory_type_bits & (1 << i) != 0
                && memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        });
        let Some(memory_type) = memory_type else {
            unsafe { device.destroy_image(image, None) };
            return Err(Error::Vulkan { call: "storage image", result: "no device-local memory type".into() });
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
                unsafe { device.destroy_image(image, None) };
                return Err(vk_error("vkAllocateMemory")(e));
            }
        };
        if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(vk_error("vkBindImageMemory")(e));
        }
        self.owned_images.push(OwnedImage { image, memory });
        Ok(StorageImage { image: image.as_raw(), format: format.as_raw() as u32, width, height })
    }

    /// Waits for the device and destroys an image of `create_storage_image` (for example after a
    /// window resize). The caller first drops every view of it (`caelix_renderer_release_target_views`
    /// after a wait for the device). An unknown image is ignored.
    pub fn destroy_storage_image(&mut self, image: &StorageImage) {
        let Some(index) = self.owned_images.iter().position(|owned| owned.image.as_raw() == image.image) else {
            return;
        };
        self.wait_idle();
        let owned = self.owned_images.swap_remove(index);
        owned.destroy(&self.device);
    }
}
