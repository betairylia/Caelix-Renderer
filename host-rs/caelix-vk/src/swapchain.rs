//! Swapchain creation and recreation.

use ash::vk;

use crate::{vk_error, Error};

pub(crate) struct Swapchain {
    loader: ash::khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
}

impl Swapchain {
    pub(crate) fn new(
        instance: &ash::Instance,
        surface_fn: &ash::khr::surface::Instance,
        loader: ash::khr::swapchain::Device,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        let mut swapchain = Self {
            loader,
            handle: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
        };
        swapchain.recreate(instance, surface_fn, surface, physical_device, width, height)?;
        Ok(swapchain)
    }

    pub(crate) fn loader(&self) -> &ash::khr::swapchain::Device {
        &self.loader
    }
    pub(crate) fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }
    pub(crate) fn is_valid(&self) -> bool {
        self.handle != vk::SwapchainKHR::null()
    }
    pub(crate) fn image(&self, index: u32) -> vk::Image {
        self.images[index as usize]
    }
    pub(crate) fn image_count(&self) -> usize {
        self.images.len()
    }
    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }
    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Creates a swapchain for the current surface size, replacing the old one.
    /// The caller guarantees the device is idle. A zero-sized surface (minimized
    /// window) leaves no swapchain; `is_valid` then returns false.
    pub(crate) fn recreate(
        &mut self,
        instance: &ash::Instance,
        surface_fn: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        // SAFETY (whole function): queries and object creation on live handles;
        // the device is idle, so the old swapchain images are not in use.
        let caps = unsafe {
            surface_fn.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .map_err(vk_error("vkGetPhysicalDeviceSurfaceCapabilitiesKHR"))?;

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            self.destroy();
            log::debug!("swapchain: surface size is 0, no swapchain");
            return Ok(());
        }

        let formats = unsafe { surface_fn.get_physical_device_surface_formats(physical_device, surface) }
            .map_err(vk_error("vkGetPhysicalDeviceSurfaceFormatsKHR"))?;
        let surface_format = formats
            .iter()
            .copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first().copied())
            .ok_or_else(|| Error::Vulkan {
                call: "vkGetPhysicalDeviceSurfaceFormatsKHR",
                result: "no surface formats".into(),
            })?;

        // TRANSFER_DST: the renderer clears the image before the scene has a TLAS.
        // TRANSFER_SRC: screenshots (Context::request_readback).
        let transfer = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
        if !caps.supported_usage_flags.contains(transfer) {
            return Err(Error::Vulkan {
                call: "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                result: "surface does not support TRANSFER_DST and TRANSFER_SRC usage".into(),
            });
        }
        let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | transfer;
        // STORAGE: the trace kernel writes the swapchain image.
        let format_properties = unsafe {
            instance.get_physical_device_format_properties(physical_device, surface_format.format)
        };
        if caps.supported_usage_flags.contains(vk::ImageUsageFlags::STORAGE)
            && format_properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
        {
            usage |= vk::ImageUsageFlags::STORAGE;
        } else {
            log::warn!(
                "swapchain format {:?} does not support STORAGE usage; creating it without",
                surface_format.format
            );
        }

        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|&mode| caps.supported_composite_alpha.contains(mode))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);

        let mut min_image_count = caps.min_image_count.max(2);
        if caps.max_image_count > 0 {
            min_image_count = min_image_count.min(caps.max_image_count);
        }

        let old = self.handle;
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old);
        let handle = unsafe { self.loader.create_swapchain(&create_info, None) }
            .map_err(vk_error("vkCreateSwapchainKHR"))?;
        if old != vk::SwapchainKHR::null() {
            unsafe { self.loader.destroy_swapchain(old, None) };
        }
        self.handle = handle;
        self.images = unsafe { self.loader.get_swapchain_images(handle) }
            .map_err(vk_error("vkGetSwapchainImagesKHR"))?;
        self.format = surface_format.format;
        self.extent = extent;
        log::debug!(
            "swapchain created: {}x{}, {} images, {:?} {:?}, usage {:?}",
            extent.width,
            extent.height,
            self.images.len(),
            surface_format.format,
            surface_format.color_space,
            usage
        );
        Ok(())
    }

    /// Destroys the swapchain. The caller guarantees the device is idle.
    pub(crate) fn destroy(&mut self) {
        if self.handle != vk::SwapchainKHR::null() {
            // SAFETY: the device is idle; the images are owned by the swapchain.
            unsafe { self.loader.destroy_swapchain(self.handle, None) };
            self.handle = vk::SwapchainKHR::null();
            self.images.clear();
        }
    }
}
