//! Vulkan device, swapchain and frame loop for the standalone host.
//!
//! This crate is the only Rust crate that uses `ash`. Its public API carries no
//! `ash` types: Vulkan handles leave it as raw `u64` values (and the loader entry
//! point as a raw pointer), so a C library can adopt the device and record into
//! the frame's command buffer.
//!
//! Frame contract:
//! 1. [`Context::begin_frame`] returns a [`Frame`]: a command buffer in the
//!    recording state and the acquired swapchain image in `UNDEFINED` layout.
//! 2. The caller records its work. The work must leave the image in
//!    `VK_IMAGE_LAYOUT_GENERAL`.
//! 3. [`Context::end_frame`] transitions the image to `PRESENT_SRC_KHR`, submits
//!    and presents.

mod debug;
mod device;
mod swapchain;

use std::ffi::{c_void, CString};

use ash::vk;
use ash::vk::Handle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use swapchain::Swapchain;

/// Number of frames the CPU may record ahead of the GPU.
const FRAMES_IN_FLIGHT: usize = 2;

/// What a library needs from the device. The host builds this from the
/// library's own description (for Caelix: `caelix_get_device_requirements`).
pub struct DeviceRequirements {
    /// Minimum Vulkan API version, encoded like `VK_MAKE_API_VERSION`.
    pub api_version: u32,
    /// Device extensions to enable, in addition to `VK_KHR_swapchain`.
    pub device_extensions: Vec<CString>,
    /// Device features to enable.
    pub features: Features,
}

/// Device features to enable. Every `true` field must be supported by the
/// selected physical device.
#[derive(Default, Clone, Copy, Debug)]
pub struct Features {
    pub buffer_device_address: bool,
    /// runtimeDescriptorArray, shaderStorageBufferArrayNonUniformIndexing,
    /// descriptorBindingPartiallyBound, descriptorBindingVariableDescriptorCount.
    pub descriptor_indexing: bool,
    pub synchronization2: bool,
    pub dynamic_rendering: bool,
    pub timeline_semaphore: bool,
    pub acceleration_structure: bool,
    pub ray_query: bool,
    pub shader_int64: bool,
    pub scalar_block_layout: bool,
}

/// Raw handles for a library that adopts this device. All as u64 / pointer, no ash types.
pub struct AdoptHandles {
    /// `PFN_vkGetInstanceProcAddr` of the loader this context uses.
    pub get_instance_proc_addr: *const c_void,
    pub instance: u64,
    pub physical_device: u64,
    pub device: u64,
    /// Graphics + compute capable queue; it also presents.
    pub graphics_queue: u64,
    pub graphics_queue_family: u32,
}

/// One frame to record. Raw Vulkan values; see the crate documentation for the
/// layout contract.
#[derive(Debug)]
pub struct Frame {
    /// `VkCommandBuffer` in the recording state.
    pub command_buffer: u64,
    /// `VkImage` of the acquired swapchain image.
    pub image: u64,
    /// `VkFormat` of the image.
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// `VkImageLayout`: always `VK_IMAGE_LAYOUT_UNDEFINED`; contents are discarded each frame.
    pub layout: u32,
}

/// Errors of this crate. Vulkan results are carried by name, so no `ash` type
/// leaks into the public API.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("window handle unavailable: {0}")]
    WindowHandle(String),
    #[error("no suitable Vulkan device: {0}")]
    NoSuitableDevice(String),
    #[error("{call} failed: {result}")]
    Vulkan { call: &'static str, result: String },
    #[error("end_frame called without a frame from begin_frame")]
    NoFrameInFlight,
}

pub(crate) fn vk_error(call: &'static str) -> impl FnOnce(vk::Result) -> Error {
    move |result| Error::Vulkan { call, result: format!("{result:?}") }
}

/// Per-frame-in-flight objects.
struct FrameSync {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    in_flight: vk::Fence,
    image_available: vk::Semaphore,
}

/// Instance, device, swapchain and frame loop.
pub struct Context {
    entry: ash::Entry,
    instance: ash::Instance,
    debug_messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface_fn: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    swapchain: Swapchain,
    /// Signalled by the submit, waited by the present. One per swapchain image,
    /// because a binary semaphore waited by a present can only be reused once
    /// that image is acquired again. Grow-only, destroyed in Drop.
    render_finished: Vec<vk::Semaphore>,
    frames: Vec<FrameSync>,
    frame_index: usize,
    width: u32,
    height: u32,
    needs_recreate: bool,
    /// Swapchain image index between begin_frame and end_frame.
    acquired: Option<u32>,
}

impl Context {
    /// Creates the instance (with validation when `enable_validation` and the
    /// layer is installed), the surface, the device and the swapchain.
    pub fn new(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        width: u32,
        height: u32,
        requirements: &DeviceRequirements,
        enable_validation: bool,
    ) -> Result<Self, Error> {
        let display_handle = window
            .display_handle()
            .map_err(|e| Error::WindowHandle(e.to_string()))?
            .as_raw();
        let window_handle = window
            .window_handle()
            .map_err(|e| Error::WindowHandle(e.to_string()))?
            .as_raw();

        // Links vulkan-1.lib from the Vulkan SDK (ash feature "linked").
        let entry = ash::Entry::linked();
        let api_version = requirements.api_version.max(vk::API_VERSION_1_3);
        let (instance, debug_messenger) =
            device::create_instance(&entry, display_handle, api_version, enable_validation)?;

        let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
        // SAFETY: the handles come from a live window; the caller keeps the window
        // alive for the lifetime of the Context (the surface is destroyed in Drop).
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }
        .map_err(vk_error("vkCreateWin32SurfaceKHR"))?;

        let selected =
            device::select_physical_device(&instance, &surface_fn, surface, requirements, api_version)?;
        let (device, queue) = device::create_device(&instance, &selected, requirements)?;

        let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
        let swapchain = Swapchain::new(
            &instance,
            &surface_fn,
            swapchain_fn,
            surface,
            selected.physical_device,
            width,
            height,
        )?;

        let mut context = Self {
            entry,
            instance,
            debug_messenger,
            surface_fn,
            surface,
            physical_device: selected.physical_device,
            device,
            queue,
            queue_family: selected.queue_family,
            swapchain,
            render_finished: Vec::new(),
            frames: Vec::new(),
            frame_index: 0,
            width,
            height,
            needs_recreate: false,
            acquired: None,
        };
        context.create_frame_sync()?;
        context.ensure_render_finished_semaphores()?;
        Ok(context)
    }

    /// Raw handles for a library that adopts this device.
    pub fn adopt_handles(&self) -> AdoptHandles {
        AdoptHandles {
            get_instance_proc_addr: self.entry.static_fn().get_instance_proc_addr as *const c_void,
            instance: self.instance.handle().as_raw(),
            physical_device: self.physical_device.as_raw(),
            device: self.device.handle().as_raw(),
            graphics_queue: self.queue.as_raw(),
            graphics_queue_family: self.queue_family,
        }
    }

    /// Marks the swapchain for recreation on the next `begin_frame`.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width != self.width || height != self.height {
            self.width = width;
            self.height = height;
            self.needs_recreate = true;
        }
    }

    /// Waits for this frame slot, acquires a swapchain image and begins the
    /// command buffer. Returns `None` when the swapchain was recreated in this
    /// call or the window size is 0; the caller skips drawing this frame.
    pub fn begin_frame(&mut self) -> Result<Option<Frame>, Error> {
        if self.acquired.is_some() {
            return Err(Error::Vulkan {
                call: "begin_frame",
                result: "previous frame was not ended".into(),
            });
        }
        if self.width == 0 || self.height == 0 {
            return Ok(None);
        }
        let in_flight = self.frames[self.frame_index].in_flight;
        // SAFETY: the fence belongs to this device.
        unsafe { self.device.wait_for_fences(&[in_flight], true, u64::MAX) }
            .map_err(vk_error("vkWaitForFences"))?;

        if self.needs_recreate || !self.swapchain.is_valid() {
            self.recreate_swapchain()?;
            return Ok(None);
        }

        let sync = &self.frames[self.frame_index];
        // SAFETY: the semaphore is unsignalled: its last signal was waited by the
        // submit of this slot, which the fence wait above confirmed complete.
        let acquire = unsafe {
            self.swapchain.loader().acquire_next_image(
                self.swapchain.handle(),
                u64::MAX,
                sync.image_available,
                vk::Fence::null(),
            )
        };
        let image_index = match acquire {
            Ok((index, suboptimal)) => {
                // SUBOPTIMAL still signals the semaphore, so this image must be
                // used; recreate on the next frame instead.
                if suboptimal {
                    self.needs_recreate = true;
                }
                index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Ok(None);
            }
            Err(e) => return Err(vk_error("vkAcquireNextImageKHR")(e)),
        };

        // SAFETY: the fence is signalled (waited above) and no longer in use; the
        // pool's command buffer finished executing for the same reason.
        unsafe {
            self.device
                .reset_fences(&[sync.in_flight])
                .map_err(vk_error("vkResetFences"))?;
            self.device
                .reset_command_pool(sync.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(vk_error("vkResetCommandPool"))?;
            self.device
                .begin_command_buffer(
                    sync.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(vk_error("vkBeginCommandBuffer"))?;
        }
        self.acquired = Some(image_index);

        let extent = self.swapchain.extent();
        Ok(Some(Frame {
            command_buffer: sync.command_buffer.as_raw(),
            image: self.swapchain.image(image_index).as_raw(),
            format: self.swapchain.format().as_raw() as u32,
            width: extent.width,
            height: extent.height,
            layout: vk::ImageLayout::UNDEFINED.as_raw() as u32,
        }))
    }

    /// Transitions the image from GENERAL to PRESENT_SRC_KHR, ends the command
    /// buffer, submits it and presents.
    pub fn end_frame(&mut self, frame: Frame) -> Result<(), Error> {
        let image_index = self.acquired.take().ok_or(Error::NoFrameInFlight)?;
        let sync = &self.frames[self.frame_index];
        let image = self.swapchain.image(image_index);
        debug_assert_eq!(frame.command_buffer, sync.command_buffer.as_raw());
        debug_assert_eq!(frame.image, image.as_raw());
        let render_finished = self.render_finished[image_index as usize];

        let to_present = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range());

        // The waits and signals use ALL_COMMANDS: the recorded work starts with a
        // barrier whose source scope is ALL_COMMANDS, so the acquire wait chains
        // into it, whatever stage the library uses.
        let wait = [vk::SemaphoreSubmitInfo::default()
            .semaphore(sync.image_available)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let command_buffers =
            [vk::CommandBufferSubmitInfo::default().command_buffer(sync.command_buffer)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait)
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signal);

        // SAFETY: the command buffer is in the recording state (begin_frame), and
        // every handle belongs to this device.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                sync.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_present)),
            );
            self.device
                .end_command_buffer(sync.command_buffer)
                .map_err(vk_error("vkEndCommandBuffer"))?;
            self.device
                .queue_submit2(self.queue, &[submit], sync.in_flight)
                .map_err(vk_error("vkQueueSubmit2"))?;
        }

        let swapchains = [self.swapchain.handle()];
        let image_indices = [image_index];
        let wait_semaphores = [render_finished];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        // SAFETY: the image was acquired from this swapchain in begin_frame.
        match unsafe { self.swapchain.loader().queue_present(self.queue, &present) } {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_recreate = true,
            Err(e) => return Err(vk_error("vkQueuePresentKHR")(e)),
        }

        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }

    /// Blocks until the device is idle.
    pub fn wait_idle(&self) {
        // SAFETY: the device is alive for the lifetime of self.
        if let Err(e) = unsafe { self.device.device_wait_idle() } {
            log::error!("vkDeviceWaitIdle failed: {e:?}");
        }
    }

    fn create_frame_sync(&mut self) -> Result<(), Error> {
        for _ in 0..FRAMES_IN_FLIGHT {
            // SAFETY: plain object creation on a live device.
            unsafe {
                let command_pool = self
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                            .queue_family_index(self.queue_family),
                        None,
                    )
                    .map_err(vk_error("vkCreateCommandPool"))?;
                // Push first so Drop destroys the pool if a later call fails.
                self.frames.push(FrameSync {
                    command_pool,
                    command_buffer: vk::CommandBuffer::null(),
                    in_flight: vk::Fence::null(),
                    image_available: vk::Semaphore::null(),
                });
                let sync = self.frames.last_mut().expect("just pushed");
                sync.command_buffer = self
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .map_err(vk_error("vkAllocateCommandBuffers"))?[0];
                sync.in_flight = self
                    .device
                    .create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                    .map_err(vk_error("vkCreateFence"))?;
                sync.image_available = self
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(vk_error("vkCreateSemaphore"))?;
            }
        }
        Ok(())
    }

    fn ensure_render_finished_semaphores(&mut self) -> Result<(), Error> {
        while self.render_finished.len() < self.swapchain.image_count() {
            // SAFETY: plain object creation on a live device.
            let semaphore = unsafe {
                self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(vk_error("vkCreateSemaphore"))?;
            self.render_finished.push(semaphore);
        }
        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<(), Error> {
        // Simple and correct: nothing may still use the old swapchain images.
        self.wait_idle();
        self.swapchain.recreate(
            &self.instance,
            &self.surface_fn,
            self.surface,
            self.physical_device,
            self.width,
            self.height,
        )?;
        self.ensure_render_finished_semaphores()?;
        self.needs_recreate = false;
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.wait_idle();
        // SAFETY: the device is idle, so no object below is in use. Objects are
        // destroyed in reverse creation order; the debug messenger goes last
        // before the instance so it reports every destruction.
        unsafe {
            for semaphore in self.render_finished.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for sync in self.frames.drain(..) {
                self.device.destroy_semaphore(sync.image_available, None);
                self.device.destroy_fence(sync.in_flight, None);
                // Frees the command buffer too.
                self.device.destroy_command_pool(sync.command_pool, None);
            }
            self.swapchain.destroy();
            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            if let Some((loader, messenger)) = self.debug_messenger.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

pub(crate) fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}
