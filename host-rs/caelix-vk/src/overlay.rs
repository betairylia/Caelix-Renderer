//! egui overlay: draws egui's primitives over the swapchain image after the library has written it.
//!
//! The renderer is `egui-ash-renderer` in dynamic rendering mode. It keeps one vertex and index
//! buffer pair per frame in flight. egui texture uploads go through one-time command buffers from
//! an own command pool; the renderer submits each upload to the queue and waits for the queue to
//! be idle, so an upload never races a frame that still reads the old texture. Textures that egui
//! frees are destroyed once the frame that drew last with them has completed.

use ash::vk;
use egui_ash_renderer::allocator::DefaultAllocator;
use egui_ash_renderer::{DynamicRendering, Options, RenderMode, Renderer};

use crate::{vk_error, Error};

pub(crate) struct Overlay {
    /// Dropped (all its GPU objects destroyed) before `device` can be destroyed: `Context::drop`
    /// drops the overlay first.
    renderer: Renderer<DefaultAllocator>,
    device: ash::Device,
    /// Pool of the one-time command buffers of the texture uploads.
    upload_pool: vk::CommandPool,
    /// Colour attachment format of the renderer's pipeline (the swapchain format).
    format: vk::Format,
    /// One view per swapchain image, for `vkCmdBeginRendering`. Empty without a swapchain.
    views: Vec<vk::ImageView>,
    /// egui textures to free: (frame that drew last with the texture, id).
    pending_frees: Vec<(u64, egui::TextureId)>,
}

fn overlay_error(call: &'static str) -> impl FnOnce(egui_ash_renderer::RendererError) -> Error {
    move |e| Error::Overlay(format!("{call}: {e}"))
}

/// The egui renderer converts its colours to sRGB in the shader unless the attachment does it.
fn is_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}

fn render_mode(format: vk::Format) -> RenderMode {
    RenderMode::DynamicRendering(DynamicRendering {
        color_attachment_format: format,
        depth_attachment_format: None,
        stencil_attachment_format: None,
    })
}

impl Overlay {
    /// Creates the egui renderer for `format` and one view per swapchain image.
    pub(crate) fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        queue_family: u32,
        format: vk::Format,
        images: &[vk::Image],
        frames_in_flight: usize,
    ) -> Result<Self, Error> {
        let renderer = Renderer::with_default_allocator(
            instance,
            physical_device,
            device.clone(),
            render_mode(format),
            Options { in_flight_frames: frames_in_flight, srgb_framebuffer: is_srgb(format), ..Options::default() },
        )
        .map_err(overlay_error("egui renderer creation"))?;
        // SAFETY: plain object creation on a live device.
        let upload_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(queue_family),
                None,
            )
        }
        .map_err(vk_error("vkCreateCommandPool"))?;
        let mut overlay = Self {
            renderer,
            device: device.clone(),
            upload_pool,
            format,
            views: Vec::new(),
            pending_frees: Vec::new(),
        };
        // On an error, dropping `overlay` destroys what exists so far.
        overlay.create_views(images)?;
        log::debug!("egui overlay created for {format:?}, {} views", overlay.views.len());
        Ok(overlay)
    }

    /// Destroys the image views. Call before the swapchain images go away; the caller guarantees
    /// that no pending frame uses them (the device is idle).
    pub(crate) fn destroy_views(&mut self) {
        for view in self.views.drain(..) {
            // SAFETY: the device is idle (caller), so the view is not in use.
            unsafe { self.device.destroy_image_view(view, None) };
        }
    }

    /// Adopts a recreated swapchain: new views, and a new pipeline when the format changed.
    /// `format` is UNDEFINED and `images` is empty when the swapchain has no images (size 0).
    pub(crate) fn set_swapchain(&mut self, format: vk::Format, images: &[vk::Image]) -> Result<(), Error> {
        self.destroy_views();
        if format != vk::Format::UNDEFINED && format != self.format {
            if is_srgb(format) != is_srgb(self.format) {
                // The renderer's sRGB option is fixed at creation; recreating the renderer would
                // lose egui's textures. Colours of the overlay are then off by the sRGB curve.
                log::warn!(
                    "egui overlay: swapchain format changed from {:?} to {format:?} (sRGB changed); \
                     overlay colours are not corrected",
                    self.format
                );
            }
            self.renderer.set_render_mode(render_mode(format)).map_err(overlay_error("egui set_render_mode"))?;
            self.format = format;
        }
        self.create_views(images)
    }

    fn create_views(&mut self, images: &[vk::Image]) -> Result<(), Error> {
        for &image in images {
            // SAFETY: the image is a live swapchain image of this device.
            let view = unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.format)
                        .subresource_range(crate::color_range()),
                    None,
                )
            }
            .map_err(vk_error("vkCreateImageView"))?;
            self.views.push(view);
        }
        Ok(())
    }

    /// Uploads egui's texture changes, then records the overlay into `cmd`: GENERAL ->
    /// COLOR_ATTACHMENT_OPTIMAL, a dynamic rendering pass (load and store) with egui's draws,
    /// COLOR_ATTACHMENT_OPTIMAL -> GENERAL. Textures that egui freed are destroyed once
    /// `frame_number` has completed (checked against `completed_frame` in later calls).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &mut self,
        queue: vk::Queue,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        image_index: u32,
        extent: vk::Extent2D,
        frame_number: u64,
        completed_frame: u64,
        mut textures_delta: egui::TexturesDelta,
        primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<(), Error> {
        // Taken out first: a TexturesDelta that is dropped with entries asserts in debug builds,
        // also on the error paths below.
        let set = std::mem::take(&mut textures_delta.set);
        let free = std::mem::take(&mut textures_delta.free);
        let mut result = Ok(());
        let renderer = &mut self.renderer;
        self.pending_frees.retain(|&(frame, id)| {
            if frame > completed_frame {
                return true;
            }
            if let Err(e) = renderer.free_texture(id) {
                result = Err(overlay_error("egui free_texture")(e));
            }
            false
        });
        result?;

        for (id, deltas) in &set {
            for delta in deltas {
                self.renderer
                    .set_texture(queue, self.upload_pool, *id, delta)
                    .map_err(overlay_error("egui set_texture"))?;
            }
        }

        let view = *self.views.get(image_index as usize).ok_or_else(|| {
            Error::Overlay(format!("no image view for swapchain image {image_index}"))
        })?;
        let to_attachment = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(crate::color_range());
        let to_general = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(crate::color_range());
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&attachment));

        // SAFETY: the command buffer is recording (between begin_frame and end_frame); the image
        // and the view belong to the acquired swapchain image.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_attachment)),
            );
            self.device.cmd_begin_rendering(cmd, &rendering);
        }
        let draw = self
            .renderer
            .cmd_draw(cmd, extent, pixels_per_point, primitives)
            .map_err(overlay_error("egui cmd_draw"));
        // SAFETY: as above; the rendering pass begun above is ended even when the draw failed, so
        // the command buffer stays valid.
        unsafe {
            self.device.cmd_end_rendering(cmd);
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_general)),
            );
        }
        draw?;

        self.pending_frees.extend(free.into_iter().map(|id| (frame_number, id)));
        Ok(())
    }
}

impl Drop for Overlay {
    fn drop(&mut self) {
        // The caller (Context) guarantees the device is idle. The renderer field drops after this
        // body and destroys its own objects (pipeline, buffers, textures, descriptor pool).
        self.destroy_views();
        // SAFETY: the device is idle, so no upload command buffer is in use.
        unsafe { self.device.destroy_command_pool(self.upload_pool, None) };
    }
}
