//! Standalone host for the Caelix renderer.
//!
//! The host owns the window, the Vulkan device and the swapchain (`caelix-vk`).
//! The library adopts the device and records into the host's command buffer
//! through the C ABI (`caelix-sys`). This crate contains no Vulkan code.

use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::time::Instant;

use anyhow::{anyhow, bail, Context as _};
use caelix_sys::*;
use clap::Parser;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Parser, Debug)]
#[command(about = "Standalone host for the Caelix native renderer")]
struct Args {
    /// Exit with code 0 after N presented frames. Default: run until the window closes.
    #[arg(long, value_name = "N")]
    frames: Option<u64>,
    /// Debug log level (includes Vulkan validation layer and loader messages).
    /// RUST_LOG overrides this.
    #[arg(long)]
    verbose: bool,
    /// Test aid: after N presented frames, request a 960x540 window size once,
    /// so the swapchain recreation path runs without user input.
    #[arg(long, value_name = "N", hide = true)]
    resize_at: Option<u64>,
}

/// Everything that lives while the window exists. Drop order matters: Drop
/// releases the library's device first, then the fields drop in declaration
/// order (Vulkan context, then window).
struct State {
    caelix_device: *mut CaelixDevice,
    context: caelix_vk::Context,
    window: Window,
}

impl Drop for State {
    fn drop(&mut self) {
        self.context.wait_idle();
        // SAFETY: the device is idle and the pointer came from caelix_device_adopt.
        unsafe { caelix_device_destroy(self.caelix_device) };
        log::debug!("caelix device destroyed");
    }
}

struct App {
    args: Args,
    state: Option<State>,
    start: Instant,
    frames_presented: u64,
    resize_requested: bool,
    error: Option<anyhow::Error>,
}

impl App {
    fn create_state(&self, event_loop: &ActiveEventLoop) -> anyhow::Result<State> {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Caelix Renderer")
                    .with_inner_size(PhysicalSize::new(1280, 720)),
            )
            .context("cannot create the window")?;
        let size = window.inner_size();

        let requirements = device_requirements();
        let context = caelix_vk::Context::new(
            &window,
            size.width,
            size.height,
            &requirements,
            cfg!(debug_assertions),
        )
        .context("cannot create the Vulkan context")?;

        let handles = context.adopt_handles();
        let desc = CaelixDeviceAdoptDesc {
            get_instance_proc_addr: handles.get_instance_proc_addr as *mut c_void,
            instance: handles.instance,
            physical_device: handles.physical_device,
            device: handles.device,
            graphics_queue: handles.graphics_queue,
            graphics_queue_family: handles.graphics_queue_family,
        };
        let mut caelix_device: *mut CaelixDevice = ptr::null_mut();
        // SAFETY: desc holds live handles that outlive the adopted device (State drop order).
        let result = unsafe { caelix_device_adopt(&desc, &mut caelix_device) };
        if result != CaelixResult_CAELIX_OK || caelix_device.is_null() {
            bail!("caelix_device_adopt failed: CaelixResult {result}");
        }
        Ok(State { caelix_device, context, window })
    }

    fn draw(&mut self, event_loop: &ActiveEventLoop) -> anyhow::Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Ok(());
        };
        if let Some(frame) = state.context.begin_frame()? {
            let target = CaelixTargetImage {
                image: frame.image,
                format: frame.format,
                width: frame.width,
                height: frame.height,
                layout: frame.layout,
            };
            let rgba = cycle_color(self.start.elapsed().as_secs_f32());
            // SAFETY: the command buffer is recording and the image is the acquired one.
            let result = unsafe {
                caelix_clear_target(state.caelix_device, frame.command_buffer, &target, rgba.as_ptr())
            };
            if result != CaelixResult_CAELIX_OK {
                bail!("caelix_clear_target failed: CaelixResult {result}");
            }
            state.context.end_frame(frame)?;
            self.frames_presented += 1;
        } else {
            log::debug!("frame skipped: swapchain recreated or window size is 0");
        }

        if let Some(at) = self.args.resize_at {
            if !self.resize_requested && self.frames_presented >= at {
                self.resize_requested = true;
                log::info!("resize test: requesting 960x540 after {} frames", self.frames_presented);
                let _ = state.window.request_inner_size(PhysicalSize::new(960, 540));
            }
        }
        if let Some(limit) = self.args.frames {
            if self.frames_presented >= limit {
                log::info!("{} frames presented, exiting", self.frames_presented);
                event_loop.exit();
                return Ok(());
            }
        }
        state.window.request_redraw();
        Ok(())
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        log::error!("{error:#}");
        self.error = Some(error);
        event_loop.exit();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match self.create_state(event_loop) {
            Ok(state) => {
                state.window.request_redraw();
                self.state = Some(state);
                self.start = Instant::now();
            }
            Err(e) => self.fail(event_loop, e),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = self.state.as_mut() {
                    log::debug!("window resized to {}x{}", size.width, size.height);
                    state.context.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.draw(event_loop) {
                    self.fail(event_loop, e);
                }
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // wait idle, caelix_device_destroy, then the context (State::drop).
        self.state = None;
    }
}

/// Forwards library log messages to `log` with target "caelix".
unsafe extern "C" fn caelix_log(level: CaelixLogLevel, message: *const c_char, _user_data: *mut c_void) {
    if message.is_null() {
        return;
    }
    // SAFETY: the library passes a NUL-terminated string valid during the call.
    let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    // bindgen emits the C enum as plain integer constants, so compare instead of match.
    let level = if level == CaelixLogLevel_CAELIX_LOG_DEBUG {
        log::Level::Debug
    } else if level == CaelixLogLevel_CAELIX_LOG_INFO {
        log::Level::Info
    } else if level == CaelixLogLevel_CAELIX_LOG_WARNING {
        log::Level::Warn
    } else {
        log::Level::Error
    };
    log::log!(target: "caelix", level, "{message}");
}

/// Converts the library's requirements to `caelix_vk` types. This lives in the
/// app so that `caelix-vk` stays independent of `caelix-sys`.
fn device_requirements() -> caelix_vk::DeviceRequirements {
    let mut raw = CaelixDeviceRequirements {
        api_version: 0,
        device_extension_count: 0,
        device_extensions: ptr::null(),
        feature_flags: 0,
    };
    // SAFETY: raw is writable; the extension strings are static in the library.
    unsafe { caelix_get_device_requirements(&mut raw) };
    let device_extensions = (0..raw.device_extension_count as usize)
        .map(|i| unsafe { CStr::from_ptr(*raw.device_extensions.add(i)) }.to_owned())
        .collect();
    let has = |bit: CaelixDeviceFeatureBits| raw.feature_flags & bit as u32 != 0;
    caelix_vk::DeviceRequirements {
        api_version: raw.api_version,
        device_extensions,
        features: caelix_vk::Features {
            buffer_device_address: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_BUFFER_DEVICE_ADDRESS),
            descriptor_indexing: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_DESCRIPTOR_INDEXING),
            synchronization2: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_SYNCHRONIZATION2),
            dynamic_rendering: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_DYNAMIC_RENDERING),
            timeline_semaphore: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_TIMELINE_SEMAPHORE),
            acceleration_structure: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_ACCELERATION_STRUCTURE),
            ray_query: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_RAY_QUERY),
            shader_int64: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_SHADER_INT64),
            scalar_block_layout: has(CaelixDeviceFeatureBits_CAELIX_FEATURE_SCALAR_BLOCK_LAYOUT),
        },
    }
}

/// A colour whose hue turns once every 10 seconds, so frames visibly advance.
fn cycle_color(seconds: f32) -> [f32; 4] {
    let hue = (seconds / 10.0).fract() * 6.0;
    let (saturation, value) = (0.6f32, 0.9f32);
    let chroma = value * saturation;
    let x = chroma * (1.0 - ((hue % 2.0) - 1.0).abs());
    let (r, g, b) = match hue as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let m = value - chroma;
    [r + m, g + m, b + m, 1.0]
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let default_filter = if args.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter)).init();

    let mut version = CaelixVersion { major: 0, minor: 0, patch: 0, abi: 0 };
    // SAFETY: version is writable; the version string is static in the library.
    let version_string = unsafe {
        caelix_get_version(&mut version);
        CStr::from_ptr(caelix_version_string())
    };
    log::info!("caelix_renderer {} (abi {})", version_string.to_string_lossy(), version.abi);
    if version.abi != CAELIX_ABI_VERSION {
        bail!(
            "ABI mismatch: library reports abi {}, bindings expect abi {}",
            version.abi,
            CAELIX_ABI_VERSION
        );
    }
    // SAFETY: caelix_log is a valid callback for the whole process lifetime.
    unsafe { caelix_set_log_callback(Some(caelix_log), ptr::null_mut()) };

    let event_loop = EventLoop::new().context("cannot create the event loop")?;
    let mut app = App {
        args,
        state: None,
        start: Instant::now(),
        frames_presented: 0,
        resize_requested: false,
        error: None,
    };
    let run_result = event_loop.run_app(&mut app);
    // In case the loop returned without `exiting` (error path).
    app.state = None;
    // SAFETY: disables library logging before the process tears down.
    unsafe { caelix_set_log_callback(None, ptr::null_mut()) };
    run_result.map_err(|e| anyhow!("event loop error: {e}"))?;
    match app.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
