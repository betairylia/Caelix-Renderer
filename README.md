# Caelix-Renderer

Native Vulkan renderer for the Caelix voxel engine. The core is a C++20 shared
library (`caelix_renderer`) that exposes a small C ABI (`include/caelix/caelix.h`).
A Rust standalone host (`host-rs/`) links the library through bindgen-generated
bindings. A Unity native plugin that uses the same C ABI comes later.

## Architecture

The host owns the Vulkan instance, the device and the swapchain. The library
never creates a device: it publishes what it needs
(`caelix_get_device_requirements`: Vulkan 1.3, extensions, feature bits), the
host creates a device that meets those requirements, and the library adopts it
(`caelix_device_adopt`). The library loads Vulkan through the host's
`vkGetInstanceProcAddr` (volk), so it sees the same layers as the host. Each
frame, the library records its work into a command buffer that the host
provides; the host submits and presents. Vulkan handles cross the C ABI as
`uint64_t`, so the header needs no Vulkan headers.

In `host-rs/`:

- `caelix-sys`: raw bindings to the C ABI.
- `caelix-vk`: instance, device, swapchain and frame loop. It is the only Rust
  crate with Vulkan code (`ash`); its public API exposes raw `u64` handles only.
- `caelix-app`: the window (winit), the CLI and the glue between the two.

### Layout contract

A render call receives the target image in the layout that
`CaelixTargetImage::layout` names, and leaves it in `VK_IMAGE_LAYOUT_GENERAL`.
The host transitions it from `GENERAL` for present (or for its own use).

## Build

Requirements: Visual Studio 2022 (MSVC and its bundled LLVM/libclang),
CMake 3.28 or newer, the Vulkan SDK (`VULKAN_SDK` set), and a Rust MSVC toolchain.

```
cmake --preset vs2022
cmake --build --preset release
cd host-rs && cargo run -p caelix-app
```

The first CMake configure fetches the pinned dependencies (volk,
VulkanMemoryAllocator) into `build/_deps/`.

## Running the host

`cargo run -p caelix-app -- [flags]`

- `--frames <N>`: exit with code 0 after N presented frames. Default: run until
  the window closes.
- `--verbose`: log level `debug` (includes the Vulkan loader and validation
  layer messages). Default level: `info`. `RUST_LOG` overrides both.
- `--resize-at <N>` (hidden test aid): after N presented frames, request a
  960x540 window size once, so the swapchain recreation path runs without user
  input.

Debug builds enable `VK_LAYER_KHRONOS_validation` when it is installed; its
messages are logged with target `vulkan`. Library messages use target `caelix`.
Release builds run without validation.

Implicit layers that other software installs can add loader warnings that are
not from this project. Example: the Overwolf layers report an old API version;
set `DISABLE_VULKAN_OW_OVERLAY_LAYER=1` and `DISABLE_VULKAN_OW_OBS_CAPTURE=1`
to disable them for a run.

## Environment variables

- `CAELIX_LIB_DIR`: directory that contains `caelix_renderer.lib` and
  `caelix_renderer.dll`. Default: `<repo>/build/bin/Release`. The Rust host
  links the Release build only.
- `LIBCLANG_PATH`: directory that contains `libclang.dll`, used by bindgen.
  `host-rs/.cargo/config.toml` sets it to the 64-bit Visual Studio 2022
  Community LLVM directory (`VC/Tools/Llvm/x64/bin`); set your own value if your
  installation is different. Do not use `VC/Tools/Llvm/bin`: its
  `libclang.dll` is 32-bit.
