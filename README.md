# Caelix-Renderer

Native Vulkan renderer for the Caelix voxel engine. The core is a C++20 shared
library (`caelix_renderer`) that exposes a small C ABI (`include/caelix/caelix.h`).
A Rust standalone host (`host-rs/`) links the library through bindgen-generated
bindings. A Unity native plugin that uses the same C ABI comes later.

## Build

Requirements: Visual Studio 2022 (MSVC and its bundled LLVM/libclang),
CMake 3.28 or newer, the Vulkan SDK (`VULKAN_SDK` set), and a Rust MSVC toolchain.

```
cmake --preset vs2022
cmake --build --preset release
cd host-rs && cargo run -p caelix-app
```

The first CMake configure fetches the pinned dependencies (volk,
VulkanMemoryAllocator, vk-bootstrap) into `build/_deps/`.

## Environment variables

- `CAELIX_LIB_DIR`: directory that contains `caelix_renderer.lib` and
  `caelix_renderer.dll`. Default: `<repo>/build/bin/Release`. The Rust host
  links the Release build only.
- `LIBCLANG_PATH`: directory that contains `libclang.dll`, used by bindgen.
  `host-rs/.cargo/config.toml` sets it to the 64-bit Visual Studio 2022
  Community LLVM directory (`VC/Tools/Llvm/x64/bin`); set your own value if your
  installation is different. Do not use `VC/Tools/Llvm/bin`: its
  `libclang.dll` is 32-bit.
