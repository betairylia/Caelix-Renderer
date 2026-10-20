# Third-party notices

This file lists the third-party software that the Caelix renderer library
(`caelix_renderer`) contains when it is built, and the obligations that come
with it. It is a summary for the project, not legal advice. The license texts
themselves govern.

## NVIDIA Real-Time Denoisers (NRD)

- Version: v4.18.0, fetched by CMake (`cmake/Dependencies.cmake`) from
  `https://github.com/NVIDIA-RTX/NRD`.
- License: "NVIDIA RTX SDKs LICENSE" with the "NVIDIA RTX Supplement" (version of
  April 12, 2021). The full text is `LICENSE.txt` in the fetched source:
  `build/_deps/nrd-src/LICENSE.txt` after the first CMake configure. Read it before
  you ship a build.
- Accepted by the owner on 2026-10-14.

### What the library contains

`caelix_renderer` links NRD as a static library (`NRD_STATIC_LIBRARY=ON`), in
object code, with NRD's SPIR-V shaders embedded (`NRD_EMBEDS_SPIRV_SHADERS=ON`).
The NRD source is not modified. The project's own shaders include NRD's shader
header `NRD.hlsli` (and the generated `NRDConfig.hlsli`) at build time; only the
compiled SPIR-V of those shaders ships, not the header.

### Obligations

1. **Distribution in object code only, inside an application with material
   additional functionality** (license section 1(c), 2(a)). The renderer may not
   be offered as a stand-alone NRD redistribution.
2. **Source notice.** Section 2(b) requires the notice "This software contains
   source code provided by NVIDIA Corporation." in modifications and derivative
   works of NRD source code that are distributed. This project does not modify
   NRD and does not distribute NRD source code: it ships only the compiled
   library and the embedded compiled shaders. The notice is therefore not
   required today. If NRD source (or a modified NRD file, `NRD.hlsli` included) is
   ever distributed with this project, add that notice to it.
3. **Credits.** Supplement section 6.1(c): an application that uses NRD must
   attribute the use of the SDK and include the NVIDIA Marks on its credit
   screen; when the application has no credit screen, prominently in its end-user
   documentation. Proposed credit line: "This product uses NVIDIA Real-Time
   Denoisers (NRD)." The game side (Titania, any host that ships this library)
   owns this obligation.
4. **NVIDIA Marks and logos need written approval.** Supplement section 6.2(c):
   use of the NVIDIA Marks (NVIDIA RTX, GeForce RTX, ...) and NVIDIA logos is
   authorized only after NVIDIA approves a sample in writing (submit at least 10
   business days before use). Do not put an NVIDIA logo on a credit screen,
   website or marketing material without that approval.
5. **No endorsement claim** (section 4(c)): do not state that the application is
   sponsored or endorsed by NVIDIA.
6. **Interoperability** (supplement section 1): an application that incorporates
   NRD must be fully interoperable with compatible NVIDIA GPUs.
7. **No open-source relicensing** (section 4(e)): NRD must not become subject to
   a license that requires source disclosure or free redistribution. Keep this in
   mind if the renderer is ever published under an open-source license.
8. **Downstream terms** (section 2(c)): whoever distributes the library must do so
   under terms at least as protective of NVIDIA's rights as the NRD license.

## NVIDIA DLSS (NGX SDK)

- Version: v310.9.16 (DLSS 310.9.16 SDK), fetched by CMake (`cmake/Dependencies.cmake`)
  from `https://github.com/NVIDIA/DLSS` only when `CAELIX_DLSS=ON` (the default on
  Windows). `CAELIX_DLSS=OFF` removes all of it: no NGX code, no NGX library, no
  DLLs.
- License: "NVIDIA RTX SDKs LICENSE" with the "NVIDIA RTX Supplement" (version of
  March 14, 2024; the same license family as NRD, with stricter terms for the DLSS and
  NGX SDKs). The full text is `LICENSE.txt` in the fetched source:
  `build/_deps/ngx-src/LICENSE.txt` after the first CMake configure. Read it before you
  ship a build. Accepted by the owner for distribution on 2026-10-20.

### What the library contains and ships

- `caelix_renderer.dll` statically links the NGX import library `nvsdk_ngx_d.lib`
  (dynamic CRT; `nvsdk_ngx_d_dbg.lib` in Debug builds) from
  `lib/Windows_x86_64/x64`, in object code. The SDK headers are used at build time
  only; no SDK source is modified or distributed.
- The feature DLLs `nvngx_dlss.dll` (Super Resolution) and `nvngx_dlssd.dll` (Ray
  Reconstruction) from `lib/Windows_x86_64/rel` are copied next to
  `caelix_renderer.dll` (and into the Unity plugin directory when
  `CAELIX_UNITY_PLUGIN_DIR` is set). These are the files that ship.
- The development DLLs of `lib/Windows_x86_64/dev` (on-screen watermark and debug
  overlay) are copied instead only with `CAELIX_DLSS_DEV_DLL=ON`. **They must never
  ship.**

### Obligations

1. **Distribution in object code only, inside an application with material
   additional functionality** (license sections 1(c), 2(a)); never as a stand-alone
   redistribution of the SDK or its DLLs.
2. **NVIDIA marks in the application** (supplement section 7.1(b), stronger than
   NRD's credit-screen rule): an application that incorporates the DLSS or NGX SDK
   must attribute the use of the SDK and include the NVIDIA Marks on its splash
   screens, in its about box (when it has one) and in the credits of a game. The marks
   and logos need NVIDIA's prior written approval of a sample (supplement section
   7.2(c): submit at least two weeks before use; no answer within ten business days
   means not approved). The game side (Titania, any host that ships this library) owns
   this obligation.
3. **Notify NVIDIA before a commercial release** (supplement section 4) of an
   application, or of a plug-in to a commercial application, that incorporates the
   DLSS or NGX SDK: `https://developer.nvidia.com/sw-notification`, with company,
   publisher and developer name, the SDK used, application name, platform, scheduled
   ship date and a link to the product or a video.
4. **NVIDIA GPUs only** (supplement section 1): the DLSS and NGX SDKs are licensed for
   applications that use them on systems with NVIDIA GPUs, and the application must
   stay fully interoperable with compatible NVIDIA GPUs. (The library itself runs on
   other GPUs without DLSS: `caelix_upscaler_query` reports the feature as
   unavailable.)
5. **Cloud use is limited** (supplement section 3): only cloud services on NVIDIA vGPU
   software are in scope.
6. **Integration quality** (supplement section 6.2): a public DLSS integration with
   material stability, performance or image quality issues must be fixed quickly;
   NVIDIA may disable it as a last resort. NVIDIA may also update the SDK over the air
   (section 6.1).
7. **No endorsement claim** (section 4(c)), **no open-source relicensing** (section
   4(e)), **downstream terms at least as protective** (section 2(c)), as for NRD above.
8. **Source notice** (section 2(b)): only for distributed modifications of SDK source;
   this project distributes none.

## Other dependencies

volk, VulkanMemoryAllocator and unordered_dense (MIT licenses) are compiled into
the library; Catch2 (BSL-1.0) and vk-bootstrap (MIT) are test-only. NRD's build
also fetches two NVIDIA projects under the MIT license (`LICENSE.txt` in
`build/_deps/shadermake-src` and `build/_deps/mathlib-src`): ShaderMake, whose
tool compiles NRD's shaders at build time and whose small `ShaderMakeBlob` library
is linked into NRD (so into `caelix_renderer`), and MathLib, a header-only math
library compiled into NRD. The MIT license requires its copyright and permission
notice in copies of the software; ship these `LICENSE.txt` files (or their text)
with the library.
