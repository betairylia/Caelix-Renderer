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
