Caelix Renderer viewer
======================

A viewer for Caelix voxel worlds (.cxw files). It path traces the world on the
GPU with Vulkan ray queries, denoises with NVIDIA NRD and can upscale with
NVIDIA DLSS.

This is a development build. It is not a product.


Requirements
------------

- Windows 10 or 11, 64-bit.
- A GPU with Vulkan 1.3 and ray query support. Tested only on an NVIDIA
  GeForce RTX 5080. DLSS needs an NVIDIA RTX GPU. Install a recent GPU
  driver.
- The Microsoft Visual C++ Redistributable 2015-2022 (x64). Most PCs with
  games have it. If the app does not start, install it from
  https://aka.ms/vs/17/release/vc_redist.x64.exe


Start
-----

1. Keep every file of this folder together.
2. Start caelix-app.exe.
3. The "Entities" window is open. Click "Load" and select a .cxw world.
4. Click a window title to open or close the window. F1 hides all windows.

The app can also start from a command line. Run "caelix-app.exe --help" for
the options. Example:

    caelix-app.exe --world my-world.cxw --upscaler dlss-sr

If the app crashes at start, an overlay program (for example Overwolf or OBS
game capture) can be the cause. Close it and try again.


Controls
--------

- W A S D: move. Q / E: down / up. Shift: 8x speed.
- Right mouse button (hold): look around.
- Mouse wheel: change the move speed.
- P: write the camera pose to the console.
- K: add a camera keyframe (Sequencer window). Space: play / pause the tour.
- F1: show / hide the windows.

Windows:
- Entities: load a world; move and turn its entities.
- Render: mode, quality, upscaler (DLSS), GI, sun, sky, fog, screenshot.
  Render > Sky > Load: a lat-long .exr or .hdr sky image (an image without
  the sun disk is best).
- Camera: position, angles, field of view, speed.
- Sequencer: camera tours; Load / Save as .json; export as PNG frames.
- Stats: frame times and memory.

Screenshots go to the folder of caelix-app.exe.


Terms
-----

The Caelix renderer code and this viewer are licensed under the MIT License
(licenses/Caelix-Renderer-LICENSE.txt), "as is", without any warranty.
Source: https://github.com/betairylia/Caelix-RenderingPlugin

The NVIDIA components (NVIDIA Real-Time Denoisers, linked into
caelix_renderer.dll; nvngx_dlss.dll and nvngx_dlssd.dll, NVIDIA DLSS) are
NVIDIA software, licensed under the NVIDIA RTX SDKs license (see
licenses/NVIDIA-NRD-LICENSE.txt and licenses/NVIDIA-DLSS-LICENSE.txt). You
must not copy them separately from this application, change them, or reverse
engineer, decompile or disassemble them, except where the law allows it.
NVIDIA does not sponsor or endorse this software.

See NOTICE.txt for the credits and the third-party licenses.
