Caelix Renderer: Unity native plugin
====================================

The native Vulkan renderer of the Caelix voxel engine for Unity, as a Unity
native plugin. It path traces the voxel world on the GPU with Vulkan ray
queries, denoises with NVIDIA NRD and can upscale with NVIDIA DLSS.

This package matches Caelix 0.2.0 (plugin packet version 8). The C# side of
the bridge is in the Caelix package: Runtime/Rendering/Native/.


Files
-----

- caelix_unity.dll      the Unity plugin (device interception, packet bridge)
- caelix_renderer.dll   the renderer library (contains NVIDIA NRD)
- nvngx_dlss.dll        NVIDIA DLSS Super Resolution
- nvngx_dlssd.dll       NVIDIA DLSS Ray Reconstruction
- NOTICE.txt            credits and third-party notices
- licenses/             license texts


Install
-------

1. Copy the four DLL files into the Caelix package folder
   Runtime/Plugins/x86_64/ (the package ships the matching .dll.meta files
   with "Load on startup" set).
2. Set the Unity graphics API for Windows to Vulkan.
3. In the Unity menu use Caelix > Native Renderer > Setup Scene, or add the
   CaelixNativeRenderer component and the CaelixNativeRendererFeature to the
   URP renderer by hand.
4. After every replacement of the DLL files, restart the Unity Editor.


Requirements
------------

- Windows 10 or 11, 64-bit. Unity 6000.5 or newer with URP 17.
- A GPU with Vulkan 1.3 and ray query support. DLSS needs an NVIDIA RTX GPU.
  Install a recent GPU driver.
- The Microsoft Visual C++ Redistributable 2015-2022 (x64):
  https://aka.ms/vs/17/release/vc_redist.x64.exe


Terms
-----

The Caelix renderer code is licensed under the MIT License
(licenses/Caelix-Renderer-LICENSE.txt). Source:
https://github.com/betairylia/Caelix-RenderingPlugin

The NVIDIA components (NVIDIA Real-Time Denoisers, linked into
caelix_renderer.dll; nvngx_dlss.dll and nvngx_dlssd.dll, NVIDIA DLSS) are
NVIDIA software, licensed under the NVIDIA RTX SDKs license (see
licenses/NVIDIA-NRD-LICENSE.txt and licenses/NVIDIA-DLSS-LICENSE.txt). You
may distribute them only in object code inside an application with material
additional functionality, never on their own. You must not change them or
reverse engineer, decompile or disassemble them, except where the law allows
it. NVIDIA does not sponsor or endorse this software.

An application that ships these files has obligations of its own: credit the
use of NVIDIA NRD and NVIDIA DLSS (credits screen, splash screen or about
box), get NVIDIA's written approval before using NVIDIA marks or logos, and
notify NVIDIA before a commercial release of an application that uses DLSS
(https://developer.nvidia.com/sw-notification). The license texts govern.

See NOTICE.txt for the credits and the component list.
