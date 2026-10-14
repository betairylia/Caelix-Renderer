// Platform layer for the Caelix kernels.
//
// This header is the ONLY place with DXC-specific syntax (vk:: intrinsics,
// [[vk::...]] attributes, loop and branch attributes). Every other shader file
// uses the macros below. A Slang port replaces this header and keeps the macro
// names.
#ifndef CAELIX_PLATFORM_HLSL
#define CAELIX_PLATFORM_HLSL

#define CAELIX_BRANCH [branch]
#define CAELIX_UNROLL [unroll]
#define CAELIX_LOOP [loop]

// Push constants: CAELIX_PUSH_CONSTANTS(FinalizePush, g_push);
#define CAELIX_PUSH_CONSTANTS(type, name) [[vk::push_constant]] type name

// Descriptor binding of a resource that has no device address (acceleration
// structures, images): CAELIX_BINDING(0, 0) RaytracingAccelerationStructure g_tlas;
#define CAELIX_BINDING(index, set_index) [[vk::binding(index, set_index)]]

// Storage image without a format qualifier (needs shaderStorageImageWriteWithoutFormat).
// DXC 1.8 infers a format from the element type (RWTexture2D<float4> -> Rgba32f)
// when the attribute is missing and has no command-line switch to stop that, so
// "unknown" must be stated: CAELIX_FORMAT_UNKNOWN RWTexture2D<float4> g_color;
#define CAELIX_FORMAT_UNKNOWN [[vk::image_format("unknown")]]

// No 16-bit types: `half` is a 32-bit float (Unity on PC does the same). DXC
// already declares `half` as a built-in type, so these are macros, not typedefs.
#define half float
#define half2 float2
#define half3 float3
#define half4 float4

// Buffer device address loads and stores. `addr` is a uint64_t device address.
// The default alignment of a scalar access is its size; the vector forms state
// their alignment, and the address must be aligned to it.
#define CAELIX_LOAD_U32(addr) vk::RawBufferLoad<uint>(addr)
#define CAELIX_LOAD_U32x2(addr) vk::RawBufferLoad<uint2>(addr, 8)
#define CAELIX_LOAD_U32x4(addr) vk::RawBufferLoad<uint4>(addr, 16)
#define CAELIX_LOAD_U64(addr) vk::RawBufferLoad<uint64_t>(addr, 8)
// A whole struct; `align` is the struct's alignment (16 when it has a float4).
#define CAELIX_LOAD_STRUCT(type, addr, align) vk::RawBufferLoad<type>(addr, align)
#define CAELIX_STORE_U32(addr, v) vk::RawBufferStore<uint>(addr, v)
#define CAELIX_STORE_U32x2(addr, v) vk::RawBufferStore<uint2>(addr, v, 8)
#define CAELIX_STORE_U32x4(addr, v) vk::RawBufferStore<uint4>(addr, v, 16)

#endif  // CAELIX_PLATFORM_HLSL
