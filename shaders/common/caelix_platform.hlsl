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
// Storage image with a known format qualifier (a SPIR-V image format name such as
// "rgba16f", "rgba8", "r32f"), so that the kernel can READ it without
// shaderStorageImageReadWithoutFormat: CAELIX_FORMAT("rgba16f") RWTexture2D<float4> g_x;
#define CAELIX_FORMAT(name) [[vk::image_format(name)]]

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

// Atomics on a buffer device address (DXC 1.8 has no vk::BufferPointer): the address becomes a
// PhysicalStorageBuffer pointer through OpConvertUToPtr, and the SPIR-V atomic takes it directly.
// Device scope. The plain forms are relaxed (order with barriers) and return the value before the
// operation; _ACQ_REL orders the caller's other uniform-memory accesses around it (a counter that
// hands data from one invocation to another); LOAD / STORE are coherent single-word accesses.
typedef vk::SpirvType</* OpTypePointer */ 32, 8, 8,
                      vk::Literal<vk::integral_constant<uint, /* PhysicalStorageBuffer */ 5349> >, uint>
    CaelixPointerU32;
[[vk::ext_instruction(/* OpConvertUToPtr */ 120)]]
CaelixPointerU32 CaelixAddressToPointerU32(uint64_t address);
[[vk::ext_instruction(/* OpAtomicLoad */ 227)]]
uint CaelixSpirvAtomicLoad(CaelixPointerU32 pointer, uint scope, uint semantics);
[[vk::ext_instruction(/* OpAtomicStore */ 228)]]
void CaelixSpirvAtomicStore(CaelixPointerU32 pointer, uint scope, uint semantics, uint value);
[[vk::ext_instruction(/* OpAtomicIAdd */ 234)]]
uint CaelixSpirvAtomicAdd(CaelixPointerU32 pointer, uint scope, uint semantics, uint value);
[[vk::ext_instruction(/* OpAtomicOr */ 241)]]
uint CaelixSpirvAtomicOr(CaelixPointerU32 pointer, uint scope, uint semantics, uint value);
[[vk::ext_instruction(/* OpAtomicCompareExchange */ 230)]]
uint CaelixSpirvAtomicCompareExchange(CaelixPointerU32 pointer, uint scope, uint equal_semantics,
                                      uint unequal_semantics, uint value, uint comparator);
// Scope Device = 1; semantics AcquireRelease (0x8) | UniformMemory (0x40) = 0x48.
#define CAELIX_ATOMIC_ADD_U32(addr, v) CaelixSpirvAtomicAdd(CaelixAddressToPointerU32(addr), 1u, 0u, v)
#define CAELIX_ATOMIC_ADD_ACQ_REL_U32(addr, v) \
    CaelixSpirvAtomicAdd(CaelixAddressToPointerU32(addr), 1u, 0x48u, v)
#define CAELIX_ATOMIC_OR_U32(addr, v) CaelixSpirvAtomicOr(CaelixAddressToPointerU32(addr), 1u, 0u, v)
#define CAELIX_ATOMIC_LOAD_U32(addr) CaelixSpirvAtomicLoad(CaelixAddressToPointerU32(addr), 1u, 0u)
#define CAELIX_ATOMIC_STORE_U32(addr, v) CaelixSpirvAtomicStore(CaelixAddressToPointerU32(addr), 1u, 0u, v)
// Publication of data from one invocation to others in the same dispatch (common/caelix_hash_table.hlsl):
// the writer's RELEASE store (semantics Release 0x4 | UniformMemory 0x40) orders its earlier writes
// before the word; a reader's ACQUIRE load (Acquire 0x2 | UniformMemory 0x40) that sees the word orders
// its later reads after it. The compare-exchange is relaxed and returns the value it found (it stored
// `v` only when that value equals `compare`).
#define CAELIX_ATOMIC_LOAD_ACQUIRE_U32(addr) CaelixSpirvAtomicLoad(CaelixAddressToPointerU32(addr), 1u, 0x42u)
#define CAELIX_ATOMIC_STORE_RELEASE_U32(addr, v) \
    CaelixSpirvAtomicStore(CaelixAddressToPointerU32(addr), 1u, 0x44u, v)
#define CAELIX_ATOMIC_CAS_U32(addr, compare, v) \
    CaelixSpirvAtomicCompareExchange(CaelixAddressToPointerU32(addr), 1u, 0u, 0u, v, compare)

#endif  // CAELIX_PLATFORM_HLSL
