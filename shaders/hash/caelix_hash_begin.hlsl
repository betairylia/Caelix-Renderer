// Hash table maintenance, kernel 1 of 3 (shaders/hash/caelix_hash_kernels.hlsl), one thread:
// - the free stack fix-up: the lookups of the previous frame popped with atomicAdd(top, -1), which
//   leaves top below 0 when the stack ran empty; top is clamped to 0 before any push of this pass;
// - the counters (the table's and the user's words) and the dispatch (count 0) are cleared, after
//   the previous frame's readback copied them.
#include "hash/caelix_hash_kernels.hlsl"

[numthreads(1, 1, 1)]
void main()
{
    const HashTable t = HashTableOfPush();
    const int top = int(CAELIX_LOAD_U32(t.stack + kHashStackTop));
    if (top < 0)
    {
        CAELIX_STORE_U32(t.stack + kHashStackTop, 0u);
    }
    for (uint w = 0; w < kHashCounterBytes / 16; ++w)
    {
        CAELIX_STORE_U32x4(t.counters + w * 16, uint4(0, 0, 0, 0));
    }
    // VkDispatchIndirectCommand (0, 1, 1) and count 0 until caelix_hash_finish.
    CAELIX_STORE_U32x4(g_push.dispatch, uint4(0, 1, 1, 0));
}
