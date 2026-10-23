// Hash table creation (shaders/hash/caelix_hash_kernels.hlsl): one thread per payload i < capacity
// writes free stack entry i = i and marks payload i free; thread 0 sets top = capacity. The counters
// and the key table are cleared by every maintenance pass.
#include "hash/caelix_hash_kernels.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint i = dispatch_id.x;
    if (i >= g_push.payload_capacity)
    {
        return;
    }
    const HashTable t = HashTableOfPush();
    CAELIX_STORE_U32(t.stack + kHashStackEntries + uint64_t(i) * 4, i);
    CAELIX_STORE_U32(HashPayloadAddress(t, i) + kHashPayloadState, kHashStateFree);
    if (i == 0)
    {
        CAELIX_STORE_U32(t.stack + kHashStackTop, g_push.payload_capacity);
    }
}
