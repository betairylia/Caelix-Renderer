// Hash table maintenance, kernel 2 of 3 (shaders/hash/caelix_hash_kernels.hlsl): one thread per
// payload of the pool, after the key table was cleared.
//   free payload                    nothing;
//   frame - touched > max_age       freed: state free, pushed onto the free stack (evicted);
//   otherwise                       its key (recomputed from the header's key words) is inserted into
//                                   the key table (HashRebuildInsert); a key whose 8-probe window is
//                                   full frees its payload (rebuild_dropped); a payload that stays is
//                                   live, and active when the previous frame touched it
//                                   (touched == frame - 1): it is appended to the active list.
// The pushes, the active list and the counters are aggregated per wave (one atomic each). No pop runs
// in this dispatch, so the pushes need no ordering with anything else.
#include "hash/caelix_hash_kernels.hlsl"

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const HashTable t = HashTableOfPush();
    const uint i = dispatch_id.x;
    // Every lane reaches the wave operations below (no early return).
    bool freed = false;
    bool evicted = false;
    bool dropped = false;
    bool live = false;
    bool active = false;
    if (i < g_push.payload_capacity)
    {
        const uint64_t header = HashPayloadAddress(t, i);
        const uint state = CAELIX_LOAD_U32(header + kHashPayloadState);
        if (state != kHashStateFree)
        {
            const uint touched = CAELIX_LOAD_U32(header + kHashPayloadTouched);
            if (g_push.frame - touched > g_push.max_age)
            {
                evicted = true;
            }
            else
            {
                uint words[kHashMaxKeyWords];
                HashPayloadKeyWords(t, i, g_push.key_words, words);
                if (HashRebuildInsert(t, HashFingerprint(words, g_push.key_words), i))
                {
                    live = true;
                    active = touched == g_push.frame - 1u;
                }
                else
                {
                    dropped = true;
                }
            }
            freed = evicted || dropped;
            if (freed)
            {
                CAELIX_STORE_U32(header + kHashPayloadState, kHashStateFree);
            }
        }
    }

    // Pushes: entries[top .. top + n) of the wave's freed payloads.
    const uint freed_count = WaveActiveCountBits(freed);
    uint push_base = 0;
    if (WaveIsFirstLane() && freed_count != 0u)
    {
        push_base = CAELIX_ATOMIC_ADD_U32(t.stack + kHashStackTop, freed_count);
    }
    push_base = WaveReadLaneFirst(push_base);
    if (freed)
    {
        const uint entry = push_base + WavePrefixCountBits(freed);
        CAELIX_STORE_U32(t.stack + kHashStackEntries + uint64_t(entry) * 4, i);
    }
    // The active list.
    const uint active_count = WaveActiveCountBits(active);
    uint active_base = 0;
    if (WaveIsFirstLane() && active_count != 0u)
    {
        active_base = CAELIX_ATOMIC_ADD_U32(g_push.dispatch + kHashDispatchCount, active_count);
    }
    active_base = WaveReadLaneFirst(active_base);
    if (active)
    {
        CAELIX_STORE_U32(g_push.active + uint64_t(active_base + WavePrefixCountBits(active)) * 4, i);
    }
    // Counters.
    const uint live_count = WaveActiveCountBits(live);
    const uint evicted_count = WaveActiveCountBits(evicted);
    const uint dropped_count = WaveActiveCountBits(dropped);
    if (WaveIsFirstLane())
    {
        HashCounterAdd(t, kHashCounterLive, live_count);
        HashCounterAdd(t, kHashCounterEvicted, evicted_count);
        HashCounterAdd(t, kHashCounterRebuildDropped, dropped_count);
    }
}
