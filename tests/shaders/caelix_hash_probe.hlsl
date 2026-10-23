// Probe kernel of caelix_hash_table_test (tests/hash_table_test.cpp): runs the device API of the hash
// table (common/caelix_hash_table.hlsl) from many threads at once.
//   mode 0 fingerprint: thread i < count: the key words at input + 32 i -> HashFingerprint at
//                       output + 8 i
//   mode 1 insert:      thread i < count: HashLookupOrInsert of the key words at input + 32 i;
//                       (result, payload) at output + 8 i; the results are counted (HashCountResult)
//   mode 2 lookup:      thread i < count: HashLookup; (result, payload) at output + 8 i
//   mode 3 user pass:   the user's pass over the active list (indirect dispatch, 64 payloads per
//                       workgroup): thread t of workgroup g takes active entry 64 g + t below the count
//                       (input = the active list, extra = the dispatch) and writes `frame` at
//                       output + 4 payload, and marks the payload ready
#include "common/caelix_hash_table.hlsl"

struct HashProbePush
{
    uint64_t keys;
    uint64_t stack;
    uint64_t counters;
    uint64_t payloads;
    uint64_t input;
    uint64_t output;
    uint64_t extra;
    uint key_mask;
    uint payload_stride;
    uint key_words;
    uint frame;
    uint mode;
    uint count;
};

CAELIX_PUSH_CONSTANTS(HashProbePush, g_push);

void LoadWords(uint i, out uint words[kHashMaxKeyWords])
{
    const uint4 a = CAELIX_LOAD_U32x4(g_push.input + uint64_t(i) * 32);
    const uint4 b = CAELIX_LOAD_U32x4(g_push.input + uint64_t(i) * 32 + 16);
    words[0] = a.x;
    words[1] = a.y;
    words[2] = a.z;
    words[3] = a.w;
    words[4] = b.x;
    words[5] = b.y;
    words[6] = b.z;
    words[7] = b.w;
}

[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID,
          uint3 dispatch_id : SV_DispatchThreadID)
{
    HashTable t;
    t.keys = g_push.keys;
    t.stack = g_push.stack;
    t.counters = g_push.counters;
    t.payloads = g_push.payloads;
    t.key_mask = g_push.key_mask;
    t.payload_stride = g_push.payload_stride;
    if (g_push.mode == 3)
    {
        const uint index = (group_id.y * kHashDispatchRowGroups + group_id.x) * 64 + thread_id.x;
        if (index < CAELIX_LOAD_U32(g_push.extra + kHashDispatchCount))
        {
            const uint payload = CAELIX_LOAD_U32(g_push.input + uint64_t(index) * 4);
            CAELIX_STORE_U32(g_push.output + uint64_t(payload) * 4, g_push.frame);
            CAELIX_STORE_U32(HashPayloadAddress(t, payload) + kHashPayloadState, kHashStateReady);
        }
        return;
    }
    const uint i = dispatch_id.x;
    if (i >= g_push.count)
    {
        return;
    }
    uint words[kHashMaxKeyWords];
    LoadWords(i, words);
    const uint2 key = HashFingerprint(words, g_push.key_words);
    if (g_push.mode == 0)
    {
        CAELIX_STORE_U32x2(g_push.output + uint64_t(i) * 8, key);
        return;
    }
    uint payload;
    uint result;
    if (g_push.mode == 1)
    {
        result = HashLookupOrInsert(t, key, words, g_push.key_words, g_push.frame, payload);
        HashCountResult(t, result);
    }
    else
    {
        result = HashLookup(t, key, g_push.frame, payload);
    }
    CAELIX_STORE_U32x2(g_push.output + uint64_t(i) * 8, uint2(result, payload));
}
