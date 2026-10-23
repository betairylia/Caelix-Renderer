// The reusable GPU hash table (docs/LIGHT_NEE.md section 2): 64-bit keys -> indices of a payload
// pool that the user owns. Layouts mirror lib/src/rhi/hash_table_layout.h (every constant has the
// same name and value there; caelix_frame_layout_test compares the two files). The table knows only
// the payload header contract (state, touched, key words); it has no knowledge of any user.
//
// Key slot (16 bytes): key_lo, key_hi, payload, pad. key_lo 0 = empty; real keys have key_lo >= 1
// and key_hi != 0 (HashFingerprint forces both), so key_lo set and key_hi 0 is an insert in
// progress. Every key slot word is accessed through atomics (common/caelix_platform.hlsl): the
// inserter claims key_lo with a compare-exchange from 0, pops a payload, writes the payload header and
// the slot's payload word, then publishes key_hi with a RELEASE store; a reader that finds key_lo
// loads key_hi with ACQUIRE order and only then the payload word, so it sees the header writes too.
// HLSL Interlocked* functions imply no ordering, hence the platform macros.
//
// Lookup rules (HashLookup, HashLookupOrInsert): probe (key_lo + i) & key_mask for i < kHashProbes.
// Both words equal = hit (the payload's touched word is set to the caller's frame). key_lo equal and
// key_hi 0 = pending: no entry this frame and no insert (this keeps a key from being inserted twice:
// a thread that lost the compare-exchange to the same key sees exactly this). key_lo equal and another
// key_hi = another key with the same low word: skipped. Empty = miss (a lookup stops; an insert claims
// it). An insert that finds the free stack empty publishes the key with payload kHashNone: the entry
// counts as absent until the next rebuild.
//
// The key table is rebuilt from the live payloads in every maintenance pass
// (shaders/hash/caelix_hash_maintain.hlsl): no tombstones, no rehash, no back reference; pending and
// payload-less entries of the previous frame disappear by construction. A 64-bit fingerprint collision
// merges two keys (about 2^-64 per pair); documented, not handled.
#ifndef CAELIX_HASH_TABLE_HLSL
#define CAELIX_HASH_TABLE_HLSL

#include "common/caelix_platform.hlsl"

static const uint kHashNone = 0xFFFFFFFFu;
static const uint kHashProbes = 8;
static const uint kHashKeySlotsPerPayload = 4;
// Key slot.
static const uint kHashSlotKeyLo = 0;
static const uint kHashSlotKeyHi = 4;
static const uint kHashSlotPayload = 8;
static const uint kHashSlotBytes = 16;
// Payload header contract.
static const uint kHashPayloadState = 0;
static const uint kHashPayloadTouched = 4;
static const uint kHashPayloadKey = 8;
static const uint kHashMaxKeyWords = 8;
static const uint kHashStateFree = 0;
static const uint kHashStateUninitialised = 1;
static const uint kHashStateReady = 2;
// Free stack: int top, then the entries.
static const uint kHashStackTop = 0;
static const uint kHashStackEntries = 4;
// Counters (64 bytes): the table's words, then kHashCounterUserWords for the user.
static const uint kHashCounterLive = 0;
static const uint kHashCounterActive = 4;
static const uint kHashCounterInserted = 8;
static const uint kHashCounterPending = 12;
static const uint kHashCounterFailedSlot = 16;
static const uint kHashCounterFailedPayload = 20;
static const uint kHashCounterEvicted = 24;
static const uint kHashCounterRebuildDropped = 28;
static const uint kHashCounterUser = 32;
static const uint kHashCounterUserWords = 8;
static const uint kHashCounterBytes = 64;
// The user pass's indirect dispatch: VkDispatchIndirectCommand, then the active count.
static const uint kHashDispatchX = 0;
static const uint kHashDispatchY = 4;
static const uint kHashDispatchZ = 8;
static const uint kHashDispatchCount = 12;
static const uint kHashDispatchBytes = 16;
static const uint kHashDispatchRowGroups = 32768;
static const uint kHashThreads = 64;
static const uint kHashMaxPayloads = 1u << 21;

// Results of a lookup.
static const uint kHashHit = 0;          // the key is present with a payload (`payload`)
static const uint kHashInserted = 1;     // this call inserted the key (`payload`, header written)
static const uint kHashPending = 2;      // an insert of the key (or one with its key_lo) is in progress
static const uint kHashFullWindow = 3;   // the 8-probe window holds 8 other keys
static const uint kHashNoPayload = 4;    // this call claimed a slot but the free stack was empty
static const uint kHashEmptyEntry = 5;   // the key is present without a payload (an earlier kHashNoPayload)
static const uint kHashMissing = 6;      // lookup only: the key is not in the table

// The table as its users see it in a kernel.
struct HashTable
{
    uint64_t keys;      // key slots
    uint64_t stack;     // the free stack
    uint64_t counters;  // the counters
    uint64_t payloads;  // the user's pool
    uint key_mask;      // key slots - 1
    uint payload_stride;
};

uint64_t HashPayloadAddress(HashTable t, uint payload)
{
    return t.payloads + uint64_t(payload) * t.payload_stride;
}

uint64_t HashSlotAddress(HashTable t, uint slot)
{
    return t.keys + uint64_t(slot & t.key_mask) * kHashSlotBytes;
}

// One splitmix64 round (Vigna): the 64-bit finaliser over x + golden ratio.
uint64_t HashSplitMix64(uint64_t x)
{
    x += (uint64_t(0x9E3779B9u) << 32) | uint64_t(0x7F4A7C15u);
    x = (x ^ (x >> 30)) * ((uint64_t(0xBF58476Du) << 32) | uint64_t(0x1CE4E5B9u));
    x = (x ^ (x >> 27)) * ((uint64_t(0x94D049BBu) << 32) | uint64_t(0x133111EBu));
    return x ^ (x >> 31);
}

// The fingerprint of `count` key words (1 .. kHashMaxKeyWords): one splitmix64 round per pair of
// words (word 2i in the low half). key_lo >= 1 and key_hi != 0 are forced.
uint2 HashFingerprint(uint words[kHashMaxKeyWords], uint count)
{
    uint64_t h = 0;
    CAELIX_UNROLL
    for (uint i = 0; i < kHashMaxKeyWords; i += 2)
    {
        if (i < count)
        {
            const uint high = i + 1 < count ? words[i + 1] : 0u;
            h = HashSplitMix64(h ^ ((uint64_t(high) << 32) | uint64_t(words[i])));
        }
    }
    const uint lo = uint(h);
    const uint hi = uint(h >> 32);
    return uint2(max(lo, 1u), hi != 0u ? hi : 1u);
}

// Sets the payload's touched word to `frame` (written only when it differs).
void HashTouch(HashTable t, uint payload, uint frame)
{
    const uint64_t touched = HashPayloadAddress(t, payload) + kHashPayloadTouched;
    if (CAELIX_LOAD_U32(touched) != frame)
    {
        CAELIX_STORE_U32(touched, frame);
    }
}

// The insert of `key` into the empty slot that this thread just claimed (key_lo written): pops a
// payload, writes its header (state uninitialised, touched, the key words; nothing else) and the
// slot's payload word, then publishes key_hi. An empty free stack publishes kHashNone.
uint HashClaim(HashTable t, uint64_t slot, uint2 key, uint words[kHashMaxKeyWords], uint count, uint frame,
               out uint payload)
{
    // i = top - 1 before the decrement; i < 0: the stack is empty (top may go below 0; the
    // maintenance pass clamps it).
    const int index = int(CAELIX_ATOMIC_ADD_U32(t.stack + kHashStackTop, 0xFFFFFFFFu)) - 1;
    payload = index >= 0 ? CAELIX_LOAD_U32(t.stack + kHashStackEntries + uint64_t(index) * 4) : kHashNone;
    if (payload != kHashNone)
    {
        const uint64_t header = HashPayloadAddress(t, payload);
        CAELIX_ATOMIC_STORE_U32(header + kHashPayloadState, kHashStateUninitialised);
        CAELIX_STORE_U32(header + kHashPayloadTouched, frame);
        CAELIX_UNROLL
        for (uint w = 0; w < kHashMaxKeyWords; ++w)
        {
            if (w < count)
            {
                CAELIX_STORE_U32(header + kHashPayloadKey + w * 4, words[w]);
            }
        }
    }
    CAELIX_ATOMIC_STORE_U32(slot + kHashSlotPayload, payload);
    CAELIX_ATOMIC_STORE_RELEASE_U32(slot + kHashSlotKeyHi, key.y);
    return payload != kHashNone ? kHashInserted : kHashNoPayload;
}

// The probe loop of both lookups. `insert`: claim the first empty slot of the window (constant at
// every call site, so the compiler removes the branch).
uint HashFind(HashTable t, uint2 key, uint words[kHashMaxKeyWords], uint count, uint frame, bool insert,
              out uint payload)
{
    payload = kHashNone;
    CAELIX_LOOP
    for (uint probe = 0; probe < kHashProbes; ++probe)
    {
        const uint64_t slot = HashSlotAddress(t, key.x + probe);
        uint lo = CAELIX_ATOMIC_LOAD_U32(slot + kHashSlotKeyLo);
        if (lo == 0u)
        {
            if (!insert)
            {
                return kHashMissing;
            }
            // The compare-exchange returns the value it found: 0 = this thread owns the slot.
            lo = CAELIX_ATOMIC_CAS_U32(slot + kHashSlotKeyLo, 0u, key.x);
            if (lo == 0u)
            {
                return HashClaim(t, slot, key, words, count, frame, payload);
            }
            // Another thread claimed it: the same key_lo is handled below (pending or a hit).
        }
        if (lo == key.x)
        {
            const uint hi = CAELIX_ATOMIC_LOAD_ACQUIRE_U32(slot + kHashSlotKeyHi);
            if (hi == 0u)
            {
                return kHashPending;
            }
            if (hi == key.y)
            {
                payload = CAELIX_ATOMIC_LOAD_U32(slot + kHashSlotPayload);
                if (payload == kHashNone)
                {
                    return kHashEmptyEntry;
                }
                HashTouch(t, payload, frame);
                return kHashHit;
            }
        }
    }
    return insert ? kHashFullWindow : kHashMissing;
}

// Lookup without insert: kHashHit (touched), kHashPending, kHashEmptyEntry or kHashMissing.
uint HashLookup(HashTable t, uint2 key, uint frame, out uint payload)
{
    uint unused[kHashMaxKeyWords] = {0, 0, 0, 0, 0, 0, 0, 0};
    return HashFind(t, key, unused, 0, frame, false, payload);
}

// Lookup that inserts a missing key (`words`: its `count` key words, written into the new payload's
// header): kHashHit, kHashInserted, kHashPending, kHashFullWindow, kHashNoPayload or kHashEmptyEntry.
// The caller counts the result (HashCountResult).
uint HashLookupOrInsert(HashTable t, uint2 key, uint words[kHashMaxKeyWords], uint count, uint frame,
                        out uint payload)
{
    return HashFind(t, key, words, count, frame, true, payload);
}

// Adds `n` to a counter word (relaxed).
void HashCounterAdd(HashTable t, uint offset, uint n)
{
    if (n != 0u)
    {
        CAELIX_ATOMIC_ADD_U32(t.counters + offset, n);
    }
}

// The inserted / pending / failed counters of a lookup result, one atomic per counter and wave (the
// first active lane adds the wave's count). Call it from every lane that looked up.
void HashCountResult(HashTable t, uint result)
{
    const uint inserted = WaveActiveCountBits(result == kHashInserted);
    const uint pending = WaveActiveCountBits(result == kHashPending);
    const uint failed_slot = WaveActiveCountBits(result == kHashFullWindow);
    const uint failed_payload = WaveActiveCountBits(result == kHashNoPayload);
    if (WaveIsFirstLane())
    {
        HashCounterAdd(t, kHashCounterInserted, inserted);
        HashCounterAdd(t, kHashCounterPending, pending);
        HashCounterAdd(t, kHashCounterFailedSlot, failed_slot);
        HashCounterAdd(t, kHashCounterFailedPayload, failed_payload);
    }
}

// The insert of the maintenance pass's key rebuild: claims the first empty slot of the window and
// stores the payload and key_hi (no lookup runs in that dispatch, so the keys are known to be unique
// and a slot with the same key_lo is simply skipped). False when the window holds 8 other keys.
bool HashRebuildInsert(HashTable t, uint2 key, uint payload)
{
    CAELIX_LOOP
    for (uint probe = 0; probe < kHashProbes; ++probe)
    {
        const uint64_t slot = HashSlotAddress(t, key.x + probe);
        if (CAELIX_ATOMIC_CAS_U32(slot + kHashSlotKeyLo, 0u, key.x) == 0u)
        {
            CAELIX_ATOMIC_STORE_U32(slot + kHashSlotPayload, payload);
            CAELIX_ATOMIC_STORE_U32(slot + kHashSlotKeyHi, key.y);
            return true;
        }
    }
    return false;
}

// The key words of a payload's header.
void HashPayloadKeyWords(HashTable t, uint payload, uint count, out uint words[kHashMaxKeyWords])
{
    const uint64_t header = HashPayloadAddress(t, payload) + kHashPayloadKey;
    CAELIX_UNROLL
    for (uint w = 0; w < kHashMaxKeyWords; ++w)
    {
        words[w] = w < count ? CAELIX_LOAD_U32(header + w * 4) : 0u;
    }
}

#endif  // CAELIX_HASH_TABLE_HLSL
