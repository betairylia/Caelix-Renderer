#pragma once
#include <cstddef>
#include <cstdint>

namespace caelix {

// GPU layouts of the reusable hash table (docs/HASH_TABLE.md; rhi/hash_table.h). The table
// maps 64-bit keys to the indices of a payload pool that its user owns; it knows nothing of what the
// payloads hold beyond the header contract below. shaders/common/caelix_hash_table.hlsl mirrors every
// constant of this file by name (caelix_frame_layout_test compares the two files, and the push
// constant block of the hash kernels with HashPush).

constexpr uint32_t kHashNone = 0xFFFFFFFFu;  // no payload
// Linear probing: a key lives in one of kHashProbes slots from its home slot (key_lo & mask).
constexpr uint32_t kHashProbes = 8;
// Key slots per payload of capacity: about 20 % key fill when the pool is 80 % full (1.2 probes per
// hit, < 0.1 % failed inserts with 8 probes; docs/HASH_TABLE.md section 3).
constexpr uint32_t kHashKeySlotsPerPayload = 4;

// ---------------------------------------------------------------------------
// Key slot. 16 bytes. key_lo 0 = empty; a real key has key_lo >= 1 and key_hi != 0, so key_lo set
// and key_hi 0 is an insert in progress (pending). Written only through atomics: the inserter's
// compare-exchange on key_lo, then the payload word, then a release store of key_hi.
struct HashKeySlot
{
    uint32_t key_lo;
    uint32_t key_hi;
    uint32_t payload;  // payload index; kHashNone = the insert found no free payload
    uint32_t pad;
};
constexpr uint32_t kHashSlotKeyLo = 0;
constexpr uint32_t kHashSlotKeyHi = 4;
constexpr uint32_t kHashSlotPayload = 8;
constexpr uint32_t kHashSlotBytes = 16;
static_assert(offsetof(HashKeySlot, key_lo) == kHashSlotKeyLo);
static_assert(offsetof(HashKeySlot, key_hi) == kHashSlotKeyHi);
static_assert(offsetof(HashKeySlot, payload) == kHashSlotPayload);
static_assert(sizeof(HashKeySlot) == kHashSlotBytes);

// ---------------------------------------------------------------------------
// Payload header contract: every payload of the user's pool starts with these words. The table
// writes state, touched and the key words when it inserts (never the rest of the payload), touches
// on every lookup, and the maintenance pass frees by age and recomputes the key from the key words.
constexpr uint32_t kHashPayloadState = 0;    // kHashState*
constexpr uint32_t kHashPayloadTouched = 4;  // the user's frame counter at the newest lookup
constexpr uint32_t kHashPayloadKey = 8;      // the user's key words (HashPush::key_words of them)
constexpr uint32_t kHashMaxKeyWords = 8;
constexpr uint32_t kHashStateFree = 0;           // in the free stack (a zero-filled pool is all free)
constexpr uint32_t kHashStateUninitialised = 1;  // inserted; the user's pass has not filled it yet
constexpr uint32_t kHashStateReady = 2;          // the user's own states are >= this

// ---------------------------------------------------------------------------
// Free stack: int top, then uint entries[capacity]. Pops (inserts) run in the user's lookup pass:
// i = atomicAdd(top, -1) - 1, i < 0 = empty (top may go below 0). Pushes (evictions) run in the
// maintenance pass after a one-thread fix-up clamped top to 0: entries[atomicAdd(top, 1)] = freed.
// Pops and pushes never run in the same dispatch.
constexpr uint32_t kHashStackTop = 0;
constexpr uint32_t kHashStackEntries = 4;
constexpr uint64_t HashStackBytes(uint32_t payload_capacity)
{
    return kHashStackEntries + uint64_t{payload_capacity} * sizeof(uint32_t);
}

// ---------------------------------------------------------------------------
// Counters (64 bytes): the table's words, then words for the user. Cleared by the first kernel of
// every maintenance pass; the library reads them back after the user's last pass of the frame, so a
// readback holds the maintenance counters of that frame and the lookup counters of its trace.
struct HashCounters
{
    uint32_t live;            // payloads alive after this maintenance (evictions and drops done)
    uint32_t active;          // of those, touched by the previous frame (the user's pass list)
    uint32_t inserted;        // lookups that inserted a key
    uint32_t pending;         // lookups that met an insert in progress (no entry this frame)
    uint32_t failed_slot;     // lookups whose 8-probe window was full (no entry this frame)
    uint32_t failed_payload;  // inserts that found the free stack empty (no entry until the rebuild)
    uint32_t evicted;         // payloads freed by age in this maintenance
    uint32_t rebuild_dropped; // payloads freed because their 8-probe window was full at the rebuild
    uint32_t user[8];         // the user's counters (cleared with the table's)
};
constexpr uint32_t kHashCounterLive = 0;
constexpr uint32_t kHashCounterActive = 4;
constexpr uint32_t kHashCounterInserted = 8;
constexpr uint32_t kHashCounterPending = 12;
constexpr uint32_t kHashCounterFailedSlot = 16;
constexpr uint32_t kHashCounterFailedPayload = 20;
constexpr uint32_t kHashCounterEvicted = 24;
constexpr uint32_t kHashCounterRebuildDropped = 28;
constexpr uint32_t kHashCounterUser = 32;
constexpr uint32_t kHashCounterUserWords = 8;
constexpr uint32_t kHashCounterBytes = 64;
static_assert(offsetof(HashCounters, live) == kHashCounterLive);
static_assert(offsetof(HashCounters, active) == kHashCounterActive);
static_assert(offsetof(HashCounters, inserted) == kHashCounterInserted);
static_assert(offsetof(HashCounters, pending) == kHashCounterPending);
static_assert(offsetof(HashCounters, failed_slot) == kHashCounterFailedSlot);
static_assert(offsetof(HashCounters, failed_payload) == kHashCounterFailedPayload);
static_assert(offsetof(HashCounters, evicted) == kHashCounterEvicted);
static_assert(offsetof(HashCounters, rebuild_dropped) == kHashCounterRebuildDropped);
static_assert(offsetof(HashCounters, user) == kHashCounterUser);
static_assert(sizeof(HashCounters::user) / sizeof(uint32_t) == kHashCounterUserWords);
static_assert(sizeof(HashCounters) == kHashCounterBytes);

// ---------------------------------------------------------------------------
// The user pass's indirect dispatch (16 bytes): VkDispatchIndirectCommand, then the active count.
// The active payloads are in the active list (uint per payload, the first `count` entries). The
// workgroups (count / items_per_group, rounded up) are laid out in rows of kHashDispatchRowGroups:
// the user's linear workgroup index is group.y * kHashDispatchRowGroups + group.x, and the user
// compares it with the count (the last row is full width).
struct HashDispatch
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t count;
};
constexpr uint32_t kHashDispatchX = 0;
constexpr uint32_t kHashDispatchY = 4;
constexpr uint32_t kHashDispatchZ = 8;
constexpr uint32_t kHashDispatchCount = 12;
constexpr uint32_t kHashDispatchBytes = 16;
constexpr uint32_t kHashDispatchRowGroups = 32768;  // <= maxComputeWorkGroupCount[0] (65535 minimum)
static_assert(offsetof(HashDispatch, x) == kHashDispatchX);
static_assert(offsetof(HashDispatch, y) == kHashDispatchY);
static_assert(offsetof(HashDispatch, z) == kHashDispatchZ);
static_assert(offsetof(HashDispatch, count) == kHashDispatchCount);
static_assert(sizeof(HashDispatch) == kHashDispatchBytes);

// Threads per workgroup of the per-payload kernels (caelix_hash_init, caelix_hash_maintain): one
// dispatch of capacity / kHashThreads workgroups, so the capacity is at most kHashMaxPayloads.
constexpr uint32_t kHashThreads = 64;
constexpr uint32_t kHashMaxPayloads = 1u << 21;
static_assert(kHashMaxPayloads / kHashThreads <= 65535);

// Results of a lookup in a kernel (HashLookup, HashLookupOrInsert); the library itself never sees
// them, the tests read them back.
constexpr uint32_t kHashHit = 0;         // present with a payload
constexpr uint32_t kHashInserted = 1;    // this lookup inserted the key
constexpr uint32_t kHashPending = 2;     // an insert of the key (or one with the same key_lo) is in progress
constexpr uint32_t kHashFullWindow = 3;  // the 8-probe window holds 8 other keys
constexpr uint32_t kHashNoPayload = 4;   // this lookup claimed a slot, the free stack was empty
constexpr uint32_t kHashEmptyEntry = 5;  // present without a payload (an earlier kHashNoPayload)
constexpr uint32_t kHashMissing = 6;     // lookup only: not in the table

// ---------------------------------------------------------------------------
// Push constants of the hash kernels (shaders/hash/caelix_hash_*.hlsl). 80 bytes. The HLSL struct
// has the same member names (caelix_frame_layout_test checks the compiled blocks against this one).
struct HashPush
{
    uint64_t keys;      // HashKeySlot[key_mask + 1]
    uint64_t stack;     // the free stack
    uint64_t counters;  // HashCounters
    uint64_t payloads;  // the user's pool: payload i at payloads + i * payload_stride
    uint64_t active;    // uint[payload_capacity]: the active list
    uint64_t dispatch;  // HashDispatch
    uint32_t key_mask;  // key slots - 1 (a power of two minus 1)
    uint32_t payload_capacity;
    uint32_t payload_stride;   // bytes per payload, a multiple of 4
    uint32_t key_words;        // 1 .. kHashMaxKeyWords key words from kHashPayloadKey
    uint32_t frame;            // the user's frame counter of this maintenance (>= 2)
    uint32_t max_age;          // a payload with frame - touched > max_age is freed
    uint32_t items_per_group;  // active payloads per workgroup of the user's pass (>= 1)
    uint32_t pad;
};
static_assert(offsetof(HashPush, key_mask) == 48);
static_assert(offsetof(HashPush, items_per_group) == 72);
static_assert(sizeof(HashPush) == 80);

}  // namespace caelix
