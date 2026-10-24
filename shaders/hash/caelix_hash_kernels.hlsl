// Shared part of the hash table's maintenance kernels (docs/HASH_TABLE.md,
// lib/src/rhi/hash_table.h): the push constants (HashPush, lib/src/rhi/hash_table_layout.h; the
// member names are checked by caelix_frame_layout_test) and the table view.
//
// Per frame and table (HashTable::RecordMaintenance):
//   caelix_hash_begin     one thread: the free stack fix-up (top clamped to 0 after the pops of the
//                         previous lookups), the counters and the dispatch cleared;
//   (vkCmdFillBuffer)     the key table cleared;
//   caelix_hash_maintain  one thread per payload: eviction by age (push), the key rebuild, the active
//                         list;
//   caelix_hash_finish    one thread: the user pass's indirect dispatch.
// caelix_hash_init (one thread per payload, once at creation) fills the free stack 0 .. capacity - 1
// and marks every payload free.
#ifndef CAELIX_HASH_KERNELS_HLSL
#define CAELIX_HASH_KERNELS_HLSL

#include "common/caelix_hash_table.hlsl"

struct HashPush
{
    uint64_t keys;
    uint64_t stack;
    uint64_t counters;
    uint64_t payloads;
    uint64_t active;
    uint64_t dispatch;
    uint key_mask;
    uint payload_capacity;
    uint payload_stride;
    uint key_words;
    uint frame;
    uint max_age;
    uint items_per_group;
    uint pad;
};

CAELIX_PUSH_CONSTANTS(HashPush, g_push);

HashTable HashTableOfPush()
{
    HashTable t;
    t.keys = g_push.keys;
    t.stack = g_push.stack;
    t.counters = g_push.counters;
    t.payloads = g_push.payloads;
    t.key_mask = g_push.key_mask;
    t.payload_stride = g_push.payload_stride;
    return t;
}

#endif  // CAELIX_HASH_KERNELS_HLSL
