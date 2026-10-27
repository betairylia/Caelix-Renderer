// Record scatter kernel: copies the records that caelix_scene_upload_bricks wrote into the staging
// ring to their record slots (Scene::RecordScatter in lib/src/scene/scene_flush.cpp). It replaces one
// vkCmdCopyBuffer region per record: the transfer engine handles regions one after another at about
// 1 us each, so 18k records cost ~17 ms that way.
//
// The source is the ring itself (ReBAR path: the ring is device-local, host-visible memory) or the
// device-local bounce buffer that one bulk copy filled at the ring's offsets (bounce path). The entry
// list (one block of RecordScatterEntry, common/caelix_record.hlsl) lies in the same source.
//
// Workgroup g copies the records of entries [g * kScatterRecordsPerGroup, ... + kScatterRecordsPerGroup)
// as one flat array of 8-byte words: thread t takes the words t, t + threads, t + 2 * threads, ... It
// issues all its loads first and its stores after them, so many loads are in flight at once. Word 0
// of a record holds u32 words 0 and 1; with kScatterKeepBoundsWord only u32 word 0 is stored and the
// destination's word 1 (kRecordBoundsWord) stays. Source and destination are 8-byte aligned (records
// are 8-byte aligned in the ring and record_bytes is a multiple of 8).
//
// The kernel reads only the list and the source and writes only record pages.
#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"

struct RecordScatterPush
{
    uint64_t list;       // RecordScatterEntry[count]
    uint64_t source;     // base of src_offset8: the ring or the bounce buffer
    uint count;          // entries
    uint record_words8;  // record_bytes / kRecordScatterSourceUnit
};

CAELIX_PUSH_CONSTANTS(RecordScatterPush, g_push);

// The kernel shape (tuning knobs; the C++ dispatch uses the same constants, gpu_structs.h).
#define CAELIX_SCATTER_RECORDS kScatterRecordsPerGroup
#define CAELIX_SCATTER_THREADS kScatterThreadsPerGroup
// 8-byte words per thread that cover the group's records at the largest record size.
#define CAELIX_SCATTER_WORDS \
    ((CAELIX_SCATTER_RECORDS * kScatterMaxRecordWords8 + CAELIX_SCATTER_THREADS - 1) / CAELIX_SCATTER_THREADS)

groupshared uint4 gs_entries[CAELIX_SCATTER_RECORDS];

[numthreads(CAELIX_SCATTER_THREADS, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID)
{
    const uint first = group_id.x * CAELIX_SCATTER_RECORDS;
    const uint t = thread_id.x;
    if (t < CAELIX_SCATTER_RECORDS)
    {
        const uint e = first + t;
        gs_entries[t] = e < g_push.count
            ? CAELIX_LOAD_U32x4(g_push.list + uint64_t(e) * kRecordScatterEntryBytes)
            : uint4(0, 0, 0, 0);
    }
    GroupMemoryBarrierWithGroupSync();

    // The dispatch has ceil(count / records) groups, so first < count.
    const uint records = min(uint(CAELIX_SCATTER_RECORDS), g_push.count - first);
    const uint words8 = g_push.record_words8;
    const uint total = records * words8;

    uint2 v[CAELIX_SCATTER_WORDS];
    CAELIX_UNROLL
    for (uint k = 0; k < CAELIX_SCATTER_WORDS; ++k)
    {
        const uint i = t + CAELIX_SCATTER_THREADS * k;
        v[k] = uint2(0, 0);
        if (i < total)
        {
            const uint r = i / words8;
            const uint w = i - r * words8;
            v[k] = CAELIX_LOAD_U32x2(g_push.source + uint64_t(gs_entries[r].x) * kRecordScatterSourceUnit +
                                     uint64_t(w) * kRecordScatterSourceUnit);
        }
    }
    CAELIX_UNROLL
    for (uint k = 0; k < CAELIX_SCATTER_WORDS; ++k)
    {
        const uint i = t + CAELIX_SCATTER_THREADS * k;
        if (i < total)
        {
            const uint r = i / words8;
            const uint w = i - r * words8;
            const uint4 entry = gs_entries[r];
            const uint64_t dst = (uint64_t(entry.z) | (uint64_t(entry.w) << 32)) +
                                 uint64_t(w) * kRecordScatterSourceUnit;
            if (w == 0 && (entry.y & kScatterKeepBoundsWord) != 0)
            {
                CAELIX_STORE_U32(dst, v[k].x);
            }
            else
            {
                CAELIX_STORE_U32x2(dst, v[k]);
            }
        }
    }
}
