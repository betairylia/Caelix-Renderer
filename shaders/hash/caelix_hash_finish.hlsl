// Hash table maintenance, kernel 3 of 3 (shaders/hash/caelix_hash_kernels.hlsl), one thread: the
// user pass's indirect dispatch for the active list, ceil(count / items_per_group) workgroups in rows
// of kHashDispatchRowGroups (x = the row width, or the count when it fits one row; y = the rows), and
// the active counter.
#include "hash/caelix_hash_kernels.hlsl"

[numthreads(1, 1, 1)]
void main()
{
    const uint count = CAELIX_LOAD_U32(g_push.dispatch + kHashDispatchCount);
    const uint per_group = max(g_push.items_per_group, 1u);
    const uint groups = (count + per_group - 1u) / per_group;
    const uint x = min(groups, kHashDispatchRowGroups);
    const uint y = (groups + kHashDispatchRowGroups - 1u) / kHashDispatchRowGroups;
    CAELIX_STORE_U32x4(g_push.dispatch, uint4(x, max(y, 1u), 1u, count));
    CAELIX_STORE_U32(g_push.counters + kHashCounterActive, count);
}
