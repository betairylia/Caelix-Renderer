// Light BVH build: the binary radix tree over the sorted leaves (Karras 2012, "Maximizing
// Parallelism in the Construction of BVHs, Octrees, and k-d Trees"), one thread per internal node
// i in [0, n - 1). Equal Morton codes are ordered by their sorted position (the key is
// (code, position)), so every key is distinct. Writes left/right (bit 31 = a leaf at that sorted
// position) and the parent of every child; node 0 is the root.
#include "scene/caelix_light_bvh_common.hlsl"

uint LtCountLeadingZeros(uint x)
{
    return x == 0 ? 32u : 31u - firstbithigh(x);
}

uint LtSortedKey(uint i)
{
    return CAELIX_LOAD_U32(LtKeys(0) + uint64_t(i) * 4);
}

// Length of the common prefix of the keys at sorted positions i and j; -1 outside [0, n).
int LtDelta(int i, int j, int n, uint key_i)
{
    if (j < 0 || j >= n)
    {
        return -1;
    }
    const uint key_j = LtSortedKey(uint(j));
    if (key_i == key_j)
    {
        return 32 + int(LtCountLeadingZeros(uint(i) ^ uint(j)));
    }
    return int(LtCountLeadingZeros(key_i ^ key_j));
}

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint index = dispatch_id.x;
    if (index + 1 >= g_push.leaf_capacity || !LtBuilding())
    {
        return;
    }
    const int n = int(LtLeafCount());
    const int i = int(index);
    if (i >= n - 1)
    {
        return;
    }
    if (i == 0)
    {
        LtStoreScratch(kLtScratchParent, 0, kLtNone);
    }
    const uint key_i = LtSortedKey(uint(i));
    // Direction of the range and its other end.
    const int d = LtDelta(i, i + 1, n, key_i) - LtDelta(i, i - 1, n, key_i) >= 0 ? 1 : -1;
    const int delta_min = LtDelta(i, i - d, n, key_i);
    int length_max = 2;
    while (LtDelta(i, i + length_max * d, n, key_i) > delta_min)
    {
        length_max *= 2;
    }
    int length = 0;
    for (int step = length_max / 2; step >= 1; step /= 2)
    {
        if (LtDelta(i, i + (length + step) * d, n, key_i) > delta_min)
        {
            length += step;
        }
    }
    const int j = i + length * d;
    // The split: the highest position in the range that shares more than delta_node bits with i.
    const int delta_node = LtDelta(i, j, n, key_i);
    int split = 0;
    int divisor = 2;
    int step2 = (length + divisor - 1) / divisor;
    while (true)
    {
        if (LtDelta(i, i + (split + step2) * d, n, key_i) > delta_node)
        {
            split += step2;
        }
        if (step2 == 1)
        {
            break;
        }
        divisor *= 2;
        step2 = (length + divisor - 1) / divisor;
    }
    const int gamma = i + split * d + min(d, 0);
    const uint left = min(i, j) == gamma ? (kLtChildLeafBit | uint(gamma)) : uint(gamma);
    const uint right = max(i, j) == gamma + 1 ? (kLtChildLeafBit | uint(gamma + 1)) : uint(gamma + 1);
    LtStoreScratch(kLtScratchLeft, uint(i), left);
    LtStoreScratch(kLtScratchRight, uint(i), right);
    LtStoreScratch((left & kLtChildLeafBit) != 0 ? kLtScratchSortedLeafParent : kLtScratchParent,
                   left & ~kLtChildLeafBit, uint(i));
    LtStoreScratch((right & kLtChildLeafBit) != 0 ? kLtScratchSortedLeafParent : kLtScratchParent,
                   right & ~kLtChildLeafBit, uint(i));
}
