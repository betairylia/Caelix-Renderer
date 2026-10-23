// Light BVH, first kernel of a tree's gather and build (one thread): decides whether this flush
// rebuilds the tree. It does when the CPU asks (an entity of the tree moved, an entity changed its
// motion or was removed, capacities grew) or when this flush's group pass committed with a rebuilt or
// released group of this tree (its leaves change; the other tree's leaves name (group, index entry)
// pairs that the pass did not change, since its blocks were only copied). Every later kernel of the
// sequence reads the decision.
#include "scene/caelix_light_bvh_common.hlsl"

[numthreads(1, 1, 1)]
void main()
{
    const uint64_t control = LtControl();
    const bool committed = g_push.pass_recorded != 0 &&
                           CAELIX_LOAD_U32(control + kLtCtlPassResult) == kLtPassOk &&
                           ((CAELIX_LOAD_U32(control + kLtCtlTreesChanged) >> g_push.tree) & 1u) != 0;
    CAELIX_STORE_U32(LtTreeState() + kLtStBuilding, (g_push.reason != 0 || committed) ? 1u : 0u);
}
