// Device API of the light tree (docs/LIGHT_TREE.md section 5): draws a light point for a receiver
// and evaluates its pdf. Includable by every kernel; it reads only the descriptor
// (caelix_scene_get_light_tree, FrameConstants::light_tree) and the buffers it points to.
//
// All positions and normals are world space. `n` is the receiver's unit normal, or 0 for a receiver
// without a surface (a volume point). `rng` is a PCG state owned by the caller; the functions
// consume a variable number of steps.
//
// Draw: one of the two world-space BVHs (static and dynamic entities) by the importance of their
// root summaries, then down the 4-wide nodes to a leaf = one light brick, then a face family, a
// face entry and a point on the face. The brick's light record is used in its entity's frame
// (x' = R^T (x - t)). At every choice a child is drawn with probability I_child / sum I
// (LightTreeImportance, one-pass weighted reservoir, one random number per child); when every
// child's importance is 0, with probability power_child / sum power instead, so a draw always
// reaches a light face and the pdfs of all light faces sum to 1 for every receiver. A node with one
// child is taken with probability 1. The pdf functions walk from the brick's leaf up to the root
// (parent links) and repeat the same importance computations, so a drawn sample's pdf equals the
// evaluated pdf up to rounding.
#ifndef CAELIX_LIGHT_TREE_HLSL
#define CAELIX_LIGHT_TREE_HLSL

#include "common/caelix_platform.hlsl"
#include "common/caelix_record.hlsl"
#include "common/caelix_light_tree_layout.hlsl"

// The descriptor address and the fields used on the hot path (the rest is loaded on use).
struct LightTree
{
    uint64_t descriptor;
    uint64_t heap;     // the current heap buffer
    uint64_t headers;  // tree header 0 (tree t at + t * kLtTreeHeaderBytes)
    uint group_shift;
};

// A light brick: its group, its index entry in the group's block (the brick's Morton rank) and the
// group's block generation when the reference was made. The reference stays valid while the group
// is not rebuilt (LightBrickRefValid).
struct LightBrickRef
{
    uint group;
    uint entry;
    uint generation;
};

struct LightPoint
{
    float3 position;  // on the light face (uniform on the unit square)
    float3 normal;    // the face's outward normal
    uint block;       // raw block value of the light voxel (material lookup by the caller)
    uint voxel;       // voxel index in the brick (x | y << 3 | z << 6)
    uint face;        // 0..5 (+X, -X, +Y, -Y, +Z, -Z)
    float pdf;        // area measure: P(brick | x, n) * P(face | brick, x, n) / face_count
    LightBrickRef brick;
};

// ---------------------------------------------------------------------------
// Random numbers (PCG, 32-bit state)

uint LightTreeRandom(inout uint rng)
{
    rng = rng * 747796405u + 2891336453u;
    const uint word = ((rng >> ((rng >> 28) + 4u)) ^ rng) * 277803737u;
    return (word >> 22) ^ word;
}

// Uniform in [0, 1).
float LightTreeRandomFloat(inout uint rng)
{
    return float(LightTreeRandom(rng) >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// Importance of a box of emitters (section 5.1), the one function behind sampling and pdf
// evaluation.

// The exact maximum of the emitter cosine of each face family over a box [lo, hi] of emitters.
void LightTreeEmitterCosines(float3 x, float3 lo, float3 hi, out float c[6])
{
    const float3 d_lo = x - hi;
    const float3 d_hi = x - lo;
    const float3 closest = clamp(float3(0.0f, 0.0f, 0.0f), d_lo, d_hi);
    const float3 closest2 = closest * closest;
    const float all2 = closest2.x + closest2.y + closest2.z;
    CAELIX_UNROLL
    for (uint a = 0; a < 3; ++a)
    {
        const float others = all2 - closest2[a];
        c[2 * a] = d_hi[a] > 0.0f ? d_hi[a] / sqrt(d_hi[a] * d_hi[a] + others) : 0.0f;
        c[2 * a + 1] = d_lo[a] < 0.0f ? -d_lo[a] / sqrt(d_lo[a] * d_lo[a] + others) : 0.0f;
    }
}

float LightTreeImportance(float3 x, float3 n, float3 lo, float3 hi, float face[6])
{
    float c[6];
    LightTreeEmitterCosines(x, lo, hi, c);
    float e = 0.0f;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        e += face[f] * c[f];
    }
    if (e <= 0.0f)
    {
        return 0.0f;
    }
    const float3 centre = 0.5f * (lo + hi);
    const float3 half_size = 0.5f * (hi - lo);
    float cr = 1.0f;
    if (any(n != 0.0f))
    {
        const float m = dot(n, centre - x) + dot(abs(n), half_size);
        if (m <= 0.0f)
        {
            return 0.0f;
        }
        const float dmin = length(max(max(lo - x, x - hi), float3(0.0f, 0.0f, 0.0f)));
        cr = dmin > 0.0f ? min(1.0f, m / dmin) : 1.0f;
    }
    const float3 to_centre = x - centre;
    const float d2 = max(dot(to_centre, to_centre), max(1.0f, dot(half_size, half_size)));
    return e * cr / d2;
}

// The running draw over a node's children: a weighted reservoir by importance and one by power
// (used when every importance is 0), fed with the same random number per child.
struct LtPick
{
    float sum_i;
    float sum_p;
    float weight_i;
    float weight_p;
    uint index_i;
    uint index_p;
};

LtPick LtPickBegin()
{
    LtPick p;
    p.sum_i = 0.0f;
    p.sum_p = 0.0f;
    p.weight_i = 0.0f;
    p.weight_p = 0.0f;
    p.index_i = kLtNone;
    p.index_p = kLtNone;
    return p;
}

void LtPickAdd(inout LtPick p, float importance, float power, uint index, float u)
{
    if (importance > 0.0f)
    {
        p.sum_i += importance;
        if (u * p.sum_i < importance)
        {
            p.weight_i = importance;
            p.index_i = index;
        }
    }
    if (power > 0.0f)
    {
        p.sum_p += power;
        if (u * p.sum_p < power)
        {
            p.weight_p = power;
            p.index_p = index;
        }
    }
}

// The drawn child and its probability; kLtNone when no child has power.
uint LtPickEnd(LtPick p, out float probability)
{
    if (p.sum_i > 0.0f)
    {
        probability = p.weight_i / p.sum_i;
        return p.index_i;
    }
    probability = p.sum_p > 0.0f ? p.weight_p / p.sum_p : 0.0f;
    return p.sum_p > 0.0f ? p.index_p : kLtNone;
}

// The probability of a child with the weights (importance, power) among children with the sums.
float LtPickProbability(float sum_i, float sum_p, float importance, float power)
{
    if (sum_i > 0.0f)
    {
        return importance / sum_i;
    }
    return sum_p > 0.0f ? power / sum_p : 0.0f;
}

// ---------------------------------------------------------------------------
// Tables

bool LightTreeOpen(uint64_t descriptor, out LightTree tree)
{
    tree.descriptor = descriptor;
    tree.heap = 0;
    tree.headers = 0;
    tree.group_shift = 0;
    if (descriptor == 0)
    {
        return false;
    }
    tree.group_shift = LtDescriptorU32(descriptor, kLtDescGroupShift);
    tree.headers = LtDescriptorU64(descriptor, kLtDescTreeHeaders);
    const uint current = CAELIX_LOAD_U32(LtDescriptorU64(descriptor, kLtDescControl) + kLtCtlCurrent) & 1u;
    tree.heap = LtDescriptorU64(descriptor, current != 0 ? kLtDescHeap1 : kLtDescHeap0);
    float power = 0.0f;
    for (uint k = 0; k < kLtTrees; ++k)
    {
        const uint64_t header = tree.headers + uint64_t(k) * kLtTreeHeaderBytes;
        if (CAELIX_LOAD_U32(header + kLtHdrRoot) != kLtNone)
        {
            power += LtPower(LtLoadSummary(header).face);
        }
    }
    return power > 0.0f;
}

// The root summary of tree k (power 0 for an empty tree).
LtSummary LtTreeSummary(LightTree t, uint k, out uint root)
{
    const uint64_t header = t.headers + uint64_t(k) * kLtTreeHeaderBytes;
    root = CAELIX_LOAD_U32(header + kLtHdrRoot);
    if (root == kLtNone)
    {
        return LtEmptySummary();
    }
    return LtLoadSummary(header);
}

uint64_t LtNodeAddress(LightTree t, uint k, uint node)
{
    const uint64_t nodes = LtDescriptorU64(t.descriptor, kLtDescNodes0 + kLtDescTreeStride * k);
    return nodes + uint64_t(node) * kLtNodeBytes;
}

uint64_t LtLeafAddress(LightTree t, uint k, uint leaf)
{
    const uint64_t leaves = LtDescriptorU64(t.descriptor, kLtDescLeaves0 + kLtDescTreeStride * k);
    return leaves + uint64_t(leaf) * kLtLeafBytes;
}

// Slot s of a 4-wide node: its inline summary.
LtSummary LtNodeSlot(uint64_t node, uint s)
{
    LtSummary r;
    const uint64_t faces = node + kLtNodeChildFace + kLtNodeFaceBytes * s;
    const uint2 f01 = CAELIX_LOAD_U32x2(faces);
    const uint2 f23 = CAELIX_LOAD_U32x2(faces + 8);
    const uint2 f45 = CAELIX_LOAD_U32x2(faces + 16);
    r.face[0] = asfloat(f01.x);
    r.face[1] = asfloat(f01.y);
    r.face[2] = asfloat(f23.x);
    r.face[3] = asfloat(f23.y);
    r.face[4] = asfloat(f45.x);
    r.face[5] = asfloat(f45.y);
    const uint64_t lo = node + kLtNodeChildLo + kLtNodeBoxBytes * s;
    const uint64_t hi = node + kLtNodeChildHi + kLtNodeBoxBytes * s;
    r.lo = asfloat(uint3(CAELIX_LOAD_U32(lo), CAELIX_LOAD_U32(lo + 4), CAELIX_LOAD_U32(lo + 8)));
    r.hi = asfloat(uint3(CAELIX_LOAD_U32(hi), CAELIX_LOAD_U32(hi + 4), CAELIX_LOAD_U32(hi + 8)));
    return r;
}

// The group's row words block_offset, block_units, generation, gpu_flags.
uint4 LtGroupRow(LightTree t, uint group)
{
    const uint64_t rows = LtDescriptorU64(t.descriptor, kLtDescGroupRows);
    return CAELIX_LOAD_U32x4(rows + uint64_t(group) * kLtGroupRowBytes);
}

// The group's entity-local voxel origin, and the frame of its entity.
float3 LtGroupFrame(LightTree t, uint group, out LtFrame frame)
{
    uint entity;
    const float3 origin =
        LtGroupOrigin(LtDescriptorU64(t.descriptor, kLtDescGroupInfos), group, t.group_shift, entity);
    frame = LtEntityFrame(LtDescriptorU64(t.descriptor, kLtDescEntityInfos), entity);
    return origin;
}

// Face family of a record: P(f) = F[f] * cmax(f) / sum over the record box (or F[f] / sum F when
// every such product is 0), in the group-local frame.
void LtFamilyWeights(LtRecord r, float3 x, out float w[6], out float sum)
{
    float c[6];
    LightTreeEmitterCosines(x, float3(r.lo), float3(r.hi), c);
    sum = 0.0f;
    CAELIX_UNROLL
    for (uint f = 0; f < 6; ++f)
    {
        w[f] = r.face[f] * c[f];
        sum += w[f];
    }
    if (sum <= 0.0f)
    {
        sum = 0.0f;
        CAELIX_UNROLL
        for (uint g = 0; g < 6; ++g)
        {
            w[g] = r.face[g];
            sum += w[g];
        }
    }
}

// ---------------------------------------------------------------------------
// The BVHs

// Draws a tree (kLtNone when both are empty) and multiplies pdf by its probability.
uint LtChooseTree(LightTree t, float3 x, float3 n, inout uint rng, inout float pdf, out uint root)
{
    LtPick pick = LtPickBegin();
    uint roots[2];
    for (uint k = 0; k < kLtTrees; ++k)
    {
        const LtSummary s = LtTreeSummary(t, k, roots[k]);
        LtPickAdd(pick, LightTreeImportance(x, n, s.lo, s.hi, s.face), LtPower(s.face), k,
                  LightTreeRandomFloat(rng));
    }
    float probability;
    const uint k = LtPickEnd(pick, probability);
    root = k != kLtNone ? roots[k] : kLtNone;
    if (k != kLtNone)
    {
        pdf *= probability;
    }
    return k;
}

float LtTreeProbability(LightTree t, float3 x, float3 n, uint target)
{
    float sum_i = 0.0f;
    float sum_p = 0.0f;
    float importance_t = 0.0f;
    float power_t = 0.0f;
    for (uint k = 0; k < kLtTrees; ++k)
    {
        uint root;
        const LtSummary s = LtTreeSummary(t, k, root);
        const float importance = LightTreeImportance(x, n, s.lo, s.hi, s.face);
        const float power = LtPower(s.face);
        sum_i += importance > 0.0f ? importance : 0.0f;
        sum_p += power > 0.0f ? power : 0.0f;
        if (k == target)
        {
            importance_t = importance;
            power_t = power;
        }
    }
    return LtPickProbability(sum_i, sum_p, importance_t, power_t);
}

// Draws a leaf of tree k from its root. kLtNone when it fails.
uint LtChooseLeaf(LightTree t, uint k, uint root, float3 x, float3 n, inout uint rng, inout float pdf)
{
    uint node = root;
    for (uint depth = 0; depth < 64; ++depth)
    {
        const uint64_t address = LtNodeAddress(t, k, node);
        const uint4 children = CAELIX_LOAD_U32x4(address + kLtNodeChild);
        uint chosen = kLtNone;
        if (LtNodeChildCount(children) == 1)
        {
            chosen = children.x;  // the build fills the slots from 0
        }
        else
        {
            LtPick pick = LtPickBegin();
            for (uint s = 0; s < kLtNodeSlots; ++s)
            {
                if (children[s] == kLtNone)
                {
                    continue;
                }
                const LtSummary slot = LtNodeSlot(address, s);
                LtPickAdd(pick, LightTreeImportance(x, n, slot.lo, slot.hi, slot.face), LtPower(slot.face), s,
                          LightTreeRandomFloat(rng));
            }
            float probability;
            const uint s = LtPickEnd(pick, probability);
            if (s == kLtNone)
            {
                return kLtNone;
            }
            pdf *= probability;
            chosen = children[s];
        }
        if ((chosen & kLtChildLeafBit) != 0)
        {
            return chosen & ~kLtChildLeafBit;
        }
        node = chosen;
    }
    return kLtNone;
}

// The probability of slot `slot` of a node.
float LtSlotProbability(uint64_t address, float3 x, float3 n, uint slot)
{
    const uint4 children = CAELIX_LOAD_U32x4(address + kLtNodeChild);
    if (LtNodeChildCount(children) == 1)
    {
        return 1.0f;
    }
    float sum_i = 0.0f;
    float sum_p = 0.0f;
    float importance_t = 0.0f;
    float power_t = 0.0f;
    for (uint s = 0; s < kLtNodeSlots; ++s)
    {
        if (children[s] == kLtNone)
        {
            continue;
        }
        const LtSummary summary = LtNodeSlot(address, s);
        const float importance = LightTreeImportance(x, n, summary.lo, summary.hi, summary.face);
        const float power = LtPower(summary.face);
        sum_i += importance > 0.0f ? importance : 0.0f;
        sum_p += power > 0.0f ? power : 0.0f;
        if (s == slot)
        {
            importance_t = importance;
            power_t = power;
        }
    }
    return LtPickProbability(sum_i, sum_p, importance_t, power_t);
}

// The probability of reaching `leaf` of tree k (walk up the parent links), 0 for an empty tree.
float LtLeafProbability(LightTree t, uint k, uint leaf, float3 x, float3 n)
{
    uint root;
    LtTreeSummary(t, k, root);
    const uint parent = CAELIX_LOAD_U32(LtLeafAddress(t, k, leaf) + kLtLeafParent);
    if (root == kLtNone || parent == kLtNone)
    {
        return 0.0f;
    }
    uint node = parent >> kLtParentSlotBits;
    uint slot = parent & kLtParentSlotMask;
    float probability = 1.0f;
    for (uint depth = 0; depth < 64; ++depth)
    {
        const uint64_t address = LtNodeAddress(t, k, node);
        probability *= LtSlotProbability(address, x, n, slot);
        const uint2 up = CAELIX_LOAD_U32x2(address + kLtNodeParent);  // parent, parent_slot
        if (up.x == kLtNone)
        {
            return probability * LtTreeProbability(t, x, n, k);
        }
        node = up.x;
        slot = up.y;
    }
    return 0.0f;
}

// The probability of the brick at index entry `entry` of the group's block, through its leaf; 0 when
// it has none.
float LtBrickProbability(LightTree t, float3 x, float3 n, uint64_t block, uint group, uint entry)
{
    const uint bvh_leaf =
        CAELIX_LOAD_U32(block + kLtBlockIndexOffset + uint64_t(entry) * kLtIndexEntryBytes + kLtIndexBvhLeaf);
    if (bvh_leaf == kLtNone)
    {
        return 0.0f;
    }
    const uint k = bvh_leaf >> kLtBvhLeafTreeShift;
    const uint leaf = bvh_leaf & kLtBvhLeafIndexMask;
    const uint2 owner = CAELIX_LOAD_U32x2(LtLeafAddress(t, k, leaf) + kLtLeafGroup);  // group, entry
    if (owner.x != group || owner.y != entry)
    {
        return 0.0f;
    }
    return LtLeafProbability(t, k, leaf, x, n);
}

// ---------------------------------------------------------------------------
// Public functions

// The group-local voxel coordinate of the brick's corner (voxel (0, 0, 0) of the brick). Part of the
// API for NEE code: with LightPoint::voxel it gives the light voxel in its group.
uint3 LightBrickVoxelOrigin(LightTree t, LightBrickRef brick)
{
    const uint4 row = LtGroupRow(t, brick.group);
    const uint pos = CAELIX_LOAD_U32(t.heap + uint64_t(row.x) * kLtUnitBytes + kLtBlockIndexOffset +
                                     uint64_t(brick.entry) * kLtIndexEntryBytes + kLtIndexPos);
    return BrickOfPos(pos, t.group_shift) * kBrickEdge;
}

bool LightBrickRefValid(LightTree t, LightBrickRef brick)
{
    const uint4 row = LtGroupRow(t, brick.group);
    return row.y != 0 && row.z == brick.generation &&
           brick.entry < CAELIX_LOAD_U32(t.heap + uint64_t(row.x) * kLtUnitBytes);  // record_count
}

bool LightTreeSampleBrick(LightTree t, float3 x, float3 n, inout uint rng, out LightBrickRef brick,
                          out float pdf_brick)
{
    brick.group = kLtNone;
    brick.entry = kLtNone;
    brick.generation = 0;
    pdf_brick = 1.0f;
    uint root;
    const uint k = LtChooseTree(t, x, n, rng, pdf_brick, root);
    if (k == kLtNone)
    {
        return false;
    }
    const uint leaf = LtChooseLeaf(t, k, root, x, n, rng, pdf_brick);
    if (leaf == kLtNone)
    {
        return false;
    }
    const uint2 owner = CAELIX_LOAD_U32x2(LtLeafAddress(t, k, leaf) + kLtLeafGroup);  // group, entry
    brick.group = owner.x;
    brick.entry = owner.y;
    brick.generation = LtGroupRow(t, owner.x).z;
    return true;
}

bool LightBrickSamplePoint(LightTree t, LightBrickRef brick, float3 x, float3 n, inout uint rng,
                           out LightPoint p)
{
    p.position = float3(0, 0, 0);
    p.normal = float3(0, 0, 0);
    p.block = 0;
    p.voxel = 0;
    p.face = 0;
    p.pdf = 0.0f;
    p.brick = brick;
    const uint4 row = LtGroupRow(t, brick.group);
    if (row.y == 0)
    {
        return false;
    }
    const uint64_t block = t.heap + uint64_t(row.x) * kLtUnitBytes;
    const LtRecord record = LtLoadEntryRecord(block, brick.entry);
    LtFrame frame;
    const float3 origin = LtGroupFrame(t, brick.group, frame);
    const float3 x_group = mul(x - frame.translation, frame.rotation) - origin;  // R^T (x - t) - origin
    float w[6];
    float sum;
    LtFamilyWeights(record, x_group, w, sum);
    if (sum <= 0.0f)
    {
        return false;
    }
    // Family by its CDF, one random number (rounding past the end takes the last family with weight).
    const float u = LightTreeRandomFloat(rng) * sum;
    uint f = kLtNone;
    uint last = 0;
    float cumulative = 0.0f;
    for (uint g = 0; g < 6; ++g)
    {
        if (w[g] > 0.0f)
        {
            last = g;
            cumulative += w[g];
            if (f == kLtNone && u < cumulative)
            {
                f = g;
            }
        }
    }
    f = f == kLtNone ? last : f;
    uint entry_first = 0;
    for (uint h = 0; h < f; ++h)
    {
        entry_first += LtFamilyCount(record, h);
    }
    const uint count = LtFamilyCount(record, f);
    if (count == 0 || w[f] <= 0.0f)
    {
        return false;
    }
    const uint entry_index = min(uint(LightTreeRandomFloat(rng) * float(count)), count - 1u);
    const uint entry =
        CAELIX_LOAD_U32(record.entries + uint64_t(entry_first + entry_index) * kLtFaceEntryBytes);
    const uint vi = entry & kLtFaceEntryVoxelMask;
    const uint3 voxel =
        BrickOfPos(record.pos, t.group_shift) * kBrickEdge + uint3(vi & 7, (vi >> 3) & 7, vi >> 6);
    const uint axis = f >> 1;
    float3 local = origin + float3(voxel);
    const float a = LightTreeRandomFloat(rng);
    const float b = LightTreeRandomFloat(rng);
    local[axis] += (f & 1) != 0 ? 0.0f : 1.0f;
    local[(axis + 1) % 3] += a;
    local[(axis + 2) % 3] += b;
    p.position = mul(frame.rotation, local) + frame.translation;
    p.normal = mul(frame.rotation, float3(FaceDirection(f)));
    p.block = entry >> kLtFaceEntryBlockShift;
    p.voxel = vi;
    p.face = f;
    p.pdf = w[f] / sum / float(count);
    return true;
}

bool LightTreeSample(LightTree t, float3 x, float3 n, inout uint rng, out LightPoint p)
{
    LightBrickRef brick;
    float pdf_brick;
    if (!LightTreeSampleBrick(t, x, n, rng, brick, pdf_brick))
    {
        p.position = float3(0, 0, 0);
        p.normal = float3(0, 0, 0);
        p.block = 0;
        p.voxel = 0;
        p.face = 0;
        p.pdf = 0.0f;
        p.brick = brick;
        return false;
    }
    const bool ok = LightBrickSamplePoint(t, brick, x, n, rng, p);
    p.pdf *= pdf_brick;
    return ok;
}

float LightTreeBrickPdf(LightTree t, float3 x, float3 n, LightBrickRef brick)
{
    if (!LightBrickRefValid(t, brick))
    {
        return 0.0f;
    }
    const uint4 row = LtGroupRow(t, brick.group);
    return LtBrickProbability(t, x, n, t.heap + uint64_t(row.x) * kLtUnitBytes, brick.group, brick.entry);
}

float LightTreePointPdf(LightTree t, float3 x, float3 n, uint group, uint3 voxel_in_group, uint face,
                        uint block_value)
{
    // Not a light voxel: 0 without a lookup.
    const uint encoding = LtDescriptorU32(t.descriptor, kLtDescEncoding);
    const uint code = BlockMaterialCode(block_value & 0xFFFFu, encoding);
    const uint64_t light_codes = LtDescriptorU64(t.descriptor, kLtDescLightCodes);
    const uint code_word = CAELIX_LOAD_U32(light_codes + uint64_t(code >> 5) * 4);
    if (((code_word >> (code & 31)) & 1u) == 0 || face > 5)
    {
        return 0.0f;
    }
    const uint4 row = LtGroupRow(t, group);
    if (row.y == 0)
    {
        return 0.0f;
    }
    const uint64_t block = t.heap + uint64_t(row.x) * kLtUnitBytes;
    const uint entry_index =
        LtFindIndexEntry(block, CAELIX_LOAD_U32(block), LtMorton(voxel_in_group / kBrickEdge), t.group_shift);
    if (entry_index == kLtNone)
    {
        return 0.0f;
    }
    const float brick_pdf = LtBrickProbability(t, x, n, block, group, entry_index);
    if (brick_pdf <= 0.0f)
    {
        return 0.0f;
    }
    const LtRecord record = LtLoadEntryRecord(block, entry_index);
    LtFrame frame;
    const float3 origin = LtGroupFrame(t, group, frame);
    const float3 x_group = mul(x - frame.translation, frame.rotation) - origin;
    float w[6];
    float sum;
    LtFamilyWeights(record, x_group, w, sum);
    const uint count = LtFamilyCount(record, face);
    if (sum <= 0.0f || w[face] <= 0.0f || count == 0)
    {
        return 0.0f;
    }
    // The family's entries are sorted by voxel index: binary search.
    uint first = 0;
    for (uint g = 0; g < face; ++g)
    {
        first += LtFamilyCount(record, g);
    }
    const uint3 v = voxel_in_group % kBrickEdge;
    const uint vi = v.x | (v.y << 3) | (v.z << 6);
    const uint64_t entries = record.entries + uint64_t(first) * kLtFaceEntryBytes;
    uint a = 0;
    uint b = count;
    bool found = false;
    while (a < b)
    {
        const uint mid = (a + b) >> 1;
        const uint entry_voxel =
            CAELIX_LOAD_U32(entries + uint64_t(mid) * kLtFaceEntryBytes) & kLtFaceEntryVoxelMask;
        if (entry_voxel == vi)
        {
            found = true;
            break;
        }
        if (entry_voxel < vi)
        {
            a = mid + 1;
        }
        else
        {
            b = mid;
        }
    }
    return found ? brick_pdf * (w[face] / sum) / float(count) : 0.0f;
}

#endif  // CAELIX_LIGHT_TREE_HLSL
