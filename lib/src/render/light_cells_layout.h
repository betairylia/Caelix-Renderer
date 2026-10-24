#pragma once
#include "rhi/hash_table_layout.h"

#include <cstddef>
#include <cstdint>

namespace caelix {

// GPU layouts of the light cells (docs/LIGHT_NEE.md section 3): the payloads of the renderer's hash
// table (rhi/hash_table.h). shaders/common/caelix_light_cells_layout.hlsl mirrors every constant of
// this file by name (caelix_frame_layout_test compares the two files, and the push constant block of
// caelix_light_cell_fill with LightCellFillPush).

// ---------------------------------------------------------------------------
// One cell: the 64-byte header, then R reservoirs, R = CaelixLightNeeOptions::reservoirs (64, 128 or 256;
// FrameConstants::light_cell_reservoirs, LightCellFillPush::reservoirs): LightCellBytes(R) bytes, 1088
// at the default R = 64.
constexpr uint32_t kLightCellDefaultReservoirs = 64;
constexpr uint32_t kLightCellMinReservoirs = 64;
constexpr uint32_t kLightCellMaxReservoirs = 256;
constexpr uint32_t kLightCellReservoirBytes = 16;

// The header. The first words are the hash table's payload contract (state, touched, the key words);
// the key is (entity, entity life, coord, level, face): kLightCellKeyWords words from kLcEntity.
struct LightCellHeader
{
    uint32_t state;        // kHashState* (free, uninitialised, ready)
    uint32_t touched;      // the renderer's light frame of the newest lookup
    uint32_t entity;       // the receivers' entity (cells are entity-local)
    uint32_t life;         // its EntityInfo::life when the cell was inserted
    int32_t coord[3];      // floor(p) >> (kLightCellBaseShift + kLightCellLevelShift * level), entity-local
    uint32_t level;        // 0: 32-voxel cells, 1: 128-voxel cells
    uint32_t face;         // the receivers' entity-local face (+X 0, -X 1, +Y 2, -Y 3, +Z 4, -Z 5)
    uint32_t fills;        // fills since the insert (the rotate mode's turn and the normalisation draw)
    float norm;            // C: running mean of the normalisation draws (the MIS proxy's normaliser)
    uint32_t norm_count;   // C_n: draws behind C, at most kLightCellNormalisationFills
    uint32_t sun_tests;    // sun shadow rays traced from receivers of the cell (decayed by the fill)
    uint32_t sun_visible;  // of those, the unoccluded ones
    uint32_t pad[2];
};
constexpr uint32_t kLcState = 0;
constexpr uint32_t kLcTouched = 4;
constexpr uint32_t kLcEntity = 8;
constexpr uint32_t kLcLife = 12;
constexpr uint32_t kLcCoord = 16;
constexpr uint32_t kLcLevel = 28;
constexpr uint32_t kLcFace = 32;
constexpr uint32_t kLcFills = 36;
constexpr uint32_t kLcNorm = 40;
constexpr uint32_t kLcNormCount = 44;
constexpr uint32_t kLcSunTests = 48;
constexpr uint32_t kLcSunVisible = 52;
constexpr uint32_t kLightCellHeaderBytes = 64;
constexpr uint32_t kLightCellKeyWords = 7;  // entity, life, coord x, y, z, level, face
static_assert(offsetof(LightCellHeader, state) == kLcState && kLcState == kHashPayloadState);
static_assert(offsetof(LightCellHeader, touched) == kLcTouched && kLcTouched == kHashPayloadTouched);
static_assert(offsetof(LightCellHeader, entity) == kLcEntity && kLcEntity == kHashPayloadKey);
static_assert(offsetof(LightCellHeader, life) == kLcLife);
static_assert(offsetof(LightCellHeader, coord) == kLcCoord);
static_assert(offsetof(LightCellHeader, level) == kLcLevel);
static_assert(offsetof(LightCellHeader, face) == kLcFace);
static_assert(kLcFace + 4 == kLcEntity + 4 * kLightCellKeyWords, "the key words are contiguous");
static_assert(offsetof(LightCellHeader, fills) == kLcFills);
static_assert(offsetof(LightCellHeader, norm) == kLcNorm);
static_assert(offsetof(LightCellHeader, norm_count) == kLcNormCount);
static_assert(offsetof(LightCellHeader, sun_tests) == kLcSunTests);
static_assert(offsetof(LightCellHeader, sun_visible) == kLcSunVisible);
static_assert(sizeof(LightCellHeader) == kLightCellHeaderBytes);
static_assert(kLightCellKeyWords <= kHashMaxKeyWords);

// A reservoir: one aligned 16-byte load. group = kLightCellNoGroup: empty.
struct LightCellReservoir
{
    uint32_t group;       // the light brick's group (LightBrickRef)
    uint32_t entry;       // its index entry
    float weight;         // W: the unbiased contribution weight of the brick for every receiver of the cell
    uint32_t generation;  // the group's GroupLightRow::generation when the brick was drawn
};
constexpr uint32_t kLcResGroup = 0;
constexpr uint32_t kLcResEntry = 4;
constexpr uint32_t kLcResWeight = 8;
constexpr uint32_t kLcResGeneration = 12;
constexpr uint32_t kLightCellNoGroup = 0xFFFFFFFFu;
static_assert(offsetof(LightCellReservoir, group) == kLcResGroup);
static_assert(offsetof(LightCellReservoir, entry) == kLcResEntry);
static_assert(offsetof(LightCellReservoir, weight) == kLcResWeight);
static_assert(offsetof(LightCellReservoir, generation) == kLcResGeneration);
static_assert(sizeof(LightCellReservoir) == kLightCellReservoirBytes);

constexpr uint32_t kLcReservoirs = kLightCellHeaderBytes;  // byte offset of reservoir 0
// The bytes of a cell with `reservoirs` reservoirs (the hash table's payload stride).
constexpr uint32_t LightCellBytes(uint32_t reservoirs)
{
    return kLightCellHeaderBytes + reservoirs * kLightCellReservoirBytes;
}
static_assert(LightCellBytes(kLightCellDefaultReservoirs) == 1088);

// ---------------------------------------------------------------------------
// Cell geometry: level L cells are 1 << (kLightCellBaseShift + kLightCellLevelShift L) voxels on an
// edge (32 and 128).
constexpr uint32_t kLightCellBaseShift = 5;
constexpr uint32_t kLightCellLevelShift = 2;
constexpr uint32_t kLightCellLevels = 2;
// The receiver box of a cell: the cell inflated by half a cell (the jittered lookups) plus this many
// voxels (the cell point lies a quarter voxel off the surface, the receiver 0.002 off it).
constexpr uint32_t kLightCellBoxMarginVoxels = 1;
// The MIS proxy's normalisation C: a running mean over at most this many fills.
constexpr uint32_t kLightCellNormalisationFills = 32;
// The sun hint counters lose 1 / 2^kLightCellSunDecayShift per fill.
constexpr uint32_t kLightCellSunDecayShift = 3;
// Reservoir mode "rotate": reservoir r is refreshed by the fills with (r + fills) % this == 0.
constexpr uint32_t kLightCellRotation = 4;
// The fill: workgroups of kLightCellFillThreads threads (LightCellFillThreadsPerCell below).
constexpr uint32_t kLightCellFillThreads = 64;
static_assert(kLightCellFillThreads == kLightCellMinReservoirs);

// The user counters of the hash table (HashCounters::user words), written by the fill.
constexpr uint32_t kLcCounterFreshDraws = kHashCounterUser + 0;     // fresh candidates drawn
constexpr uint32_t kLcCounterNormDraws = kHashCounterUser + 4;      // normalisation draws
constexpr uint32_t kLcCounterStaleRefreshed = kHashCounterUser + 8; // reservoirs refreshed early (stale)

// The fill's reservoir modes (LightCellFillPush::mode; CaelixLightNeeOptions::reservoir_mode).
constexpr uint32_t kLightCellModeOff = 1;     // every reservoir refreshed every fill
constexpr uint32_t kLightCellModeRotate = 2;  // a rotating quarter refreshed every fill

// The fill's thread mapping (docs/LIGHT_NEE.md section 11.6; caelix_light_cell_fill.hlsl has the same
// function): the threads of one cell, min(kLightCellFillThreads, the reservoirs due per fill: R in mode
// "off", R / kLightCellRotation in mode "rotate"). A workgroup holds kLightCellFillThreads / that many
// cells; thread j of a cell owns the R / that many reservoirs from j x that, so in mode "rotate" a
// thread owns kLightCellRotation consecutive reservoirs of which exactly one is due, and the waves
// carry only reservoirs that draw. R = 64: "off" one cell per workgroup, thread r = reservoir r;
// "rotate" 4 cells of 16 threads with 4 reservoirs each.
constexpr uint32_t LightCellFillThreadsPerCell(uint32_t reservoirs, uint32_t mode)
{
    const uint32_t due = mode == kLightCellModeRotate ? reservoirs / kLightCellRotation : reservoirs;
    return due < kLightCellFillThreads ? due : kLightCellFillThreads;
}
static_assert(LightCellFillThreadsPerCell(64, kLightCellModeOff) == 64);
static_assert(LightCellFillThreadsPerCell(64, kLightCellModeRotate) == 16);
static_assert(LightCellFillThreadsPerCell(128, kLightCellModeRotate) == 32);
static_assert(LightCellFillThreadsPerCell(256, kLightCellModeRotate) == 64);
static_assert(LightCellFillThreadsPerCell(256, kLightCellModeOff) == 64);
static_assert(kLightCellMinReservoirs / kLightCellRotation >= 1 &&
              kLightCellFillThreads % (kLightCellMinReservoirs / kLightCellRotation) == 0,
              "every mapping divides the workgroup into whole cells");

// ---------------------------------------------------------------------------
// Push constants of caelix_light_cell_fill (shaders/render/caelix_light_cell_fill.hlsl). 56 bytes.
struct LightCellFillPush
{
    uint64_t payloads;    // the cells
    uint64_t active;      // the hash table's active list
    uint64_t dispatch;    // the hash table's dispatch (its count)
    uint64_t counters;    // the hash table's counters (the user words)
    uint64_t light_tree;  // the scene's light tree descriptor
    uint32_t frame_seed;  // CaelixRenderOptions::frame_seed of the frame
    uint32_t mode;        // kLightCellMode*
    uint32_t candidates;  // M: 1 (a plain draw, W = 1 / p) or 2 (streaming RIS by the leaf importance)
    uint32_t reservoirs;  // R: reservoirs per cell (64, 128 or 256; LightCellBytes(R) per cell)
};
static_assert(offsetof(LightCellFillPush, frame_seed) == 40);
static_assert(sizeof(LightCellFillPush) == 56);

}  // namespace caelix
