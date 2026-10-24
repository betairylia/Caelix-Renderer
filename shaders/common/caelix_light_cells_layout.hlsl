// GPU layouts of the light cells (docs/LIGHT_NEE.md section 3), the payloads of the renderer's hash
// table (common/caelix_hash_table.hlsl). Mirrors lib/src/render/light_cells_layout.h: every constant
// here has the same name and value there (caelix_frame_layout_test compares the two files).
//
// One cell (LightCellBytes(R) bytes, 1088 at the default R = 64): the 64-byte header, then R reservoirs
// of 16 bytes (R = CaelixFrameConstants::light_cell_reservoirs, LightCellFillPush::reservoirs: 64, 128
// or 256).
//   0   state         kHashState* (the hash table's payload contract)
//   4   touched       the renderer's light frame of the newest lookup
//   8   entity        \
//   12  life           |
//   16  coord (int3)   |  the key words (kLightCellKeyWords)
//   28  level          |
//   32  face          /   the receivers' entity-local face
//   36  fills         fills since the insert
//   40  norm          C (float): running mean of the normalisation draws
//   44  norm_count    C_n
//   48  sun_tests     sun shadow rays traced from receivers of the cell
//   52  sun_visible   of those, the unoccluded ones
//   64  reservoirs    { uint group (kLightCellNoGroup = empty); uint entry; float W; uint generation }
#ifndef CAELIX_LIGHT_CELLS_LAYOUT_HLSL
#define CAELIX_LIGHT_CELLS_LAYOUT_HLSL

#include "common/caelix_hash_table.hlsl"

static const uint kLightCellDefaultReservoirs = 64;
static const uint kLightCellMinReservoirs = 64;
static const uint kLightCellMaxReservoirs = 256;
static const uint kLightCellReservoirBytes = 16;
static const uint kLcState = 0;
static const uint kLcTouched = 4;
static const uint kLcEntity = 8;
static const uint kLcLife = 12;
static const uint kLcCoord = 16;
static const uint kLcLevel = 28;
static const uint kLcFace = 32;
static const uint kLcFills = 36;
static const uint kLcNorm = 40;
static const uint kLcNormCount = 44;
static const uint kLcSunTests = 48;
static const uint kLcSunVisible = 52;
static const uint kLightCellHeaderBytes = 64;
static const uint kLightCellKeyWords = 7;
static const uint kLcResGroup = 0;
static const uint kLcResEntry = 4;
static const uint kLcResWeight = 8;
static const uint kLcResGeneration = 12;
static const uint kLightCellNoGroup = 0xFFFFFFFFu;
static const uint kLcReservoirs = 64;
static const uint kLightCellBaseShift = 5;
static const uint kLightCellLevelShift = 2;
static const uint kLightCellLevels = 2;
static const uint kLightCellBoxMarginVoxels = 1;
static const uint kLightCellNormalisationFills = 32;
static const uint kLightCellSunDecayShift = 3;
static const uint kLightCellRotation = 4;
static const uint kLightCellFillThreads = 64;
static const uint kLcCounterFreshDraws = kHashCounterUser + 0;
static const uint kLcCounterNormDraws = kHashCounterUser + 4;
static const uint kLcCounterStaleRefreshed = kHashCounterUser + 8;
static const uint kLightCellModeOff = 1;
static const uint kLightCellModeRotate = 2;

// The bytes of a cell with `reservoirs` reservoirs (the hash table's payload stride).
uint LightCellBytes(uint reservoirs)
{
    return kLightCellHeaderBytes + reservoirs * kLightCellReservoirBytes;
}

// The fill's thread mapping (lib/src/render/light_cells_layout.h has the same function): the threads
// of one cell, min(kLightCellFillThreads, the reservoirs due per fill).
uint LightCellFillThreadsPerCell(uint reservoirs, uint mode)
{
    const uint due = mode == kLightCellModeRotate ? reservoirs / kLightCellRotation : reservoirs;
    return min(kLightCellFillThreads, due);
}

#endif  // CAELIX_LIGHT_CELLS_LAYOUT_HLSL
