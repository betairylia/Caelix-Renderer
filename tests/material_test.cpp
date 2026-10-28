// The scene's material table (caelix_scene_set_materials, caelix_scene_set_glass_palette,
// caelix_scene_get_material), checked through the public C ABI:
// - packed: every opaque and emissive code equals the decode that the shader did
//   before step 6d (ported below); glass ids 1..132 equal data/glass-palette.json
//   (parsed here); air and the reserved ids are the vacuum entry; face bits of a
//   transparent code are ignored;
// - the glass palette can be replaced (packed only);
// - block-ID: the pastel hash default, a full host table round trip, the rest of
//   the table back to the default;
// - invalid entries and arguments are rejected and leave the table unchanged;
// - the subsurface and texture fields round-trip exactly and their ranges are checked;
// - the packing of the GPU entries (scene/material_pack.h, included directly).
// The GPU side of the table (upload, lookups) is checked by caelix_trace_test.
#include "test_scenes.h"

#include "scene/material_pack.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace test;

namespace {

constexpr uint32_t kEntries = 65536;

CaelixMaterial Get(Rig& rig, uint32_t code)
{
    CaelixMaterial m{};
    REQUIRE(caelix_scene_get_material(rig.scene, code, &m) == CAELIX_OK);
    return m;
}

bool Same(const CaelixMaterial& a, const CaelixMaterial& b)
{
    return std::memcmp(&a, &b, sizeof(CaelixMaterial)) == 0;
}

CaelixMaterial Make(float r, float g, float b, float smoothness, float metallic, float ior,
                    float extinction, float emission = 0.0f)
{
    CaelixMaterial m{};
    m.albedo[0] = r;
    m.albedo[1] = g;
    m.albedo[2] = b;
    m.smoothness = smoothness;
    m.emission[0] = m.emission[1] = m.emission[2] = emission;
    m.metallic = metallic;
    m.ior = ior;
    m.extinction = extinction;
    return m;
}

// The packed decode of shaders/render/caelix_shading.hlsl DecodeMaterial before
// step 6d (Unity's GetMaterial_Color), for opaque (bit 15) and emissive (bits
// 15..14 = 01) codes.
CaelixMaterial OldPackedDecode(uint32_t v)
{
    CaelixMaterial m{};
    m.ior = 1.5f;
    if ((v & 0x8000u) != 0)
    {
        m.albedo[0] = static_cast<float>((v >> 10) & 31u) / 31.0f;
        m.albedo[1] = static_cast<float>((v >> 5) & 31u) / 31.0f;
        m.albedo[2] = static_cast<float>(v & 31u) / 31.0f;
    }
    else
    {
        m.albedo[0] = static_cast<float>((v >> 10) & 15u) / 15.0f;
        m.albedo[1] = static_cast<float>((v >> 6) & 15u) / 15.0f;
        m.albedo[2] = static_cast<float>((v >> 2) & 15u) / 15.0f;
        const float strength = std::exp2(3.0f * static_cast<float>(v & 3u));
        for (int c = 0; c < 3; ++c)
        {
            m.emission[c] = m.albedo[c] * strength;
        }
    }
    return m;
}

// The pastel hash colour of the block-ID encoding (caelix_shading.hlsl HashU32, Pastel).
uint32_t HashU32(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

CaelixMaterial PastelOpaque(uint32_t hash)
{
    CaelixMaterial m{};
    m.albedo[0] = 0.35f + 0.6f * (static_cast<float>(hash & 0xFFu) / 255.0f);
    m.albedo[1] = 0.35f + 0.6f * (static_cast<float>((hash >> 8) & 0xFFu) / 255.0f);
    m.albedo[2] = 0.35f + 0.6f * (static_cast<float>((hash >> 16) & 0xFFu) / 255.0f);
    m.ior = 1.5f;
    return m;
}

// One entry of glass-palette.json.
struct GlassEntry
{
    int id = 0;
    float tint[3] = {};
    float ior = 0.0f;
    float extinction = 0.0f;
    float smoothness = 0.0f;
};

// The numbers after `"key":` in `object` (one, or the elements of an array).
std::vector<double> Numbers(const std::string& object, const char* key)
{
    std::vector<double> out;
    const std::string quoted = std::string("\"") + key + "\"";
    size_t at = object.find(quoted);
    REQUIRE(at != std::string::npos);
    at = object.find(':', at) + 1;
    while (at < object.size() && (object[at] == ' ' || object[at] == '\n' || object[at] == '\r' ||
                                  object[at] == '\t'))
    {
        ++at;
    }
    const bool array = object[at] == '[';
    const size_t end = array ? object.find(']', at) : object.find_first_of(",}", at);
    std::string text = object.substr(array ? at + 1 : at, end - (array ? at + 1 : at));
    for (char& ch : text)
    {
        if (ch == ',')
        {
            ch = ' ';
        }
    }
    std::istringstream stream(text);
    double value = 0.0;
    while (stream >> value)
    {
        out.push_back(value);
    }
    return out;
}

// A small parser for the known shape of glass-palette.json: {"notes": ..., "entries": [{...}, ...]}.
std::vector<GlassEntry> ReadPalette()
{
    std::ifstream file(CAELIX_GLASS_PALETTE_JSON, std::ios::binary);
    REQUIRE(file.good());
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    std::vector<GlassEntry> entries;
    size_t at = text.find("\"entries\"");
    REQUIRE(at != std::string::npos);
    while ((at = text.find('{', at)) != std::string::npos)
    {
        const size_t end = text.find('}', at);
        const std::string object = text.substr(at, end - at + 1);
        GlassEntry entry;
        entry.id = static_cast<int>(Numbers(object, "id").at(0));
        const std::vector<double> tint = Numbers(object, "tint");
        REQUIRE(tint.size() == 3);
        for (int c = 0; c < 3; ++c)
        {
            entry.tint[c] = static_cast<float>(tint[c]);
        }
        entry.ior = static_cast<float>(Numbers(object, "ior").at(0));
        entry.extinction = static_cast<float>(Numbers(object, "extinction").at(0));
        entry.smoothness = static_cast<float>(Numbers(object, "smoothness").at(0));
        entries.push_back(entry);
        at = end;
    }
    return entries;
}

void FlushOnce(Rig& rig)
{
    FlushFrame(rig.gpu, rig.scene, rig.frames);
}

}  // namespace

TEST_CASE("materials: packed table matches the old decode, the glass palette and the vacuum entry")
{
    Rig rig;
    OpenRig(rig, "caelix_material_test", RigDesc(0, 0, 0));

    // Opaque and emissive codes: bit for bit the decode of the shader before 6d.
    uint32_t mismatches = 0;
    for (uint32_t code = 0x4000; code < kEntries; ++code)
    {
        if (!Same(Get(rig, code), OldPackedDecode(code)))
        {
            if (mismatches < 4)
            {
                std::printf("packed code 0x%04X differs from the old decode\n", code);
            }
            ++mismatches;
        }
    }
    std::printf("packed opaque + emissive codes checked: %u, mismatches %u\n", kEntries - 0x4000u,
                mismatches);
    CHECK(mismatches == 0);
    // Spot checks of the decode itself.
    {
        const CaelixMaterial red = Get(rig, 0x8000u | (31u << 10));
        CHECK(red.albedo[0] == 1.0f);
        CHECK(red.albedo[1] == 0.0f);
        CHECK(red.smoothness == 0.0f);
        CHECK(red.metallic == 0.0f);
        CHECK(red.ior == 1.5f);
        const CaelixMaterial emitter = Get(rig, 0x4000u | (15u << 10) | 1u);  // rgb4 red, strength 1
        CHECK(emitter.albedo[0] == 1.0f);
        CHECK(emitter.emission[0] == 8.0f);
        CHECK(emitter.emission[1] == 0.0f);
    }

    // Glass ids 1..132 against the JSON; every face-bit variant gives the same entry.
    const std::vector<GlassEntry> palette = ReadPalette();
    REQUIRE(palette.size() == 132);
    uint32_t glass_mismatches = 0;
    for (const GlassEntry& entry : palette)
    {
        const CaelixMaterial m = Get(rig, static_cast<uint32_t>(entry.id));
        const CaelixMaterial expected = Make(entry.tint[0], entry.tint[1], entry.tint[2],
                                             entry.smoothness, 0.0f, entry.ior, entry.extinction);
        if (!Same(m, expected))
        {
            std::printf("glass id %d differs from the JSON\n", entry.id);
            ++glass_mismatches;
        }
        for (uint32_t faces = 1; faces < 64; faces += 7)
        {
            if (!Same(Get(rig, static_cast<uint32_t>(entry.id) | (faces << 8)), expected))
            {
                ++glass_mismatches;
            }
        }
    }
    std::printf("glass ids checked against the JSON: %zu, mismatches %u\n", palette.size(),
                glass_mismatches);
    CHECK(glass_mismatches == 0);
    {
        const CaelixMaterial amber7 = Get(rig, 35);
        CHECK(amber7.albedo[1] == 0.72f);
        CHECK(amber7.extinction == 0.032f);
        CHECK(amber7.ior == 1.55f);
    }

    // Air (id 0, with or without face bits) and the reserved ids: the vacuum entry.
    const CaelixMaterial vacuum = Make(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f);
    CHECK(Same(Get(rig, 0), vacuum));
    CHECK(Same(Get(rig, 0x3F00), vacuum));
    CHECK(Same(Get(rig, 133), vacuum));
    CHECK(Same(Get(rig, 255 | (5u << 8)), vacuum));

    // Stats and argument checks.
    CHECK(Stats(rig.scene).material_table_bytes == 65536ull * 64ull);
    ExpectLibraryErrors(1, [&] {
        CaelixMaterial m{};
        CHECK(caelix_scene_get_material(rig.scene, 0x10000u, &m) == CAELIX_ERROR_INVALID_ARGUMENT);
    });
    ExpectLibraryErrors(1, [&] {
        CHECK(caelix_scene_get_material(rig.scene, 1, nullptr) == CAELIX_ERROR_INVALID_ARGUMENT);
    });

    // The glass palette: ids 1 and 2 replaced, entry 0 ignored, id 3 unchanged.
    {
        const CaelixMaterial before3 = Get(rig, 3);
        CaelixMaterial entries[3] = {Make(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f),
                                     Make(0.5f, 0.25f, 0.125f, 1.0f, 0.0f, 1.4f, 0.01f),
                                     Make(0.1f, 0.2f, 0.3f, 0.5f, 0.0f, 1.2f, 0.0f)};
        entries[1].reserved = 7u;  // ignored, stored as 0
        REQUIRE(caelix_scene_set_glass_palette(rig.scene, entries, 3) == CAELIX_OK);
        entries[1].reserved = 0u;
        CHECK(Same(Get(rig, 1), entries[1]));
        CHECK(Same(Get(rig, 1 | (0x21u << 8)), entries[1]));
        CHECK(Same(Get(rig, 2), entries[2]));
        CHECK(Same(Get(rig, 3), before3));
        CHECK(Same(Get(rig, 0), vacuum));
        // An invalid entry: nothing changes.
        CaelixMaterial bad[2] = {vacuum, Make(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f)};  // ior 0
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_scene_set_glass_palette(rig.scene, bad, 2) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        CHECK(Same(Get(rig, 1), entries[1]));
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_scene_set_glass_palette(rig.scene, bad, 257) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
    }

    // A full host table replaces everything; count 0 restores the built-in table.
    {
        std::vector<CaelixMaterial> table(16, Make(0.25f, 0.5f, 0.75f, 0.3f, 0.6f, 1.33f, 0.002f, 2.0f));
        REQUIRE(caelix_scene_set_materials(rig.scene, table.data(), 16) == CAELIX_OK);
        CHECK(Same(Get(rig, 15), table[15]));
        CHECK(Same(Get(rig, 1 | (1u << 8)), table[1]));  // face bits removed before the lookup
        // Past count: the built-in default (glass id 16 from the JSON, an opaque decode).
        const GlassEntry& id16 = palette[15];
        REQUIRE(id16.id == 16);
        CHECK(Same(Get(rig, 16), Make(id16.tint[0], id16.tint[1], id16.tint[2], id16.smoothness, 0.0f,
                                      id16.ior, id16.extinction)));
        CHECK(Same(Get(rig, 0x8000u | 0x1234u), OldPackedDecode(0x8000u | 0x1234u)));
        // Count 0: the built-in table, whose transparent codes come from the CURRENT glass palette
        // (id 1 was replaced above, ior 1.4).
        REQUIRE(caelix_scene_set_materials(rig.scene, nullptr, 0) == CAELIX_OK);
        CHECK(Get(rig, 1).ior == 1.4f);
        CHECK(Get(rig, 3).ior == 1.33f);
        CHECK(Same(Get(rig, 0x8123), OldPackedDecode(0x8123)));
        table[3].extinction = -1.0f;
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_scene_set_materials(rig.scene, table.data(), 16) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        table[3].extinction = 0.0f;
        table[4].albedo[1] = std::numeric_limits<float>::quiet_NaN();
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_scene_set_materials(rig.scene, table.data(), 16) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        CHECK(Get(rig, 1).ior == 1.4f);  // unchanged
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_scene_set_materials(rig.scene, table.data(), kEntries + 1) ==
                  CAELIX_ERROR_INVALID_ARGUMENT);
        });
    }

    // The table reaches the GPU at a flush (the upload itself is checked by the trace test).
    FlushOnce(rig);
    FlushOnce(rig);
    CloseRig(rig);
}

TEST_CASE("materials: block-ID table: pastel default, host table round trip")
{
    Rig rig;
    OpenRig(rig, "caelix_material_test", RigDesc(1, 0, 0));

    // Defaults: opaque codes the pastel hash of code & 0x7FFF, transparent ids the pastel hash of
    // id | 0x10000 (face bits ignored), air the vacuum entry.
    uint32_t mismatches = 0;
    for (uint32_t code = 0x8000; code < kEntries; code += 37)
    {
        if (!Same(Get(rig, code), PastelOpaque(HashU32(code & 0x7FFFu))))
        {
            ++mismatches;
        }
    }
    for (uint32_t id = 1; id < 512; ++id)
    {
        if (!Same(Get(rig, id | (0x15u << 9)), PastelOpaque(HashU32(id | 0x10000u))))
        {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
    CHECK(Same(Get(rig, 0), Make(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f)));

    // The glass palette is a packed-encoding feature.
    ExpectLibraryErrors(1, [&] {
        const CaelixMaterial entry = Make(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.5f, 0.0f);
        CHECK(caelix_scene_set_glass_palette(rig.scene, &entry, 1) == CAELIX_ERROR_INVALID_ARGUMENT);
    });

    // A host table over codes [0, 0x8003): read every default, change three entries, write back.
    const uint32_t count = 0x8003;
    std::vector<CaelixMaterial> table(count);
    for (uint32_t code = 0; code < count; ++code)
    {
        table[code] = Get(rig, code);
    }
    table[0x8001] = Make(0.9f, 0.9f, 0.9f, 1.0f, 1.0f, 1.5f, 0.0f);        // smooth metal
    table[0x8002] = Make(1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.5f, 0.0f, 4.0f);  // emitter
    table[7] = Make(0.6f, 0.9f, 0.7f, 1.0f, 0.0f, 1.33f, 0.01f);           // transparent id 7
    REQUIRE(caelix_scene_set_materials(rig.scene, table.data(), count) == CAELIX_OK);
    CHECK(Same(Get(rig, 0x8001), table[0x8001]));
    CHECK(Same(Get(rig, 0x8002), table[0x8002]));
    CHECK(Same(Get(rig, 7), table[7]));
    CHECK(Same(Get(rig, 7 | (0x3Fu << 9)), table[7]));              // face bits removed before the lookup
    CHECK(Same(Get(rig, 0x8003), PastelOpaque(HashU32(0x0003u))));  // past count: default
    uint32_t round_trip_mismatches = 0;
    for (uint32_t code = 0; code < count; ++code)
    {
        if (!Same(Get(rig, code), table[code]) && (code & 0x8000u) != 0)
        {
            ++round_trip_mismatches;
        }
    }
    CHECK(round_trip_mismatches == 0);

    FlushOnce(rig);
    CloseRig(rig);
}

TEST_CASE("materials: sparse entries over the current table")
{
    Rig rig;
    OpenRig(rig, "caelix_material_test", RigDesc(1, 0, 0));

    // Two entries over the default table: an opaque code and transparent id 7.
    const CaelixMaterial untouched_default = Get(rig, 0x8006);
    CaelixMaterialEntry entries[2] = {};
    entries[0].code = 0x8005;
    entries[0].material = Make(0.9f, 0.1f, 0.2f, 0.5f, 1.0f, 1.5f, 0.0f);
    entries[1].code = 7;
    entries[1].material = Make(0.6f, 0.9f, 0.7f, 1.0f, 0.0f, 1.33f, 0.01f);
    REQUIRE(caelix_scene_set_material_entries(rig.scene, entries, 2) == CAELIX_OK);
    CHECK(Same(Get(rig, 0x8005), entries[0].material));
    CHECK(Same(Get(rig, 7), entries[1].material));
    CHECK(Same(Get(rig, 7 | (0x3Fu << 9)), entries[1].material));  // face bits removed before the lookup
    CHECK(Same(Get(rig, 0x8006), untouched_default));
    CHECK(Same(Get(rig, 0x8006), PastelOpaque(HashU32(0x0006u))));
    // count 0 is a no-op.
    REQUIRE(caelix_scene_set_material_entries(rig.scene, nullptr, 0) == CAELIX_OK);

    // An invalid batch changes nothing: one entry with ior 0.
    {
        CaelixMaterialEntry bad[2] = {};
        bad[0].code = 0x8005;
        bad[0].material = Make(0.1f, 0.1f, 0.1f, 0.0f, 0.0f, 1.5f, 0.0f);
        bad[1].code = 7;
        bad[1].material = Make(0.2f, 0.2f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f);
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_scene_set_material_entries(rig.scene, bad, 2) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        CHECK(Same(Get(rig, 0x8005), entries[0].material));
        CHECK(Same(Get(rig, 7), entries[1].material));
    }
    // A code above 0xFFFF is rejected the same way.
    {
        CaelixMaterialEntry bad[2] = {};
        bad[0].code = 0x8005;
        bad[0].material = Make(0.1f, 0.1f, 0.1f, 0.0f, 0.0f, 1.5f, 0.0f);
        bad[1].code = 0x10000;
        bad[1].material = Make(0.2f, 0.2f, 0.2f, 0.0f, 0.0f, 1.5f, 0.0f);
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_scene_set_material_entries(rig.scene, bad, 2) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        CHECK(Same(Get(rig, 0x8005), entries[0].material));
    }
    // count > 0 with null entries.
    ExpectLibraryErrors(1, [&] {
        CHECK(caelix_scene_set_material_entries(rig.scene, nullptr, 1) == CAELIX_ERROR_INVALID_ARGUMENT);
    });

    FlushOnce(rig);
    CloseRig(rig);
}

namespace {

// A material with every subsurface and texture field set.
CaelixMaterial FullMaterial()
{
    CaelixMaterial m = Make(0.3f, 0.6f, 0.9f, 0.25f, 0.5f, 1.45f, 0.02f, 3.0f);
    m.emission[1] = 1.5f;
    m.sss_model = 3u;
    m.sss_probability = 0.75f;
    m.sss_albedo[0] = 0.9f;
    m.sss_albedo[1] = 0.5f;
    m.sss_albedo[2] = 0.125f;
    m.sss_anisotropy = -0.4f;
    m.sss_steps = 200u;
    for (int i = 0; i < 8; ++i)
    {
        m.sss_params[i] = 0.5f * static_cast<float>(i) - 1.0f;
    }
    m.texture_set = 0xBEEFu;
    m.connect_group = 0x1234u;
    return m;
}

}  // namespace

TEST_CASE("materials: subsurface and texture fields round-trip; their ranges are checked")
{
    Rig rig;
    OpenRig(rig, "caelix_material_test", RigDesc(1, 0, 0));

    CaelixMaterialEntry entry{};
    entry.code = 0x8010;
    entry.material = FullMaterial();
    entry.material.reserved = 9u;  // ignored, stored as 0
    REQUIRE(caelix_scene_set_material_entries(rig.scene, &entry, 1) == CAELIX_OK);
    const CaelixMaterial expected = FullMaterial();
    CHECK(Same(Get(rig, 0x8010), expected));

    // Each invalid value is rejected and leaves the table unchanged.
    struct Bad
    {
        const char* what;
        void (*apply)(CaelixMaterial&);
    };
    const Bad bad[] = {
        {"sss_probability 1.5", [](CaelixMaterial& m) { m.sss_probability = 1.5f; }},
        {"sss_model 256", [](CaelixMaterial& m) { m.sss_model = 256u; }},
        {"sss_albedo[1] -0.1", [](CaelixMaterial& m) { m.sss_albedo[1] = -0.1f; }},
        {"sss_anisotropy 2", [](CaelixMaterial& m) { m.sss_anisotropy = 2.0f; }},
        {"sss_steps 300", [](CaelixMaterial& m) { m.sss_steps = 300u; }},
        {"texture_set 70000", [](CaelixMaterial& m) { m.texture_set = 70000u; }},
        {"connect_group 70000", [](CaelixMaterial& m) { m.connect_group = 70000u; }},
        {"emission[0] 70000", [](CaelixMaterial& m) { m.emission[0] = 70000.0f; }},
        {"sss_params[7] NaN", [](CaelixMaterial& m) { m.sss_params[7] = std::numeric_limits<float>::quiet_NaN(); }},
    };
    for (const Bad& b : bad)
    {
        INFO(b.what);
        CaelixMaterialEntry changed{};
        changed.code = 0x8010;
        changed.material = Make(0.1f, 0.1f, 0.1f, 0.0f, 0.0f, 1.5f, 0.0f);
        b.apply(changed.material);
        ExpectLibraryErrors(1, [&] {
            CHECK(caelix_scene_set_material_entries(rig.scene, &changed, 1) == CAELIX_ERROR_INVALID_ARGUMENT);
        });
        CHECK(Same(Get(rig, 0x8010), expected));
    }
    // The edges of the ranges are valid.
    {
        CaelixMaterialEntry edge{};
        edge.code = 0x8011;
        edge.material = Make(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.5f, 0.0f, 65504.0f);
        edge.material.sss_model = 255u;
        edge.material.sss_probability = 1.0f;
        edge.material.sss_albedo[0] = 1.0f;
        edge.material.sss_anisotropy = -1.0f;
        edge.material.sss_steps = 255u;
        edge.material.sss_params[0] = -65504.0f;
        edge.material.texture_set = 0xFFFFu;
        edge.material.connect_group = 0xFFFFu;
        REQUIRE(caelix_scene_set_material_entries(rig.scene, &edge, 1) == CAELIX_OK);
        CHECK(Same(Get(rig, 0x8011), edge.material));
    }

    FlushOnce(rig);
    CloseRig(rig);
}

TEST_CASE("materials: packing of the GPU hot and subsurface entries")
{
    using caelix::FloatToHalf;
    CHECK(FloatToHalf(1.0f) == 0x3C00);
    CHECK(FloatToHalf(0.5f) == 0x3800);
    CHECK(FloatToHalf(65504.0f) == 0x7BFF);
    CHECK(FloatToHalf(-2.0f) == 0xC000);
    CHECK(FloatToHalf(0.0f) == 0x0000);
    CHECK(FloatToHalf(-0.0f) == 0x8000);
    // Round to nearest even: 1 + 2^-11 is halfway between 0x3C00 and 0x3C01 -> 0x3C00 (even);
    // 1 + 3 * 2^-11 is halfway between 0x3C01 and 0x3C02 -> 0x3C02 (even).
    CHECK(FloatToHalf(1.0f + std::ldexp(1.0f, -11)) == 0x3C00);
    CHECK(FloatToHalf(1.0f + 3.0f * std::ldexp(1.0f, -11)) == 0x3C02);
    // Subnormal halves (units of 2^-24) and the clamps.
    CHECK(FloatToHalf(std::ldexp(1.0f, -24)) == 0x0001);
    CHECK(FloatToHalf(std::ldexp(1.0f, -25)) == 0x0000);
    CHECK(FloatToHalf(std::ldexp(3.0f, -25)) == 0x0002);
    CHECK(FloatToHalf(std::ldexp(1.0f, -14)) == 0x0400);
    CHECK(FloatToHalf(70000.0f) == 0x7BFF);
    CHECK(FloatToHalf(-70000.0f) == 0xFBFF);
    CHECK(FloatToHalf(std::numeric_limits<float>::quiet_NaN()) == 0x7E00);

    const CaelixMaterial m = FullMaterial();
    auto h2 = [](float low, float high) {
        return static_cast<uint32_t>(FloatToHalf(low)) | (static_cast<uint32_t>(FloatToHalf(high)) << 16);
    };
    uint32_t hot[8] = {};
    caelix::PackMaterialHot(m, hot);
    CHECK(hot[0] == h2(m.albedo[0], m.albedo[1]));
    CHECK(hot[1] == h2(m.albedo[2], m.smoothness));
    CHECK(hot[2] == h2(m.emission[0], m.emission[1]));
    CHECK(hot[3] == h2(m.emission[2], m.metallic));
    CHECK(hot[4] == h2(m.ior, m.extinction));
    CHECK((hot[5] & 0xFFFFu) == FloatToHalf(m.sss_probability));
    CHECK(((hot[5] >> 16) & 0xFFu) == m.sss_model);
    CHECK((hot[5] >> 24) == 0u);
    CHECK((hot[6] & 0xFFFFu) == m.texture_set);
    CHECK((hot[6] >> 16) == m.connect_group);
    CHECK(hot[7] == 0u);
    // The byte offsets of the layout constants.
    CHECK(caelix::kMaterialBytes == 32u);
    CHECK(caelix::kMaterialAlbedo == 0u);
    CHECK(caelix::kMaterialEmission == 2u * 4u);
    CHECK(caelix::kMaterialIor == 4u * 4u);
    CHECK(caelix::kMaterialSss == 5u * 4u);
    CHECK(caelix::kMaterialTexture == 6u * 4u);

    uint32_t sub[8] = {};
    caelix::PackMaterialSubsurface(m, sub);
    CHECK(sub[0] == h2(1.0f - m.sss_albedo[0], 1.0f - m.sss_albedo[1]));
    CHECK(sub[1] == h2(1.0f - m.sss_albedo[2], m.sss_anisotropy));
    // w2: the steps and the shadow length sqrt(steps (1 + g) / (1 - g)); w3: the mirror of hot w5.
    CHECK((sub[2] & 0xFFFFu) == m.sss_steps);
    CHECK((sub[2] >> 16) == FloatToHalf(std::sqrt(200.0f * (1.0f - 0.4f) / (1.0f + 0.4f))));
    CHECK(sub[3] == (hot[5] & 0x00FFFFFFu));
    CHECK((sub[3] & 0xFFFFu) == FloatToHalf(m.sss_probability));
    CHECK(((sub[3] >> 16) & 0xFFu) == m.sss_model);
    {
        // The shadow length: 16 steps at g = 0 is 4 voxels; g clamps to [-0.95, 0.95]; no steps, no length.
        CaelixMaterial s16 = m;
        s16.sss_steps = 16u;
        s16.sss_anisotropy = 0.0f;
        CHECK(caelix::SubsurfaceShadowLength(s16) == 4.0f);
        s16.sss_anisotropy = 1.0f;
        CHECK(caelix::SubsurfaceShadowLength(s16) == std::sqrt(16.0f * (1.0f + 0.95f) / (1.0f - 0.95f)));
        s16.sss_anisotropy = -1.0f;
        CHECK(caelix::SubsurfaceShadowLength(s16) == std::sqrt(16.0f * (1.0f + -0.95f) / (1.0f - -0.95f)));
        s16.sss_steps = 0u;
        CHECK(caelix::SubsurfaceShadowLength(s16) == 0.0f);
        uint32_t packed[8] = {};
        caelix::PackMaterialSubsurface(s16, packed);
        CHECK(packed[2] == 0u);
        s16.sss_steps = 16u;
        s16.sss_anisotropy = 0.0f;
        caelix::PackMaterialSubsurface(s16, packed);
        CHECK(packed[2] == (16u | (0x4400u << 16)));  // h(4) = 0x4400
    }
    for (int i = 0; i < 4; ++i)
    {
        CHECK(sub[4 + i] == h2(m.sss_params[2 * i], m.sss_params[2 * i + 1]));
    }
    CHECK(caelix::kSubsurfaceBytes == 32u);
    CHECK(caelix::kSubsurfaceAlbedo == 0u);
    CHECK(caelix::kSubsurfaceSteps == 2u * 4u);
    CHECK(caelix::kSubsurfaceProbability == 3u * 4u);
    CHECK(caelix::kSubsurfaceParams == 4u * 4u);
}
