// The GPU layouts that C++ and HLSL share, checked without a GPU:
//   - the uniform block of the frame constants in the compiled kernels against the C layout of
//     lib/src/render/frame_constants.h (below);
//   - the push constant blocks of the scene and light tree kernels against their C++ structs, member
//     by member and by name;
//   - every named constant of the HLSL layout headers (caelix_record.hlsl, caelix_material.hlsl,
//     caelix_light_tree_layout.hlsl) against the C++ constant of the same name.
// The test reads the .spv files of the build (CAELIX_SHADER_SPV_DIR, written by cmake/Shaders.cmake)
// and parses the SPIR-V binary, and reads the HLSL sources (CAELIX_SHADER_SOURCE_DIR).
//
// For each of the frame-constant kernels (the six kernels and the trace variant) it finds the
// variable of storage class Uniform whose pointee is a struct decorated Block (exactly one), checks
// its DescriptorSet / Binding against kFrameSetIndex / kFrameConstantsBinding, recomputes the C
// natural layout of the block's member sequence (each member aligned to its scalar size; arrays and
// matrices without padding) and requires every member's Offset decoration to equal that offset,
// every ArrayStride / MatrixStride to equal the element / column size, and the end of the last member
// to equal sizeof(FrameConstants) (whose field offsets the static_asserts of frame_constants.h pin).
// DXC's constant buffer layout differs from the C layout when a vector crosses a 16-byte row or an
// array element is not 16 bytes; such a change of CaelixFrameConstants fails here.
#include "render/frame_constants.h"
#include "rhi/hash_table_layout.h"
#include "scene/gpu_structs.h"
#include "light/light_tree_layout.h"
#include "scene/material_table.h"
#include "scene/record_layout.h"

#include <catch2/catch_test_macros.hpp>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef CAELIX_SHADER_SPV_DIR
#error "CAELIX_SHADER_SPV_DIR must name the build's shader folder"
#endif
#ifndef CAELIX_SHADER_SOURCE_DIR
#error "CAELIX_SHADER_SOURCE_DIR must name the shader source folder"
#endif

namespace {

// SPIR-V opcodes, decorations and storage classes used here (SPIR-V 1.6 specification).
constexpr uint32_t kOpMemberName = 6;
constexpr uint32_t kOpTypeInt = 21;
constexpr uint32_t kOpTypeFloat = 22;
constexpr uint32_t kOpTypeVector = 23;
constexpr uint32_t kOpTypeMatrix = 24;
constexpr uint32_t kOpTypeArray = 28;
constexpr uint32_t kOpTypeStruct = 30;
constexpr uint32_t kOpTypePointer = 32;
constexpr uint32_t kOpConstant = 43;
constexpr uint32_t kOpVariable = 59;
constexpr uint32_t kOpDecorate = 71;
constexpr uint32_t kOpMemberDecorate = 72;
constexpr uint32_t kDecorationBlock = 2;
constexpr uint32_t kDecorationArrayStride = 6;
constexpr uint32_t kDecorationMatrixStride = 7;
constexpr uint32_t kDecorationBinding = 33;
constexpr uint32_t kDecorationDescriptorSet = 34;
constexpr uint32_t kDecorationOffset = 35;
constexpr uint32_t kStorageClassUniform = 2;
constexpr uint32_t kStorageClassPushConstant = 9;
constexpr uint32_t kSpirvMagic = 0x07230203u;

struct Type
{
    uint32_t opcode = 0;
    std::vector<uint32_t> operands;  // the words after the result id
};

struct Module
{
    std::map<uint32_t, Type> types;                                   // result id -> type
    std::map<uint32_t, uint32_t> constants;                           // result id -> first literal word
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> decorations;    // (id, decoration) -> first literal
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> member_offsets; // (struct, member) -> Offset
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> member_matrix_strides;
    std::map<std::pair<uint32_t, uint32_t>, std::string> member_names;
    std::vector<std::pair<uint32_t, uint32_t>> uniform_variables;     // (variable id, pointer type id)
    std::vector<std::pair<uint32_t, uint32_t>> push_variables;        // (variable id, pointer type id)
};

std::vector<uint32_t> ReadWords(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        return {};
    }
    const std::streamsize bytes = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    return words;
}

// The SPIR-V literal string in the `count` words at `words` (nul-terminated, 4 bytes per word).
std::string LiteralString(const uint32_t* words, uint32_t count)
{
    std::string text;
    for (uint32_t i = 0; i < count; ++i)
    {
        for (uint32_t b = 0; b < 4; ++b)
        {
            const char c = static_cast<char>((words[i] >> (8 * b)) & 0xFFu);
            if (c == '\0')
            {
                return text;
            }
            text += c;
        }
    }
    return text;
}

// Fills `module`; returns an error text or an empty string.
std::string Parse(const std::vector<uint32_t>& words, Module& module)
{
    if (words.size() < 5 || words[0] != kSpirvMagic)
    {
        return "not a SPIR-V module";
    }
    size_t at = 5;
    while (at < words.size())
    {
        const uint32_t count = words[at] >> 16;
        const uint32_t opcode = words[at] & 0xFFFFu;
        if (count == 0 || at + count > words.size())
        {
            return "truncated instruction";
        }
        const uint32_t* w = &words[at];
        switch (opcode)
        {
        case kOpTypeInt:
        case kOpTypeFloat:
        case kOpTypeVector:
        case kOpTypeMatrix:
        case kOpTypeArray:
        case kOpTypeStruct:
        case kOpTypePointer:
            module.types[w[1]] = Type{opcode, std::vector<uint32_t>(w + 2, w + count)};
            break;
        case kOpConstant:
            if (count >= 4)
            {
                module.constants[w[2]] = w[3];
            }
            break;
        case kOpVariable:
            if (count >= 4 && w[3] == kStorageClassUniform)
            {
                module.uniform_variables.emplace_back(w[2], w[1]);
            }
            if (count >= 4 && w[3] == kStorageClassPushConstant)
            {
                module.push_variables.emplace_back(w[2], w[1]);
            }
            break;
        case kOpDecorate:
            if (count >= 3)
            {
                module.decorations[{w[1], w[2]}] = count >= 4 ? w[3] : 0u;
            }
            break;
        case kOpMemberDecorate:
            if (count >= 5 && w[3] == kDecorationOffset)
            {
                module.member_offsets[{w[1], w[2]}] = w[4];
            }
            else if (count >= 5 && w[3] == kDecorationMatrixStride)
            {
                module.member_matrix_strides[{w[1], w[2]}] = w[4];
            }
            break;
        case kOpMemberName:
            if (count >= 4)
            {
                module.member_names[{w[1], w[2]}] = LiteralString(w + 3, count - 3);
            }
            break;
        default:
            break;
        }
        at += count;
    }
    return {};
}

// The C natural size and alignment of a member type; `error` gets a text when the type is not
// supported or a stride differs from the natural element size.
struct Layout
{
    uint32_t size = 0;
    uint32_t align = 1;
};

Layout NaturalLayout(const Module& module, uint32_t type_id, std::optional<uint32_t> matrix_stride,
                     std::string& error)
{
    const auto found = module.types.find(type_id);
    if (found == module.types.end())
    {
        error = "unknown type id " + std::to_string(type_id);
        return {};
    }
    const Type& type = found->second;
    switch (type.opcode)
    {
    case kOpTypeInt:
    case kOpTypeFloat:
    {
        const uint32_t bytes = type.operands[0] / 8;
        return {bytes, bytes};
    }
    case kOpTypeVector:
    {
        const Layout component = NaturalLayout(module, type.operands[0], std::nullopt, error);
        return {component.size * type.operands[1], component.align};
    }
    case kOpTypeMatrix:
    {
        const Layout column = NaturalLayout(module, type.operands[0], std::nullopt, error);
        if (!matrix_stride || *matrix_stride != column.size)
        {
            error = "MatrixStride " + (matrix_stride ? std::to_string(*matrix_stride) : std::string("missing")) +
                    " != column size " + std::to_string(column.size);
        }
        return {column.size * type.operands[1], column.align};
    }
    case kOpTypeArray:
    {
        const Layout element = NaturalLayout(module, type.operands[0], matrix_stride, error);
        const auto length = module.constants.find(type.operands[1]);
        if (length == module.constants.end())
        {
            error = "array length is not an OpConstant";
            return {};
        }
        const auto stride = module.decorations.find({type_id, kDecorationArrayStride});
        if (stride == module.decorations.end() || stride->second != element.size)
        {
            error = "ArrayStride " +
                    (stride == module.decorations.end() ? std::string("missing") : std::to_string(stride->second)) +
                    " != element size " + std::to_string(element.size);
        }
        return {element.size * length->second, element.align};
    }
    default:
        error = "unsupported member type (opcode " + std::to_string(type.opcode) + ")";
        return {};
    }
}

// Checks the frame constant block of one kernel; returns the error texts (empty = it matches).
std::vector<std::string> CheckKernel(const std::string& name)
{
    const std::string path = std::string(CAELIX_SHADER_SPV_DIR) + "/" + name + ".spv";
    const std::vector<uint32_t> words = ReadWords(path);
    if (words.empty())
    {
        return {"cannot read " + path};
    }
    Module module;
    const std::string parse_error = Parse(words, module);
    if (!parse_error.empty())
    {
        return {path + ": " + parse_error};
    }

    // The uniform block: a Uniform variable -> pointer -> struct decorated Block.
    std::vector<std::pair<uint32_t, uint32_t>> blocks;  // (variable, struct)
    for (const auto& [variable, pointer] : module.uniform_variables)
    {
        const auto pointer_type = module.types.find(pointer);
        if (pointer_type == module.types.end() || pointer_type->second.opcode != kOpTypePointer)
        {
            continue;
        }
        const uint32_t pointee = pointer_type->second.operands[1];
        const auto pointee_type = module.types.find(pointee);
        if (pointee_type != module.types.end() && pointee_type->second.opcode == kOpTypeStruct &&
            module.decorations.count({pointee, kDecorationBlock}) != 0)
        {
            blocks.emplace_back(variable, pointee);
        }
    }
    if (blocks.size() != 1)
    {
        return {name + ": " + std::to_string(blocks.size()) + " uniform blocks (expected exactly 1)"};
    }
    const auto [variable, block] = blocks[0];
    std::vector<std::string> errors;
    const auto set = module.decorations.find({variable, kDecorationDescriptorSet});
    const auto binding = module.decorations.find({variable, kDecorationBinding});
    if (set == module.decorations.end() || set->second != caelix::kFrameSetIndex ||
        binding == module.decorations.end() || binding->second != caelix::kFrameConstantsBinding)
    {
        errors.push_back(name + ": the uniform block is not at set " + std::to_string(caelix::kFrameSetIndex) +
                         ", binding " + std::to_string(caelix::kFrameConstantsBinding));
    }

    const std::vector<uint32_t>& members = module.types[block].operands;
    uint32_t end = 0;
    for (uint32_t i = 0; i < members.size(); ++i)
    {
        const auto member_name = module.member_names.find({block, i});
        const std::string label = name + ": member " + std::to_string(i) + " (" +
                                  (member_name != module.member_names.end() ? member_name->second : "?") + ")";
        std::optional<uint32_t> matrix_stride;
        if (const auto stride = module.member_matrix_strides.find({block, i});
            stride != module.member_matrix_strides.end())
        {
            matrix_stride = stride->second;
        }
        std::string type_error;
        const Layout layout = NaturalLayout(module, members[i], matrix_stride, type_error);
        if (!type_error.empty())
        {
            errors.push_back(label + ": " + type_error);
            return errors;
        }
        const uint32_t natural = (end + layout.align - 1) / layout.align * layout.align;
        const auto offset = module.member_offsets.find({block, i});
        if (offset == module.member_offsets.end())
        {
            errors.push_back(label + ": no Offset decoration");
            return errors;
        }
        if (offset->second != natural)
        {
            errors.push_back(label + ": SPIR-V offset " + std::to_string(offset->second) + " != C offset " +
                             std::to_string(natural));
            return errors;  // the first mismatching member; the later ones follow from it
        }
        end = natural + layout.size;
    }
    if (end != sizeof(caelix::FrameConstants))
    {
        errors.push_back(name + ": the block ends at " + std::to_string(end) + ", sizeof(FrameConstants) = " +
                         std::to_string(sizeof(caelix::FrameConstants)));
    }
    if (errors.empty())
    {
        std::printf("%s: %zu members, every offset matches, size %u\n", name.c_str(), members.size(), end);
    }
    return errors;
}

}  // namespace

TEST_CASE("frame constants: the uniform block of every frame-constant kernel has the C layout")
{
    const char* const kernels[] = {"caelix_path_trace",     "caelix_path_trace_no_light_nee",
                                   "caelix_path_trace_sss", "caelix_path_trace_sss_no_light_nee",
                                   "caelix_composite",      "caelix_cross_resolve",
                                   "caelix_sky_sh",         "caelix_upscale_encode",
                                   "caelix_voxel_id"};
    for (const char* kernel : kernels)
    {
        const std::vector<std::string> errors = CheckKernel(kernel);
        std::string text;
        for (const std::string& error : errors)
        {
            text += error + "\n";
        }
        INFO(text);
        CHECK(errors.empty());
    }
}

// ---------------------------------------------------------------------------
// Push constant blocks against the C++ structs

namespace {

struct Field
{
    const char* name;
    size_t offset;
};
#define CAELIX_FIELD(Struct, member) Field{#member, offsetof(Struct, member)}

// Checks the push constant block of one kernel: one block, every member named like a field of the C++
// struct and at its offset, as many members as fields.
std::vector<std::string> CheckPushBlock(const std::string& name, const std::vector<Field>& fields)
{
    const std::string path = std::string(CAELIX_SHADER_SPV_DIR) + "/" + name + ".spv";
    const std::vector<uint32_t> words = ReadWords(path);
    if (words.empty())
    {
        return {"cannot read " + path};
    }
    Module module;
    const std::string parse_error = Parse(words, module);
    if (!parse_error.empty())
    {
        return {path + ": " + parse_error};
    }
    if (module.push_variables.size() != 1)
    {
        return {name + ": " + std::to_string(module.push_variables.size()) +
                " push constant blocks (expected 1)"};
    }
    const auto pointer = module.types.find(module.push_variables[0].second);
    if (pointer == module.types.end() || pointer->second.opcode != kOpTypePointer)
    {
        return {name + ": the push constant variable has no pointer type"};
    }
    const uint32_t block = pointer->second.operands[1];
    const auto block_type = module.types.find(block);
    if (block_type == module.types.end() || block_type->second.opcode != kOpTypeStruct)
    {
        return {name + ": the push constant block is not a struct"};
    }
    std::vector<std::string> errors;
    const size_t members = block_type->second.operands.size();
    if (members != fields.size())
    {
        errors.push_back(name + ": " + std::to_string(members) + " members, the C++ struct has " +
                         std::to_string(fields.size()) + " fields");
    }
    for (uint32_t i = 0; i < members; ++i)
    {
        const auto member_name = module.member_names.find({block, i});
        const auto offset = module.member_offsets.find({block, i});
        if (member_name == module.member_names.end() || offset == module.member_offsets.end())
        {
            errors.push_back(name + ": member " + std::to_string(i) + " has no name or no offset");
            continue;
        }
        const Field* field = nullptr;
        for (const Field& f : fields)
        {
            field = member_name->second == f.name ? &f : field;
        }
        if (field == nullptr)
        {
            errors.push_back(name + ": member " + member_name->second + " is not a field of the C++ struct");
        }
        else if (field->offset != offset->second)
        {
            errors.push_back(name + ": member " + member_name->second + " at " +
                             std::to_string(offset->second) + ", the C++ field at " +
                             std::to_string(field->offset));
        }
    }
    if (errors.empty())
    {
        std::printf("%s: %zu push constant members match the C++ struct\n", name.c_str(), members);
    }
    return errors;
}

}  // namespace

TEST_CASE("push constants: the scene and light tree kernels have the C++ layouts")
{
    using namespace caelix;
    const std::vector<Field> finalize = {
        CAELIX_FIELD(FinalizePush, dirty_list),      CAELIX_FIELD(FinalizePush, page_table),
        CAELIX_FIELD(FinalizePush, slot_tables),     CAELIX_FIELD(FinalizePush, reverse_tables),
        CAELIX_FIELD(FinalizePush, group_infos),     CAELIX_FIELD(FinalizePush, entity_infos),
        CAELIX_FIELD(FinalizePush, group_grids),     CAELIX_FIELD(FinalizePush, aabb_pool),
        CAELIX_FIELD(FinalizePush, dirty_count),     CAELIX_FIELD(FinalizePush, records_per_page_shift),
        CAELIX_FIELD(FinalizePush, record_bytes),    CAELIX_FIELD(FinalizePush, group_shift),
        CAELIX_FIELD(FinalizePush, encoding),        CAELIX_FIELD(FinalizePush, pad),
        CAELIX_FIELD(FinalizePush, light_codes),     CAELIX_FIELD(FinalizePush, light_group_rows),
    };
    const std::vector<Field> group = {
        CAELIX_FIELD(LightGroupPush, descriptor),      CAELIX_FIELD(LightGroupPush, page_table),
        CAELIX_FIELD(LightGroupPush, slot_tables),     CAELIX_FIELD(LightGroupPush, reverse_tables),
        CAELIX_FIELD(LightGroupPush, materials),       CAELIX_FIELD(LightGroupPush, offsets),
        CAELIX_FIELD(LightGroupPush, staged_flags),    CAELIX_FIELD(LightGroupPush, records_per_page_shift),
        CAELIX_FIELD(LightGroupPush, record_bytes),    CAELIX_FIELD(LightGroupPush, group_base),
        CAELIX_FIELD(LightGroupPush, group_count),     CAELIX_FIELD(LightGroupPush, capacity_units),
        CAELIX_FIELD(LightGroupPush, pad),
    };
    const std::vector<Field> bvh = {
        CAELIX_FIELD(LightBvhPush, descriptor),    CAELIX_FIELD(LightBvhPush, scratch),
        CAELIX_FIELD(LightBvhPush, offsets),       CAELIX_FIELD(LightBvhPush, tree),
        CAELIX_FIELD(LightBvhPush, leaf_capacity), CAELIX_FIELD(LightBvhPush, node_capacity),
        CAELIX_FIELD(LightBvhPush, group_count),   CAELIX_FIELD(LightBvhPush, reason),
        CAELIX_FIELD(LightBvhPush, pass_recorded), CAELIX_FIELD(LightBvhPush, group_base),
        CAELIX_FIELD(LightBvhPush, pad),
    };
    const std::vector<Field> scan = {
        CAELIX_FIELD(LightScanPush, data),         CAELIX_FIELD(LightScanPush, total_address),
        CAELIX_FIELD(LightScanPush, gate_address), CAELIX_FIELD(LightScanPush, count),
        CAELIX_FIELD(LightScanPush, pad),
    };
    const std::vector<Field> sort = {
        CAELIX_FIELD(LightSortPush, keys_in),      CAELIX_FIELD(LightSortPush, values_in),
        CAELIX_FIELD(LightSortPush, keys_out),     CAELIX_FIELD(LightSortPush, values_out),
        CAELIX_FIELD(LightSortPush, histogram),    CAELIX_FIELD(LightSortPush, count_address),
        CAELIX_FIELD(LightSortPush, gate_address), CAELIX_FIELD(LightSortPush, shift),
        CAELIX_FIELD(LightSortPush, block_count),
    };
    const std::vector<Field> scatter = {
        CAELIX_FIELD(RecordScatterPush, list),  CAELIX_FIELD(RecordScatterPush, source),
        CAELIX_FIELD(RecordScatterPush, count), CAELIX_FIELD(RecordScatterPush, record_words8),
    };
    const std::vector<Field> hash = {
        CAELIX_FIELD(HashPush, keys),             CAELIX_FIELD(HashPush, stack),
        CAELIX_FIELD(HashPush, counters),         CAELIX_FIELD(HashPush, payloads),
        CAELIX_FIELD(HashPush, active),           CAELIX_FIELD(HashPush, dispatch),
        CAELIX_FIELD(HashPush, key_mask),         CAELIX_FIELD(HashPush, payload_capacity),
        CAELIX_FIELD(HashPush, payload_stride),   CAELIX_FIELD(HashPush, key_words),
        CAELIX_FIELD(HashPush, frame),            CAELIX_FIELD(HashPush, max_age),
        CAELIX_FIELD(HashPush, items_per_group),  CAELIX_FIELD(HashPush, pad),
    };
    const std::vector<std::pair<const char*, const std::vector<Field>*>> kernels = {
        {"caelix_hash_init", &hash},
        {"caelix_hash_begin", &hash},
        {"caelix_hash_maintain", &hash},
        {"caelix_hash_finish", &hash},
        {"caelix_finalize", &finalize},
        {"caelix_record_scatter", &scatter},
        {"caelix_light_tree_size", &group},
        {"caelix_light_tree_scan", &group},
        {"caelix_light_tree_build", &group},
        {"caelix_light_tree_commit", &group},
        {"caelix_light_bvh_begin", &bvh},
        {"caelix_light_gather_count", &bvh},
        {"caelix_light_gather_emit", &bvh},
        {"caelix_light_bvh_bounds", &bvh},
        {"caelix_light_bvh_morton", &bvh},
        {"caelix_light_bvh_karras", &bvh},
        {"caelix_light_bvh_aggregate", &bvh},
        {"caelix_light_bvh_depth", &bvh},
        {"caelix_light_bvh_wide", &bvh},
        {"caelix_light_bvh_finish", &bvh},
        {"caelix_light_bvh_scan", &scan},
        {"caelix_light_bvh_sort_histogram", &sort},
        {"caelix_light_bvh_sort_scatter", &sort},
    };
    for (const auto& [kernel, fields] : kernels)
    {
        const std::vector<std::string> errors = CheckPushBlock(kernel, *fields);
        std::string text;
        for (const std::string& error : errors)
        {
            text += error + "\n";
        }
        INFO(text);
        CHECK(errors.empty());
    }
}

// ---------------------------------------------------------------------------
// Named constants of the HLSL layout headers against the C++ constants

namespace {

// The value of a constant expression of an HLSL layout header: integer literals (decimal or hex, an
// optional u suffix), earlier constants, parentheses and + - * / << | &. nullopt when it is not one.
class Expression
{
public:
    Expression(const std::string& text, const std::map<std::string, uint64_t>& known)
        : text_(text), known_(known)
    {
    }

    std::optional<uint64_t> Evaluate()
    {
        const std::optional<uint64_t> value = Or();
        Skip();
        if (!value || at_ != text_.size())
        {
            return std::nullopt;
        }
        return value;
    }

private:
    void Skip()
    {
        while (at_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[at_])) != 0)
        {
            ++at_;
        }
    }
    bool Take(const char* op)
    {
        Skip();
        const size_t n = std::char_traits<char>::length(op);
        if (text_.compare(at_, n, op) == 0)
        {
            at_ += n;
            return true;
        }
        return false;
    }
    std::optional<uint64_t> Or()
    {
        std::optional<uint64_t> a = And();
        while (a && Take("|"))
        {
            const std::optional<uint64_t> b = And();
            a = b ? std::optional<uint64_t>(*a | *b) : std::nullopt;
        }
        return a;
    }
    std::optional<uint64_t> And()
    {
        std::optional<uint64_t> a = Shift();
        while (a && Take("&"))
        {
            const std::optional<uint64_t> b = Shift();
            a = b ? std::optional<uint64_t>(*a & *b) : std::nullopt;
        }
        return a;
    }
    std::optional<uint64_t> Shift()
    {
        std::optional<uint64_t> a = Sum();
        while (a && Take("<<"))
        {
            const std::optional<uint64_t> b = Sum();
            a = b ? std::optional<uint64_t>(*a << *b) : std::nullopt;
        }
        return a;
    }
    std::optional<uint64_t> Sum()
    {
        std::optional<uint64_t> a = Product();
        while (a)
        {
            if (Take("+"))
            {
                const std::optional<uint64_t> b = Product();
                a = b ? std::optional<uint64_t>(*a + *b) : std::nullopt;
            }
            else if (Take("-"))
            {
                const std::optional<uint64_t> b = Product();
                a = b ? std::optional<uint64_t>(*a - *b) : std::nullopt;
            }
            else
            {
                break;
            }
        }
        return a;
    }
    std::optional<uint64_t> Product()
    {
        std::optional<uint64_t> a = Primary();
        while (a)
        {
            if (Take("*"))
            {
                const std::optional<uint64_t> b = Primary();
                a = b ? std::optional<uint64_t>(*a * *b) : std::nullopt;
            }
            else if (Take("/"))
            {
                const std::optional<uint64_t> b = Primary();
                a = b && *b != 0 ? std::optional<uint64_t>(*a / *b) : std::nullopt;
            }
            else
            {
                break;
            }
        }
        return a;
    }
    std::optional<uint64_t> Primary()
    {
        Skip();
        if (Take("("))
        {
            const std::optional<uint64_t> value = Or();
            return Take(")") ? value : std::nullopt;
        }
        if (at_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[at_])) != 0)
        {
            size_t used = 0;
            const uint64_t value = std::stoull(text_.substr(at_), &used, 0);
            at_ += used;
            if (at_ < text_.size() && (text_[at_] == 'u' || text_[at_] == 'U'))
            {
                ++at_;
            }
            return value;
        }
        size_t end = at_;
        while (end < text_.size() &&
               (std::isalnum(static_cast<unsigned char>(text_[end])) != 0 || text_[end] == '_'))
        {
            ++end;
        }
        const auto found = known_.find(text_.substr(at_, end - at_));
        if (end == at_ || found == known_.end())
        {
            return std::nullopt;
        }
        at_ = end;
        return found->second;
    }

    const std::string& text_;
    const std::map<std::string, uint64_t>& known_;
    size_t at_ = 0;
};

// The named constants of one HLSL header in order: `static const uint kName = <expression>;` and
// `#define kName <expression>`. `errors` gets the ones whose value cannot be computed.
std::vector<std::pair<std::string, uint64_t>> HlslConstants(const std::string& file,
                                                            std::map<std::string, uint64_t>& known,
                                                            std::vector<std::string>& errors)
{
    std::ifstream in(std::string(CAELIX_SHADER_SOURCE_DIR) + "/" + file);
    if (!in)
    {
        errors.push_back("cannot read shaders/" + file);
        return {};
    }
    const std::regex constant(R"(^\s*static\s+const\s+uint\s+(k\w+)\s*=\s*([^;]+);)");
    const std::regex macro(R"(^\s*#define\s+(k\w+)\s+(.*?)\s*(//.*)?$)");
    std::vector<std::pair<std::string, uint64_t>> out;
    std::string line;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        std::smatch match;
        if (!std::regex_search(line, match, constant) && !std::regex_search(line, match, macro))
        {
            continue;
        }
        const std::string name = match[1];
        const std::string text = match[2];
        const std::optional<uint64_t> value = Expression(text, known).Evaluate();
        if (!value)
        {
            errors.push_back(file + ": " + name + " = " + text + " is not a constant expression");
            continue;
        }
        known[name] = *value;
        out.emplace_back(name, *value);
    }
    return out;
}

}  // namespace

TEST_CASE("layout constants: every HLSL layout constant equals the C++ constant of the same name")
{
    using namespace caelix;
    const RecordLayout record = RecordLayout::ForBlockBytes(2);  // the HLSL record is the 16-bit one
#define CAELIX_CONSTANT(name) {#name, static_cast<uint64_t>(name)}
    const std::map<std::string, uint64_t> cpp = {
        // shaders/common/caelix_record.hlsl: lib/src/scene/record_layout.h, gpu_structs.h
        CAELIX_CONSTANT(kPageTableEntryBytes),
        CAELIX_CONSTANT(kRecordInfoWords),
        CAELIX_CONSTANT(kRecordOccupancyWords),
        {"kRecordPayloadWord", record.payload_first_word()},
        {"kRecordWords", record.record_words},
        CAELIX_CONSTANT(kRecordWordBytes),
        CAELIX_CONSTANT(kRecordPosMask),
        CAELIX_CONSTANT(kRecordCoarseShift),
        CAELIX_CONSTANT(kRecordCoarseMask),
        CAELIX_CONSTANT(kRecordBoundsWord),
        CAELIX_CONSTANT(kRecordBoundsAxisBits),
        CAELIX_CONSTANT(kRecordBoundsAxisMask),
        CAELIX_CONSTANT(kRecordBoundsMaxShift),
        CAELIX_CONSTANT(kBrickEdge),
        CAELIX_CONSTANT(kBrickVoxels),
        CAELIX_CONSTANT(kPayloadOddShift),
        CAELIX_CONSTANT(kSlotRowBytes),
        CAELIX_CONSTANT(kSlotRowPosMask),
        CAELIX_CONSTANT(kSlotRowStateShift),
        {"kSlotStateResident", kSlotResident},
        CAELIX_CONSTANT(kInvalidRecord),
        CAELIX_CONSTANT(kAabbRowBytes),
        {"kQuietNaN", kQuietNaNBits},
        CAELIX_CONSTANT(kNoBrickId),
        CAELIX_CONSTANT(kNoGroup),
        CAELIX_CONSTANT(kDirtyEntryBytes),
        CAELIX_CONSTANT(kGroupInfoCoord),
        CAELIX_CONSTANT(kGroupInfoEntity),
        CAELIX_CONSTANT(kGroupInfoSlotRowBase),
        CAELIX_CONSTANT(kGroupInfoReverseBase),
        CAELIX_CONSTANT(kGroupInfoAabbRowBase),
        CAELIX_CONSTANT(kGroupInfoHighWater),
        CAELIX_CONSTANT(kGroupInfoBytes),
        CAELIX_CONSTANT(kEntityInfoGridMin),
        CAELIX_CONSTANT(kEntityInfoGridBase),
        CAELIX_CONSTANT(kEntityInfoGridDims),
        CAELIX_CONSTANT(kEntityInfoLife),
        CAELIX_CONSTANT(kEntityInfoCurRow0),
        CAELIX_CONSTANT(kEntityInfoCurRow1),
        CAELIX_CONSTANT(kEntityInfoCurRow2),
        CAELIX_CONSTANT(kEntityInfoPrevRow0),
        CAELIX_CONSTANT(kEntityInfoPrevRow1),
        CAELIX_CONSTANT(kEntityInfoPrevRow2),
        CAELIX_CONSTANT(kEntityInfoBytes),
        CAELIX_CONSTANT(kEncodingPacked),
        CAELIX_CONSTANT(kEncodingBlockId),
        CAELIX_CONSTANT(kPackedOpaqueMask),
        CAELIX_CONSTANT(kPackedIdMask),
        CAELIX_CONSTANT(kPackedFaceShift),
        CAELIX_CONSTANT(kBlockIdOpaqueMask),
        CAELIX_CONSTANT(kBlockIdIdMask),
        CAELIX_CONSTANT(kBlockIdFaceShift),
        CAELIX_CONSTANT(kRecordScatterEntryBytes),
        CAELIX_CONSTANT(kRecordScatterSourceUnit),
        CAELIX_CONSTANT(kScatterKeepBoundsWord),
        CAELIX_CONSTANT(kScatterRecordsPerGroup),
        CAELIX_CONSTANT(kScatterThreadsPerGroup),
        CAELIX_CONSTANT(kScatterMaxRecordWords8),
        // shaders/common/caelix_material.hlsl: lib/src/scene/material_pack.h (through material_table.h)
        CAELIX_CONSTANT(kMaterialBytes),
        CAELIX_CONSTANT(kMaterialAlbedo),
        CAELIX_CONSTANT(kMaterialEmission),
        CAELIX_CONSTANT(kMaterialIor),
        CAELIX_CONSTANT(kMaterialSss),
        CAELIX_CONSTANT(kMaterialTexture),
        CAELIX_CONSTANT(kSubsurfaceBytes),
        CAELIX_CONSTANT(kSubsurfaceAlbedo),
        CAELIX_CONSTANT(kSubsurfaceSteps),
        CAELIX_CONSTANT(kSubsurfaceProbability),
        CAELIX_CONSTANT(kSubsurfaceParams),
        {"kSssModelNone", CAELIX_SSS_MODEL_NONE},
        {"kSssModelHg", CAELIX_SSS_MODEL_HG},
        // shaders/common/caelix_light_tree_layout.hlsl: lib/src/light/light_tree_layout.h
        CAELIX_CONSTANT(kLtNone),
        CAELIX_CONSTANT(kLtTrees),
        CAELIX_CONSTANT(kLtMaxLightBricksPerGroup),
        CAELIX_CONSTANT(kLtRecordFaceCountShift),
        CAELIX_CONSTANT(kLtRecordFaceCountMask),
        CAELIX_CONSTANT(kLtRecordStaleBit),
        CAELIX_CONSTANT(kLtDescControl),
        CAELIX_CONSTANT(kLtDescHeap0),
        CAELIX_CONSTANT(kLtDescHeap1),
        CAELIX_CONSTANT(kLtDescGroupRows),
        CAELIX_CONSTANT(kLtDescGroupInfos),
        CAELIX_CONSTANT(kLtDescEntityInfos),
        CAELIX_CONSTANT(kLtDescEntityMotion),
        CAELIX_CONSTANT(kLtDescGrids),
        CAELIX_CONSTANT(kLtDescNodes0),
        CAELIX_CONSTANT(kLtDescLeaves0),
        CAELIX_CONSTANT(kLtDescTreeStride),
        CAELIX_CONSTANT(kLtDescLightCodes),
        CAELIX_CONSTANT(kLtDescTreeHeaders),
        CAELIX_CONSTANT(kLtDescGroupShift),
        CAELIX_CONSTANT(kLtDescEncoding),
        CAELIX_CONSTANT(kLtDescGroupCount),
        CAELIX_CONSTANT(kLtDescriptorBytes),
        CAELIX_CONSTANT(kLtSummaryFace),
        CAELIX_CONSTANT(kLtSummaryLo),
        CAELIX_CONSTANT(kLtSummaryHi),
        CAELIX_CONSTANT(kLtSummaryBytes),
        CAELIX_CONSTANT(kLtHdrRoot),
        CAELIX_CONSTANT(kLtHdrLeafCount),
        CAELIX_CONSTANT(kLtHdrNodeCount),
        CAELIX_CONSTANT(kLtTreeHeaderBytes),
        CAELIX_CONSTANT(kLtStBuilding),
        CAELIX_CONSTANT(kLtStLeavesFound),
        CAELIX_CONSTANT(kLtStLeavesDropped),
        CAELIX_CONSTANT(kLtStBuilds),
        CAELIX_CONSTANT(kLtStNodeOverflow),
        CAELIX_CONSTANT(kLtStBounds),
        CAELIX_CONSTANT(kLtTreeStateBytes),
        CAELIX_CONSTANT(kLtCtlCurrent),
        CAELIX_CONSTANT(kLtCtlPassResult),
        CAELIX_CONSTANT(kLtCtlNeededUnits),
        CAELIX_CONSTANT(kLtCtlLightGroups),
        CAELIX_CONSTANT(kLtCtlBricksDropped),
        CAELIX_CONSTANT(kLtCtlLightBricks),
        CAELIX_CONSTANT(kLtCtlLightFaces),
        CAELIX_CONSTANT(kLtCtlPasses),
        CAELIX_CONSTANT(kLtCtlFailedPasses),
        CAELIX_CONSTANT(kLtCtlPassFlags),
        CAELIX_CONSTANT(kLtCtlRecordsRecomputed),
        CAELIX_CONSTANT(kLtCtlRecordsCopied),
        CAELIX_CONSTANT(kLtCtlTreesChanged),
        CAELIX_CONSTANT(kLtCtlTreeHeaders),
        CAELIX_CONSTANT(kLtCtlTreeStates),
        CAELIX_CONSTANT(kLtControlBytes),
        CAELIX_CONSTANT(kLtPassFailed),
        CAELIX_CONSTANT(kLtPassOk),
        CAELIX_CONSTANT(kLtPassNoChange),
        CAELIX_CONSTANT(kLtPassFlagChanged),
        CAELIX_CONSTANT(kLtPassFlagTreeShift),
        CAELIX_CONSTANT(kLtRowBlockOffset),
        CAELIX_CONSTANT(kLtRowBlockUnits),
        CAELIX_CONSTANT(kLtRowGeneration),
        CAELIX_CONSTANT(kLtRowGpuFlags),
        CAELIX_CONSTANT(kLtRowCpuFlags),
        CAELIX_CONSTANT(kLtRowBricks),
        CAELIX_CONSTANT(kLtRowLightFaces),
        CAELIX_CONSTANT(kLtGroupRowBytes),
        CAELIX_CONSTANT(kLtRowBricksMask),
        CAELIX_CONSTANT(kLtRowDroppedShift),
        CAELIX_CONSTANT(kLtGenerationLifeShift),
        CAELIX_CONSTANT(kLtGenerationVersionMask),
        CAELIX_CONSTANT(kLtGpuLightChanged),
        CAELIX_CONSTANT(kLtCpuRebuild),
        CAELIX_CONSTANT(kLtCpuReleased),
        CAELIX_CONSTANT(kLtCpuFresh),
        CAELIX_CONSTANT(kLtUnitBytes),
        CAELIX_CONSTANT(kLtBlockIndexOffset),
        CAELIX_CONSTANT(kLtIndexPos),
        CAELIX_CONSTANT(kLtIndexDenseId),
        CAELIX_CONSTANT(kLtIndexRecordOffset),
        CAELIX_CONSTANT(kLtIndexBvhLeaf),
        CAELIX_CONSTANT(kLtIndexEntryBytes),
        CAELIX_CONSTANT(kLtBvhLeafTreeShift),
        CAELIX_CONSTANT(kLtBvhLeafIndexMask),
        CAELIX_CONSTANT(kLtRecordFace),
        CAELIX_CONSTANT(kLtRecordBoxLo),
        CAELIX_CONSTANT(kLtRecordBoxHi),
        CAELIX_CONSTANT(kLtRecordPos),
        CAELIX_CONSTANT(kLtRecordEntryCount),
        CAELIX_CONSTANT(kLtRecordCountsLo),
        CAELIX_CONSTANT(kLtRecordCountsHi),
        CAELIX_CONSTANT(kLtRecordHeaderBytes),
        CAELIX_CONSTANT(kLtFamilyCountBits),
        CAELIX_CONSTANT(kLtFamilyCountMask),
        CAELIX_CONSTANT(kLtFamiliesPerCountWord),
        CAELIX_CONSTANT(kLtFaceEntryBytes),
        CAELIX_CONSTANT(kLtFaceEntryVoxelMask),
        CAELIX_CONSTANT(kLtFaceEntryBlockShift),
        CAELIX_CONSTANT(kLtVoxelBits),
        CAELIX_CONSTANT(kLtVoxelMask),
        CAELIX_CONSTANT(kLtLeafGroup),
        CAELIX_CONSTANT(kLtLeafEntry),
        CAELIX_CONSTANT(kLtLeafParent),
        CAELIX_CONSTANT(kLtLeafBytes),
        CAELIX_CONSTANT(kLtParentSlotBits),
        CAELIX_CONSTANT(kLtParentSlotMask),
        CAELIX_CONSTANT(kLtNodeChildFace),
        CAELIX_CONSTANT(kLtNodeChildLo),
        CAELIX_CONSTANT(kLtNodeChildHi),
        CAELIX_CONSTANT(kLtNodeChild),
        CAELIX_CONSTANT(kLtNodeParent),
        CAELIX_CONSTANT(kLtNodeParentSlot),
        CAELIX_CONSTANT(kLtNodeBytes),
        CAELIX_CONSTANT(kLtNodeSlots),
        CAELIX_CONSTANT(kLtNodeFaceBytes),
        CAELIX_CONSTANT(kLtNodeBoxBytes),
        CAELIX_CONSTANT(kLtChildLeafBit),
        CAELIX_CONSTANT(kLtScratchKeys),
        CAELIX_CONSTANT(kLtScratchValues),
        CAELIX_CONSTANT(kLtScratchSortBufferStride),
        CAELIX_CONSTANT(kLtScratchLeft),
        CAELIX_CONSTANT(kLtScratchRight),
        CAELIX_CONSTANT(kLtScratchParent),
        CAELIX_CONSTANT(kLtScratchSortedLeafParent),
        CAELIX_CONSTANT(kLtScratchCounter),
        CAELIX_CONSTANT(kLtScratchWide),
        CAELIX_CONSTANT(kLtScratchAggregates),
        CAELIX_CONSTANT(kLtScratchHistogram),
        CAELIX_CONSTANT(kLtSortBits),
        CAELIX_CONSTANT(kLtSortDigits),
        CAELIX_CONSTANT(kLtSortPasses),
        CAELIX_CONSTANT(kLtSortBlock),
        // shaders/common/caelix_hash_table.hlsl: lib/src/rhi/hash_table_layout.h
        CAELIX_CONSTANT(kHashNone),
        CAELIX_CONSTANT(kHashProbes),
        CAELIX_CONSTANT(kHashKeySlotsPerPayload),
        CAELIX_CONSTANT(kHashSlotKeyLo),
        CAELIX_CONSTANT(kHashSlotKeyHi),
        CAELIX_CONSTANT(kHashSlotPayload),
        CAELIX_CONSTANT(kHashSlotBytes),
        CAELIX_CONSTANT(kHashPayloadState),
        CAELIX_CONSTANT(kHashPayloadTouched),
        CAELIX_CONSTANT(kHashPayloadKey),
        CAELIX_CONSTANT(kHashMaxKeyWords),
        CAELIX_CONSTANT(kHashStateFree),
        CAELIX_CONSTANT(kHashStateUninitialised),
        CAELIX_CONSTANT(kHashStateReady),
        CAELIX_CONSTANT(kHashStackTop),
        CAELIX_CONSTANT(kHashStackEntries),
        CAELIX_CONSTANT(kHashCounterLive),
        CAELIX_CONSTANT(kHashCounterActive),
        CAELIX_CONSTANT(kHashCounterInserted),
        CAELIX_CONSTANT(kHashCounterPending),
        CAELIX_CONSTANT(kHashCounterFailedSlot),
        CAELIX_CONSTANT(kHashCounterFailedPayload),
        CAELIX_CONSTANT(kHashCounterEvicted),
        CAELIX_CONSTANT(kHashCounterRebuildDropped),
        CAELIX_CONSTANT(kHashCounterUser),
        CAELIX_CONSTANT(kHashCounterUserWords),
        CAELIX_CONSTANT(kHashCounterBytes),
        CAELIX_CONSTANT(kHashDispatchX),
        CAELIX_CONSTANT(kHashDispatchY),
        CAELIX_CONSTANT(kHashDispatchZ),
        CAELIX_CONSTANT(kHashDispatchCount),
        CAELIX_CONSTANT(kHashDispatchBytes),
        CAELIX_CONSTANT(kHashDispatchRowGroups),
        CAELIX_CONSTANT(kHashThreads),
        CAELIX_CONSTANT(kHashMaxPayloads),
        CAELIX_CONSTANT(kHashHit),
        CAELIX_CONSTANT(kHashInserted),
        CAELIX_CONSTANT(kHashPending),
        CAELIX_CONSTANT(kHashFullWindow),
        CAELIX_CONSTANT(kHashNoPayload),
        CAELIX_CONSTANT(kHashEmptyEntry),
        CAELIX_CONSTANT(kHashMissing),
    };
#undef CAELIX_CONSTANT
    std::map<std::string, uint64_t> known;
    std::vector<std::string> errors;
    size_t checked = 0;
    for (const char* file : {"common/caelix_record.hlsl", "common/caelix_material.hlsl",
                             "common/caelix_light_tree_layout.hlsl", "common/caelix_hash_table.hlsl"})
    {
        for (const auto& [name, value] : HlslConstants(file, known, errors))
        {
            const auto found = cpp.find(name);
            if (found == cpp.end())
            {
                errors.push_back(std::string(file) + ": " + name +
                                 " has no C++ constant in this test's table");
            }
            else if (found->second != value)
            {
                errors.push_back(std::string(file) + ": " + name + " = " + std::to_string(value) + ", C++ " +
                                 std::to_string(found->second));
            }
            ++checked;
        }
    }
    for (const auto& [name, value] : cpp)
    {
        if (known.count(name) == 0)
        {
            errors.push_back(name + " is in the C++ table but in no HLSL layout header");
        }
    }
    std::string text;
    for (const std::string& error : errors)
    {
        text += error + "\n";
    }
    INFO(text);
    CHECK(errors.empty());
    std::printf("%zu HLSL layout constants compared with C++\n", checked);
}
