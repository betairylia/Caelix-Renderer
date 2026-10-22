// The uniform block of the frame constants in the compiled kernels against the C layout of
// lib/src/render/frame_constants.h. No GPU: the test reads the .spv files of the build
// (CAELIX_SHADER_SPV_DIR, written by cmake/Shaders.cmake) and parses the SPIR-V binary.
//
// For each of the five frame-constant kernels it finds the variable of storage class Uniform whose
// pointee is a struct decorated Block (exactly one), checks its DescriptorSet / Binding against
// kFrameSetIndex / kFrameConstantsBinding, recomputes the C natural layout of the block's member
// sequence (each member aligned to its scalar size; arrays and matrices without padding) and requires
// every member's Offset decoration to equal that offset, every ArrayStride / MatrixStride to equal
// the element / column size, and the end of the last member to equal sizeof(FrameConstants) (whose
// field offsets the static_asserts of frame_constants.h pin). DXC's constant buffer layout differs
// from the C layout when a vector crosses a 16-byte row or an array element is not 16 bytes; such a
// change of CaelixFrameConstants fails here.
#include "render/frame_constants.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef CAELIX_SHADER_SPV_DIR
#error "CAELIX_SHADER_SPV_DIR must name the build's shader folder"
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
    const char* const kernels[] = {"caelix_path_trace", "caelix_composite", "caelix_cross_resolve",
                                   "caelix_sky_sh", "caelix_upscale_encode"};
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
