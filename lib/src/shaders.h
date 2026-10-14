#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace caelix {

// One SPIR-V module compiled from shaders/ at build time (cmake/Shaders.cmake)
// and embedded into the library.
struct EmbeddedShader {
  const char* name;  // file name without extension, e.g. "caelix_finalize"
  const uint32_t* words;
  size_t word_count;
};

// Defined in the generated caelix_shaders_embedded.cpp.
extern const EmbeddedShader kEmbeddedShaders[];
extern const size_t kEmbeddedShaderCount;

// The embedded SPIR-V of `name`, or an empty span when there is none.
std::span<const uint32_t> FindEmbeddedShader(std::string_view name);

}  // namespace caelix
