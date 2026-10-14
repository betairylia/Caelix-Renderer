#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

namespace caelix {

// The SPIR-V words of the shader `name` (file name without extension).
//
// When the environment variable CAELIX_SHADER_DIR is set and
// <CAELIX_SHADER_DIR>/<name>.spv exists, that file is loaded (logged at INFO),
// so a shader can be changed without rebuilding the library. Otherwise the
// array embedded at build time is used. Throws std::runtime_error (after an
// error log) when the file cannot be read, is not SPIR-V, or no embedded array
// exists.
std::vector<uint32_t> LoadShader(std::string_view name);

}  // namespace caelix
