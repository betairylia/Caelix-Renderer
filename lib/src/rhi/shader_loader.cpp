#include "rhi/shader_loader.h"

#include "core/log.h"
#include "rhi/shaders.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace caelix {
namespace {

constexpr uint32_t kSpirvMagic = 0x07230203u;

// The value of an environment variable, or an empty string.
std::string GetEnv(const char* variable)
{
#if defined(_MSC_VER)
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, variable) != 0 || value == nullptr)
    {
        return {};
    }
    std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(variable);
    return value != nullptr ? std::string(value) : std::string();
#endif
}

[[noreturn]] void Fail(const std::string& message)
{
    CAELIX_LOGE("%s", message.c_str());
    throw std::runtime_error(message);
}

}  // namespace

std::span<const uint32_t> FindEmbeddedShader(std::string_view name)
{
    for (size_t i = 0; i < kEmbeddedShaderCount; ++i)
    {
        if (name == kEmbeddedShaders[i].name)
        {
            return {kEmbeddedShaders[i].words, kEmbeddedShaders[i].word_count};
        }
    }
    return {};
}

std::vector<uint32_t> LoadShader(std::string_view name)
{
    const std::string shader(name);
    const std::string dir = GetEnv("CAELIX_SHADER_DIR");
    if (!dir.empty())
    {
        const std::filesystem::path path = std::filesystem::path(dir) / (shader + ".spv");
        std::error_code error;
        if (std::filesystem::is_regular_file(path, error))
        {
            const std::string path_text = path.string();
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
            {
                Fail("shader " + shader + ": cannot open " + path_text);
            }
            const std::streamoff bytes = file.tellg();
            if (bytes <= 0 || bytes % 4 != 0)
            {
                Fail("shader " + shader + ": " + path_text + " is not a SPIR-V word stream");
            }
            std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(words.data()), bytes))
            {
                Fail("shader " + shader + ": cannot read " + path_text);
            }
            if (words[0] != kSpirvMagic)
            {
                Fail("shader " + shader + ": " + path_text + " has no SPIR-V magic number");
            }
            CAELIX_LOGI("shader %s loaded from %s", shader.c_str(), path_text.c_str());
            return words;
        }
    }
    const std::span<const uint32_t> embedded = FindEmbeddedShader(name);
    if (embedded.empty())
    {
        Fail("shader " + shader + ": no embedded SPIR-V with this name");
    }
    return {embedded.begin(), embedded.end()};
}

}  // namespace caelix
