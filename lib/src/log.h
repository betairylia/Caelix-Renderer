#pragma once
#include "caelix/caelix.h"

namespace caelix {

// Stores the host's log callback. Thread-safe.
void SetLogCallback(CaelixLogCallback callback, void* user_data);

// printf-style. Formats the message and forwards it to the host's callback.
// Does nothing when no callback is set. Never throws.
void log(CaelixLogLevel level, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

}  // namespace caelix

// The level names are shortened on purpose: CAELIX_LOG_INFO etc. are already the
// names of the CaelixLogLevel enumerators in the public header.
#define CAELIX_LOGD(...) ::caelix::log(CAELIX_LOG_DEBUG, __VA_ARGS__)
#define CAELIX_LOGI(...) ::caelix::log(CAELIX_LOG_INFO, __VA_ARGS__)
#define CAELIX_LOGW(...) ::caelix::log(CAELIX_LOG_WARNING, __VA_ARGS__)
#define CAELIX_LOGE(...) ::caelix::log(CAELIX_LOG_ERROR, __VA_ARGS__)
