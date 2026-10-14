#ifndef CAELIX_H
#define CAELIX_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32)
  #if defined(CAELIX_BUILD_SHARED)
    #define CAELIX_API __declspec(dllexport)
  #else
    #define CAELIX_API __declspec(dllimport)
  #endif
#else
  #define CAELIX_API __attribute__((visibility("default")))
#endif

/* Bumped on every breaking change of this header. */
#define CAELIX_ABI_VERSION 0u

typedef struct CaelixVersion {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  uint32_t abi;   /* == CAELIX_ABI_VERSION of the built library */
} CaelixVersion;

/* Fills *out with the library's version. Never fails. */
CAELIX_API void caelix_get_version(CaelixVersion* out);
/* Static string like "0.1.0", owned by the library. */
CAELIX_API const char* caelix_version_string(void);
#ifdef __cplusplus
}
#endif
#endif
