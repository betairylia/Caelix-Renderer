#pragma once
// The one include for ankerl::unordered_dense. Its include directory is already
// a SYSTEM directory (see cmake/Dependencies.cmake); the pragma keeps /W4 quiet
// when a generator does not support external includes.
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <ankerl/unordered_dense.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
