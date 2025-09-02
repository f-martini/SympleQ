#pragma once

#include "Base/Commons.h"

#include "Base/FileSystem/Environment.h"
#include "Base/FileSystem/FileSystem.h"

#if defined(_WIN32) || defined(__CYGWIN__)

#ifdef BASE_BUILD
#define BASE_API __declspec(dllexport)
#else
#define BASE_API __declspec(dllimport)
#endif

#else

#define BASE_API __attribute__((visibility("default")))

#endif

namespace Base {

BASE_API const char* GetVersion();

}  // namespace Base