#pragma once

#include "LightMath/Arithmetic.h"

#if defined(_WIN32) || defined(__CYGWIN__)

#ifdef LIGHTMATH_BUILD
#define LIGHTMATH_API __declspec(dllexport)
#else
#define LIGHTMATH_API __declspec(dllimport)
#endif

#else

#define LIGHTMATH_API __attribute__((visibility("default")))

#endif