#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <cctype>

// Parse exactly N finite raw features; never accept a partial curve.
inline bool parse_features(char* line, double* values, size_t count) {
    char* cursor = line;
    for (size_t i = 0; i < count; ++i) {
        while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
        errno = 0;
        char* end = nullptr;
        double value = std::strtod(cursor, &end);
        if (end == cursor || errno == ERANGE || !std::isfinite(value)) return false;
        while (std::isspace(static_cast<unsigned char>(*end))) ++end;
        if ((i + 1 < count && *end != ',') || (i + 1 == count && *end != '\0')) return false;
        values[i] = value;
        cursor = end + (i + 1 < count ? 1 : 0);
    }
    return true;
}

inline bool quantize_feature(double raw, double scale, double offset, double multiplier,
                             int8_t& result, bool& clipped) {
    const double value = (raw * scale + offset) * multiplier;
    if (!std::isfinite(value)) return false;
    clipped = value < -127.0 || value > 127.0;
    // Match numpy.round (ties to even) used by calibration/export.
    result = static_cast<int8_t>(std::nearbyint(std::fmax(-127.0, std::fmin(127.0, value))));
    return true;
}

inline bool is_degraded(float soh) { return soh <= 80.0f; }
