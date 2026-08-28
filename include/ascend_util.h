// ============================================================================
// Shared utilities for Ascend CANN wrapper libraries
// ============================================================================

#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace ascend {

inline uint16_t floatToFp16(float value) {
    if (std::isnan(value)) {
        return 0x7E00;
    }
    if (std::isinf(value)) {
        return value < 0.0f ? 0xFC00 : 0x7C00;
    }

    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        mantissa >>= shift;
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }

    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

inline float fp16ToFloat(uint16_t value) {
    const uint32_t sign = (value & 0x8000) << 16;
    const uint32_t exponent = (value >> 10) & 0x1F;
    const uint32_t mantissa = value & 0x3FF;

    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t mant = mantissa;
            int32_t exp = -14;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            bits = sign | static_cast<uint32_t>((exp + 127) << 23) | (mant << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000 | (mantissa << 13);
    } else {
        bits = sign | static_cast<uint32_t>((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline void cpuGemmFp16AccFp32(
    const std::vector<uint16_t>& a,
    const std::vector<uint16_t>& b,
    const std::vector<float>& c,
    std::vector<float>& out,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    out.assign(static_cast<size_t>(M) * N, 0.0f);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                const float av = fp16ToFloat(a[static_cast<size_t>(m) * K + k]);
                const float bv = fp16ToFloat(b[static_cast<size_t>(k) * N + n]);
                sum += av * bv;
            }
            out[static_cast<size_t>(m) * N + n] = sum + c[static_cast<size_t>(m) * N + n];
        }
    }
}

} // namespace ascend