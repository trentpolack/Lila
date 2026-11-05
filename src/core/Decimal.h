#pragma once

#include <cctype>
#include <charconv>
#include <cstdint>
#include <type_traits>

#if defined(__APPLE__)
#include <fast_float/fast_float.h>
#endif

namespace lila
{
/** Locale-independent from_chars semantics, including float/double on macOS before 26. */
template <class T> inline std::from_chars_result fromChars(const char* first, const char* last, T& out)
{
#if defined(__APPLE__)
    if constexpr (std::is_floating_point_v<T>)
    {
        // fast_float may write infinity or zero on a range error; std::from_chars leaves out unchanged.
        T value{};
        const auto result = fast_float::from_chars(first, last, value);
        if (result.ec == std::errc())
            out = value;
        return {result.ptr, result.ec};
    }
    else
#endif
        return std::from_chars(first, last, out);
}

/**
 * A plain decimal (optional minus, digits, optional fraction and exponent) of at most 15 significant digits with a power
 * of ten within 22 of them, converted exactly: both are exact doubles, so one multiply or divide rounds correctly, as
 * from_chars would. Null for anything else (longer mantissas, far exponents, inf, nan, hex), which from_chars takes.
 */
inline const char* fastDecimal(const char* p, const char* end, double& out)
{
    static constexpr double kPow10[] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
                                        1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
    const bool negative = p < end && *p == '-';
    if (negative)
        ++p;
    uint64_t digits = 0;
    int count = 0, scale = 0;
    const char* start = p;
    for (; p < end && unsigned(*p - '0') < 10; ++p)
    {
        if (digits != 0 || *p != '0')
            ++count;
        digits = digits * 10 + uint64_t(*p - '0');
        if (count > 15)
            return nullptr;
    }
    const bool whole = p != start;
    bool fraction = false;
    if (p < end && *p == '.')
    {
        ++p;
        for (; p < end && unsigned(*p - '0') < 10; ++p, --scale)
        {
            fraction = true;
            if (digits != 0 || *p != '0')
                ++count;
            digits = digits * 10 + uint64_t(*p - '0');
            if (count > 15)
                return nullptr;
        }
    }
    if (!whole && !fraction)
        return nullptr;
    if (p < end && (*p == 'e' || *p == 'E'))
    {
        const char* e = p + 1;
        bool expNegative = false;
        if (e < end && (*e == '-' || *e == '+'))
            expNegative = *e++ == '-';
        int exponent = 0;
        const char* expStart = e;
        for (; e < end && unsigned(*e - '0') < 10; ++e)
        {
            if (exponent > 1000)
                return nullptr;
            exponent = exponent * 10 + (*e - '0');
        }
        if (e == expStart)
            return nullptr;
        scale += expNegative ? -exponent : exponent;
        p = e;
    }
    if (p < end && (std::isalpha(static_cast<unsigned char>(*p)) || *p == '.'))
        return nullptr;
    if (scale < -22 || scale > 22)
        return nullptr;
    double value = double(digits);
    value = scale < 0 ? value / kPow10[-scale] : value * kPow10[scale];
    out = negative ? -value : value;
    return p;
}

/** A double as std::from_chars reads it (same result, same stopping point), through the exact fast path when it applies. */
inline std::from_chars_result parseDouble(const char* first, const char* last, double& out)
{
    if (const char* p = fastDecimal(first, last, out))
        return {p, std::errc()};
    return fromChars(first, last, out);
}
} // namespace lila
