#include <doctest/doctest.h>

#include "core/Decimal.h"

#include <bit>
#include <cmath>
#include <limits>
#include <string_view>

using namespace lila;

TEST_CASE("portable numeric parsing handles float and integer destinations")
{
    for (float expected : {0.0f, -0.0f, 0.1f, std::numeric_limits<float>::denorm_min(),
                           std::numeric_limits<float>::max(), std::nextafter(1.0f, 2.0f)})
    {
        char text[64];
        const auto written = std::to_chars(text, text + sizeof(text), expected);
        REQUIRE(written.ec == std::errc());
        float actual = 42.0f;
        const auto read = fromChars(text, written.ptr, actual);
        CHECK(read.ec == std::errc());
        CHECK(read.ptr == written.ptr);
        CHECK(std::bit_cast<uint32_t>(actual) == std::bit_cast<uint32_t>(expected));
    }
    for (std::string_view text : {"bad", "+1", "1e99", "1e-99"})
    {
        float value = 42.0f;
        CHECK(fromChars(text.data(), text.data() + text.size(), value).ec != std::errc());
        CHECK(value == 42.0f);
    }
    const std::string_view integer = "-123rest";
    int value = 0;
    const auto result = fromChars(integer.data(), integer.data() + integer.size(), value);
    CHECK(result.ec == std::errc());
    CHECK(result.ptr == integer.data() + 4);
    CHECK(value == -123);
}

TEST_CASE("decimal parsing respects token boundaries and preserves values on errors")
{
    for (std::string_view text : {"", "words", "+1.5", " 1.5", "1e9999", "-1e9999", "1e-9999"})
    {
        INFO(text);
        double value = 42.0;
        const auto result = parseDouble(text.data(), text.data() + text.size(), value);
        CHECK(result.ec != std::errc());
        CHECK(value == 42.0);
    }
    for (std::string_view text : {"1.25rest", "1.25e+", "1.25,2.5"})
    {
        INFO(text);
        double value = 0.0;
        const auto result = parseDouble(text.data(), text.data() + text.size(), value);
        CHECK(result.ec == std::errc());
        CHECK(result.ptr == text.data() + 4);
        CHECK(value == 1.25);
    }
    const char bounded[] = {'1', '.', '2', '5', '9'};
    double value = 0.0;
    const auto result = parseDouble(bounded, bounded + 4, value);
    CHECK(result.ec == std::errc());
    CHECK(result.ptr == bounded + 4);
    CHECK(value == 1.25);
}

TEST_CASE("decimal parsing preserves binary precision across the double range")
{
    for (double expected : {0.0, -0.0, 0.1, -0.1, 1.0 / 3.0, 1e-300, 1e300,
                            std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(),
                            std::numeric_limits<double>::max(), std::nextafter(1.0, 2.0)})
    {
        char text[64];
        const auto written = std::to_chars(text, text + sizeof(text), expected);
        REQUIRE(written.ec == std::errc());
        INFO(std::string_view(text, written.ptr));
        double actual = 42.0;
        const auto read = parseDouble(text, written.ptr, actual);
        CHECK(read.ec == std::errc());
        CHECK(read.ptr == written.ptr);
        CHECK(std::bit_cast<uint64_t>(actual) == std::bit_cast<uint64_t>(expected));
    }
}

TEST_CASE("decimal fallback accepts infinity and NaN")
{
    for (std::string_view text : {"inf", "-inf", "infinity", "nan", "nan(payload)"})
    {
        INFO(text);
        double value = 0.0;
        const auto result = parseDouble(text.data(), text.data() + text.size(), value);
        CHECK(result.ec == std::errc());
        CHECK(result.ptr == text.data() + text.size());
        CHECK_FALSE(std::isfinite(value));
        if (text.front() == '-')
            CHECK(std::signbit(value));
    }
}
