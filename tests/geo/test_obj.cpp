#include "TestUtil.h"

#include "core/Decimal.h"

#include "geo/Normals.h"
#include "geo/ObjIO.h"
#include "geo/Primitives.h"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

using namespace lila::geo;
using namespace lila::geo::test;

TEST_CASE("obj: parse basics")
{
    const char* text = R"(# test
mtllib test.mtl
o first
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
g top
usemtl red
s 1
f 1/1/1 2/2/1 3/3/1 4/4/1
usemtl blue
f -4/1/1 -1/2/1 -3/3/1
o second
v 5 5 5
v 6 5 5
v 6 6 5
f 6 7 8
f 6 7
)";
    ObjReadResult r = readObj(text);
    REQUIRE(r.ok());
    REQUIRE(r.objects.size() == 2);
    CHECK(r.mtllibs.size() == 1);
    const Mesh& a = r.objects[0].mesh;
    CHECK(r.objects[0].name == "first");
    CHECK(a.numFaces() == 2);
    CHECK(a.numPoints() == 5);
    CHECK(isValid(a));
    REQUIRE(a.materialNames().size() == 2);
    CHECK(a.materialNames()[1] == "blue");
    std::span<const int32_t> mat = a.faceAttrs().get<int32_t>(attr::kMaterial);
    CHECK(mat[0] == 0);
    CHECK(mat[1] == 1);
    CHECK(a.faceAttrs().has("group"));
    CHECK(r.objects[0].groupNames.at(0) == "top");
    std::span<const glm::vec2> uv = a.cornerAttrs().get<glm::vec2>("uv0");
    REQUIRE(uv.size() == 7);
    CHECK(uv[2] == glm::vec2(1, 1));
    CHECK(a.faceSize(1) == 3);

    const Mesh& b = r.objects[1].mesh;
    CHECK(r.objects[1].name == "second");
    CHECK(b.numFaces() == 1);
    CHECK(b.numPoints() == 3);
    CHECK(b.position(0).x == 5.0);
}

TEST_CASE("obj: round trip keeps topology, uvs, hard edges and materials")
{
    Mesh cyl = makeCylinder({.radius = 50.0, .height = 100.0, .subdivAxis = 12, .subdivHeight = 2, .capFill = CapFill::Ngon});
    std::span<int32_t> mat = cyl.faceAttrs().getOrAdd<int32_t>(attr::kMaterial);
    cyl.materialNames() = {"side", "cap"};
    for (Index f = 0; f < cyl.faceCapacity(); ++f)
        mat[f] = cyl.faceSize(f) > 4 ? 1 : 0;

    const std::string text = writeObj(cyl);
    ObjReadResult r = readObj(text);
    REQUIRE(r.objects.size() == 1);
    const Mesh& m = r.objects[0].mesh;
    CHECK(isValid(m));
    CHECK(m.numPoints() == cyl.numPoints());
    CHECK(m.numFaces() == cyl.numFaces());
    CHECK(m.numEdges() == cyl.numEdges());
    CHECK(hardCount(m) == hardCount(cyl));
    CHECK(m.materialNames().size() == 2);
    for (Index p = 0; p < m.pointCapacity(); ++p)
        CHECK(m.position(p) == cyl.position(p));
    std::span<const glm::vec2> uvA = std::as_const(cyl).cornerAttrs().get<glm::vec2>("uv0");
    std::span<const glm::vec2> uvB = m.cornerAttrs().get<glm::vec2>("uv0");
    for (Index c = 0; c < m.cornerCapacity(); ++c)
        CHECK(uvA[c] == uvB[c]);
    std::span<const int32_t> matB = m.faceAttrs().get<int32_t>(attr::kMaterial);
    for (Index f = 0; f < m.faceCapacity(); ++f)
        CHECK(m.materialNames()[matB[f]] == cyl.materialNames()[mat[f]]);

    // The text itself is stable.
    CHECK(writeObj(m) == text);
}

TEST_CASE("obj: hard edges from normals without smoothing groups")
{
    Mesh cube = makeCube();
    ObjWriteOptions wo;
    wo.smoothingGroups = false;
    ObjReadResult r = readObj(writeObj(cube, wo));
    REQUIRE(r.objects.size() == 1);
    CHECK(hardCount(r.objects[0].mesh) == 12);
}

TEST_CASE("obj: coordinates read exactly as from_chars reads them, whatever their formatting")
{
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> mag(-30.0, 30.0);
    const char* formats[] = {"%.6f", "%.9g", "%.17g", "%.3e", "%.15g", "%g", "%.0f", "%.12f"};
    std::vector<std::string> numbers = {"0", "-0", "1.", ".5", "-.25", "1e5", "1E-7", "123456789012345", "1234567890123456",
                                        "0.000000000000000000001", "1e22", "1e23", "1e-22", "1e-23", "007.50", "-0.0"};
    for (int i = 0; i < 3000; ++i)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), formats[i % 8], (i % 3 == 0 ? -1.0 : 1.0) * std::pow(10.0, mag(rng) / 5.0) * mag(rng));
        numbers.push_back(buf);
    }
    while (numbers.size() % 3)
        numbers.push_back("1");
    const size_t points = numbers.size() / 3;
    // Points no face uses are dropped, so every three points get a face.
    ObjReadOptions opts;
    std::string all;
    for (size_t i = 0; i < numbers.size(); i += 3)
        all += "v " + numbers[i] + " " + numbers[i + 1] + " " + numbers[i + 2] + "\n";
    for (size_t i = 0; i + 2 < points; i += 3)
        all += "f " + std::to_string(i + 1) + " " + std::to_string(i + 2) + " " + std::to_string(i + 3) + "\n";
    const ObjReadResult every = readObj(all, opts);
    REQUIRE(every.objects.size() == 1);
    const Mesh& m = every.objects[0].mesh;
    REQUIRE(size_t(m.pointCapacity()) == points - points % 3);
    for (Index p = 0; p < m.pointCapacity(); ++p)
        for (int c = 0; c < 3; ++c)
        {
            const std::string& s = numbers[size_t(p) * 3 + size_t(c)];
            const char* b = s.data() + (s[0] == '+' ? 1 : 0);
            double want = 0.0;
            lila::fromChars(b, s.data() + s.size(), want);
            INFO(s);
            CHECK(std::memcmp(&want, &m.position(p)[c], sizeof(double)) == 0);
        }
}

TEST_CASE("obj: lines without faces report an error")
{
    ObjReadResult r = readObj("v 0 0 0\r\nv 1e-15 0 0\r\nv 0 1 0\r\nv 0 0 1\r\nl 1 2\r\nl 3 4\r\n");
    CHECK_FALSE(r.ok());
    CHECK(r.objects.empty());
    CHECK(r.error.find("2 line elements") != std::string::npos);

    ObjReadResult mixed = readObj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nl 1 2\n");
    CHECK(mixed.ok());
    CHECK(mixed.objects.size() == 1);
}

TEST_CASE("perf: obj 1M faces")
{
    Mesh m = makePlane({1000.0, 1000.0, 1000, 1000});
    auto t0 = std::chrono::steady_clock::now();
    const std::string text = writeObj(m);
    const double tWrite = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    ObjReadResult r = readObj(text);
    const double tRead = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE(r.objects.size() == 1);
    CHECK(r.objects[0].mesh.numFaces() == 1000000);
    MESSAGE("obj 1M quads (" << text.size() / (1 << 20) << " MB): write " << tWrite << "s, read " << tRead << "s");
}
