# Third-party notices

Lila includes or links the following third-party components. Each is licensed under its own terms, which govern that component. The full license texts are in each component's distribution (vcpkg installs them to `vcpkg_installed/<triplet>/share/<port>/copyright`) and must ship with Lila's binaries.

## Copyleft (weak)

| Component | License | Copyright | Source |
|---|---|---|---|
| Qt 6 (Core, Gui, Widgets, Svg, Network, Xml, Sql, PrintSupport, OpenGL, OpenGLWidgets, UiTools) | LGPL-3.0 | The Qt Company Ltd. and other contributors | https://www.qt.io |
| PySide6 | LGPL-3.0 | The Qt Company Ltd. | https://www.qt.io/qt-for-python |
| Eigen | MPL-2.0 | Eigen contributors | https://eigen.tuxfamily.org |
| FFmpeg (libavcodec, libavformat, libavutil, libswscale, libswresample) | LGPL-2.1-or-later | The FFmpeg developers | https://ffmpeg.org |

Qt and PySide6 are dynamically linked and unmodified. You may replace the Qt and PySide6 libraries shipped with Lila with your own builds of the same versions. Their source code is available from https://download.qt.io. Eigen is used unmodified; its source is available from https://gitlab.com/libeigen/eigen.

FFmpeg 9.0.2 is dynamically linked (avcodec, avformat, avutil, swscale and swresample DLLs beside the executable) and unmodified, built without its GPL and nonfree parts and without its H.264 and HEVC decoders (Windows decodes those through Media Foundation). You may replace those DLLs with your own builds of a compatible version. The corresponding source code is available from https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz, and on request from Lila's maintainers for three years from the date you received this release.

## Permissive

| Component | License | Copyright | Source |
|---|---|---|---|
| Python | PSF-2.0 | Python Software Foundation | https://www.python.org |
| oneTBB | Apache-2.0 | Intel Corporation | https://github.com/uxlfoundation/oneTBB |
| hwloc | BSD-3-Clause | The hwloc contributors (Indiana University, University of Tennessee, HLRS Stuttgart, Inria and others) | https://github.com/open-mpi/hwloc |
| Alembic | BSD-3-Clause | Lucasfilm Entertainment Company Ltd. and Sony Pictures Imageworks Inc. | https://github.com/alembic/alembic |
| OpenUSD (with the components it bundles: RapidJSON, double-conversion, PEGTL and others, listed in its notice) | Tomorrow Open Source Technology License 1.0 (Apache-2.0 with a different trademark section); bundled components under their own permissive licenses | Pixar Animation Studios | https://github.com/PixarAnimationStudios/OpenUSD |
| Draco | Apache-2.0 | Google LLC | https://github.com/google/draco |
| Imath | BSD-3-Clause | Contributors to the OpenEXR Project | https://github.com/AcademySoftwareFoundation/Imath |
| Manifold | Apache-2.0 | The Manifold Authors | https://github.com/elalish/manifold |
| Clipper2 | BSL-1.0 | Angus Johnson | https://github.com/AngusJohnson/Clipper2 (also built with Z values from `src/curve/third_party/clipper2`) |
| Slang | Apache-2.0 WITH LLVM-exception | Slang contributors | https://github.com/shader-slang/slang |
| Vulkan-Headers | Apache-2.0 OR MIT | The Khronos Group Inc. | https://github.com/KhronosGroup/Vulkan-Headers |
| Vulkan Memory Allocator | MIT | Advanced Micro Devices, Inc. | https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator |
| volk | MIT | Arseny Kapoulkine | https://github.com/zeux/volk |
| meshoptimizer | MIT | Arseny Kapoulkine | https://github.com/zeux/meshoptimizer |
| Jolt Physics | MIT | Jorrit Rouwe | https://github.com/jrouwe/JoltPhysics |
| miniaudio | MIT-0 OR Unlicense | David Reid | https://github.com/mackron/miniaudio |
| GLM | MIT | G-Truc Creation | https://github.com/g-truc/glm |
| fast_float (macOS) | MIT | The fast_float authors | https://github.com/fastfloat/fast_float |
| nlohmann/json | MIT | Niels Lohmann | https://github.com/nlohmann/json |
| cpp-httplib | MIT | yhirose | https://github.com/yhirose/cpp-httplib |
| Brotli | MIT | The Brotli Authors | https://github.com/google/brotli |
| OpenSSL | Apache-2.0 | The OpenSSL Project Authors | https://www.openssl.org |
| MD4C (Qt's Markdown reader) | MIT | Martin Mitáš | https://github.com/mity/md4c |
| SQLite (Qt's SQLite driver) | Public domain | D. Richard Hipp and the SQLite developers | https://sqlite.org |
| Qt.py | MIT | Marcus Ottosson and contributors | https://github.com/mottosso/Qt.py |
| ufbx | MIT OR Unlicense | Samuli Raivio | https://github.com/ufbx/ufbx |
| cgltf | MIT | Johannes Kuhlmann | `third_party/cgltf` (licence in `third_party/cgltf/LICENSE`), from https://github.com/jkuhlmann/cgltf at v1.15 |
| stb_image | MIT OR Unlicense | Sean Barrett | `third_party/stb/stb_image.h` (licence at the end of the file), from https://github.com/nothings/stb, v2.30 |
| bcdec | MIT OR Unlicense | Sergii Kudlai | `third_party/bcdec/bcdec.h` (licence at the end of the file), from https://github.com/iOrange/bcdec, v0.985 |
| libwebp | BSD-3-Clause | Google Inc. | https://chromium.googlesource.com/webm/libwebp |
| nanobind | BSD-3-Clause | Wenzel Jakob | https://github.com/wjakob/nanobind |
| Zstandard | BSD-3-Clause (dual GPL-2.0, BSD chosen) | Meta Platforms, Inc. and affiliates | https://github.com/facebook/zstd |
| Tracy Profiler | BSD-3-Clause | Bartosz Taudul | https://github.com/wolfpld/tracy |
| Capstone (tracy-capture) | BSD-3-Clause | Nguyen Anh Quynh | https://github.com/capstone-engine/capstone |
| doctest (tests only, not distributed) | MIT | Viktor Kirilov | https://github.com/doctest/doctest |
| MikkTSpace | Zlib | Morten S. Mikkelsen | `src/geo/third_party/mikktspace` |
| Dem Bones | BSD-3-Clause | Electronic Arts Inc. | `third_party/dem-bones` (licence in `third_party/dem-bones/LICENSE.md`), from https://github.com/electronicarts/dem-bones at 09b899b |
| stb_truetype | MIT OR Unlicense | Sean Barrett | `src/curve/third_party/stb` |
| NanoSVG | Zlib | Mikko Mononen | `src/curve/third_party/nanosvg` |

## Assets

| Asset | License | Copyright | Source |
|---|---|---|---|
| Archivo (Latin static instances) | OFL-1.1 | The Archivo Project Authors | `resources/fonts/Archivo-LICENSE.txt` |
| Inter | OFL-1.1 | The Inter Project Authors | `resources/fonts/Inter-LICENSE.txt` |
| JetBrains Mono | OFL-1.1 | The JetBrains Mono Project Authors | `resources/fonts/JetBrainsMono-LICENSE.txt` |
| Tabler Icons | MIT | Paweł Kuna | `resources/icons/tabler/LICENSE` |
| Studio Small 09 HDRI | CC0-1.0 | Sergej Majboroda, Poly Haven | https://polyhaven.com/a/studio_small_09 |

## Training data

The auto-pose model weights are trained on the motion capture data below. The data itself is not distributed with Lila. Terms as retrieved are recorded in `tools/autopose/DATA-PROVENANCE.md`.

| Dataset | License | Copyright | Source |
|---|---|---|---|
| CMU Graphics Lab Motion Capture Database | Free for all uses; may be included in commercially sold products; the data may not be resold, even converted | Carnegie Mellon University | http://mocap.cs.cmu.edu |
| Open Motion Project | CC-BY-3.0 (https://creativecommons.org/licenses/by/3.0/) | ACCAD, The Ohio State University | https://accad.ohio-state.edu/research/motion-lab/mocap-system-and-data |

The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

Open Motion Project by ACCAD/The Ohio State University is licensed under a Creative Commons Attribution 3.0 Unported License.
