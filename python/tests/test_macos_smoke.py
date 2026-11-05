"""First-milestone acceptance: edited-mesh roundtrip, shaded capture, and a real compute bake.

Run in a disposable process: lilabatch --batch python/tests/test_macos_smoke.py
Creates temporary output and prints its location; raises on missing GPU or CPU fallback.
The native interactive viewport still needs the manual checks in README.md.
"""

import json
import os
from pathlib import Path
import struct
import tempfile
import zlib

def check_png(path, width, height, reference=None):
    """Decode the non-interlaced 8-bit PNGs Lila writes; optionally reject a background-only capture."""
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    assert struct.unpack(">II", data[16:24]) == (width, height), path
    assert len(data) > 64, path
    bits, color, compression, filtering, interlace = data[24:29]
    assert bits == 8 and color in (0, 2, 4, 6) and (compression, filtering, interlace) == (0, 0, 0), path
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    pos, compressed = 8, []
    while pos < len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        if data[pos + 4:pos + 8] == b"IDAT":
            compressed.append(data[pos + 8:pos + 8 + size])
        pos += size + 12
    raw = zlib.decompress(b"".join(compressed))
    stride = width * channels
    assert len(raw) == height * (stride + 1), path
    previous, pixels = bytearray(stride), []
    for y in range(height):
        start = y * (stride + 1)
        method, row = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        assert method in range(5), path
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            up = previous[x]
            upper_left = previous[x - channels] if x >= channels else 0
            if method == 1:
                row[x] = (row[x] + left) & 255
            elif method == 2:
                row[x] = (row[x] + up) & 255
            elif method == 3:
                row[x] = (row[x] + (left + up) // 2) & 255
            elif method == 4:
                prediction = left + up - upper_left
                nearest = min((left, up, upper_left), key=lambda value: abs(prediction - value))
                row[x] = (row[x] + nearest) & 255
        for x in range(0, stride, channels):
            pixels.append(tuple(row[x:x + 3]) if channels >= 3 else (row[x],) * 3)
        previous = row
    if reference is not None:
        empty = check_png(reference, width, height)
        changed = sum(max(abs(a - b) for a, b in zip(pixel, background)) > 8 for pixel, background in zip(pixels, empty))
        assert changed >= max(1, width * height // 100), "Shaded capture contains no visible mesh relative to the empty scene"
    return pixels


def main():
    # Force the portable GPU path even on machines with hardware ray queries.
    os.environ["LILA_VK_NO_RT"] = "1"
    os.environ["LILA_BAKE_GPU_TRACE"] = "sw"
    import lila
    from lila import bake, ops

    output = Path(tempfile.mkdtemp(prefix="lila_macos_smoke_"))
    print(f"Smoke artifacts: {output}", flush=True)
    ops.newScene()
    empty = output / "empty.png"
    ops.captureView(path=str(empty), width=256, height=256, display="shaded", view="three-quarter", grid=False, label=False)
    cube = ops.polyCube(name="MacCube", width=10, height=10, depth=10)[0]
    ops.move(cube, by=[10, 20, 30])
    scene = output / "edited-cube.lila"
    ops.saveScene(str(scene))
    assert scene.stat().st_size > 0
    ops.newScene()
    ops.openScene(str(scene), force=True)
    assert lila.node(cube).exists
    assert tuple(lila.node(cube).translate) == (10, 20, 30)
    ops.select(cube)
    capture = output / "shaded-cube.png"
    ops.captureView(path=str(capture), width=256, height=256, display="shaded", view="three-quarter",
                    frameSelection=True, grid=False, label=False)
    check_png(capture, 256, 256, reference=empty)

    ops.newScene()
    low = ops.polyPlane(width=100.0, depth=100.0, name="tile_low")[0]
    high = ops.polyPlane(width=100.0, depth=100.0, name="tile_high")[0]
    ops.move(high, by=[0, 1, 0])
    project = bake.project("MacBake", folder=str(output / "bake"))
    ops.setBakeOutput(project=project, resolution=16, supersampling=1, format="png", bits=8, name="{group}_{map}")
    ops.autoGroupBake(low, high, project=project)
    ops.bakeGroup("tile_bake", frontal=3.0, rear=3.0)
    result = bake.bake(project, maps=["normal", "height"], backend="gpu")
    assert result["backendMode"] == "compute", result
    assert result["stats"]["samples"] > 0, result
    assert result["stats"]["misses"] == 0, result
    assert {item["map"] for item in result["outputs"]} == {"normal", "height"}, result
    for item in result["outputs"]:
        check_png(item["file"], 16, 16)
    (output / "bake-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS: mesh roundtrip, shaded capture, GPU compute bake. Artifacts: {output}")


if __name__ == "__main__":
    main()
