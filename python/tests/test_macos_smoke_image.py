"""Stdlib-only regression checks for the smoke test's missing-mesh detection."""

from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from test_macos_smoke import check_png


def png(path, color, filter_type=0):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    row = bytes(color) * 4
    if filter_type == 1:
        row = row[:3] + bytes(9)  # Sub filter encodes repeated pixels as zero differences.
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress((bytes([filter_type]) + row) * 4))
    path.write_bytes(data + chunk(b"IEND", b""))


class CaptureCheck(unittest.TestCase):
    def test_identical_pixels_fail_even_with_different_png_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "empty.png", Path(directory) / "cube.png"
            png(a, (20, 30, 40))
            png(b, (20, 30, 40), filter_type=1)
            with self.assertRaises(AssertionError):
                check_png(b, 4, 4, reference=a)

    def test_dither_sized_changes_do_not_count_as_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "empty.png", Path(directory) / "cube.png"
            png(a, (20, 30, 40))
            png(b, (21, 31, 41))
            with self.assertRaises(AssertionError):
                check_png(b, 4, 4, reference=a)

    def test_visible_change_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "empty.png", Path(directory) / "cube.png"
            png(a, (20, 30, 40))
            png(b, (100, 120, 140), filter_type=1)
            check_png(b, 4, 4, reference=a)


if __name__ == "__main__":
    unittest.main()
