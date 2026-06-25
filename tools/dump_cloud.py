#!/usr/bin/env python3
"""Extract sensor_msgs/PointCloud2 messages from a rosbag2 .db3 and dump each
cloud as a flat binary of float32 x,y,z triplets (NaN/inf points dropped).

Binary format per output file:
    int32   n                 number of valid points
    float32 xyz[n*3]          x0,y0,z0, x1,y1,z1, ...

No ROS install needed -- decodes CDR directly.

Usage:
    python3 dump_cloud.py BAG.db3 --topic /.../cloud_registered --out-dir data --max 2
"""
import argparse
import math
import sqlite3
import struct
import sys


class Reader:
    """Aligned little-endian CDR reader (offsets relative to body start)."""

    def __init__(self, body):
        self.b = body
        self.o = 0

    def align(self, n):
        r = self.o % n
        if r:
            self.o += n - r

    def u32(self):
        self.align(4)
        v = struct.unpack_from("<I", self.b, self.o)[0]
        self.o += 4
        return v

    def i32(self):
        self.align(4)
        v = struct.unpack_from("<i", self.b, self.o)[0]
        self.o += 4
        return v

    def u8(self):
        v = self.b[self.o]
        self.o += 1
        return v

    def string(self):
        n = self.u32()
        v = self.b[self.o : self.o + n - 1].decode("ascii", "ignore")
        self.o += n
        return v


def parse_header(r):
    sec = r.i32()
    nsec = r.u32()
    frame = r.string()
    height = r.u32()
    width = r.u32()
    nfields = r.u32()
    fields = []
    for _ in range(nfields):
        name = r.string()
        off = r.u32()
        dt = r.u8()
        cnt = r.u32()
        fields.append((name, off, dt, cnt))
    is_big = r.u8()
    point_step = r.u32()
    row_step = r.u32()
    data_len = r.u32()
    data = r.b[r.o : r.o + data_len]
    return {
        "frame": frame,
        "height": height,
        "width": width,
        "fields": fields,
        "point_step": point_step,
        "data": data,
    }


def extract_xyz(meta):
    fields = {f[0]: f[1] for f in meta["fields"]}
    ox, oy, oz = fields["x"], fields["y"], fields["z"]
    step = meta["point_step"]
    data = meta["data"]
    n = meta["height"] * meta["width"]
    out = []
    for i in range(n):
        base = i * step
        x = struct.unpack_from("<f", data, base + ox)[0]
        y = struct.unpack_from("<f", data, base + oy)[0]
        z = struct.unpack_from("<f", data, base + oz)[0]
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            out.append((x, y, z))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--topic", default="/barracuda/zed_node/point_cloud/cloud_registered")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--max", type=int, default=2, help="how many clouds to dump")
    ap.add_argument("--info", action="store_true", help="print first cloud metadata and exit")
    args = ap.parse_args()

    con = sqlite3.connect(args.bag)
    cur = con.cursor()
    row = cur.execute("SELECT id FROM topics WHERE name=?", (args.topic,)).fetchone()
    if not row:
        sys.exit(f"topic not found: {args.topic}")
    tid = row[0]
    rows = cur.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT ?",
        (tid, args.max),
    ).fetchall()

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    for i, (blob,) in enumerate(rows):
        meta = parse_header(Reader(blob[4:]))  # strip 4-byte encapsulation header
        if args.info and i == 0:
            print("frame_id:", meta["frame"])
            print("height,width:", meta["height"], meta["width"], "=> points:", meta["height"] * meta["width"])
            print("point_step:", meta["point_step"])
            print("fields:")
            for f in meta["fields"]:
                print("  ", f)
            return
        pts = extract_xyz(meta)
        path = os.path.join(args.out_dir, f"cloud{i}.bin")
        with open(path, "wb") as fh:
            fh.write(struct.pack("<i", len(pts)))
            for x, y, z in pts:
                fh.write(struct.pack("<fff", x, y, z))
        print(f"wrote {path}: {len(pts)} valid points (of {meta['height']*meta['width']})")


if __name__ == "__main__":
    main()
