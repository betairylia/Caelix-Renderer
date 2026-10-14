"""Converts Titania's generated VoxelMaterials.hlsl (static MatTableOpaque / MatTableTransparent
arrays) into the materials JSON the host loads with --materials (block-ID encoding).

Usage: python -I tools/hlsl_materials_to_json.py <VoxelMaterials.hlsl> <out.json>

Mapping (Unity GetMaterial): MatTableOpaque[i] <-> code 0x8000 + i  -> {"id": i, "transparent": false}
                             MatTableTransparent[i] <-> code i      -> {"id": i, "transparent": true}
Entries are written verbatim, including Unity's magenta fallback rows for unused ids."""
import json, re, sys

ROW = re.compile(r"\{\{\s*([^}]*)\}\s*,\s*\{\s*([^}]*)\}\s*,\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^}]+)\}")

def parse_table(text, name):
    m = re.search(r"static VoxelMaterial %s\[(\d+)\]\s*=\s*\{(.*?)\n\};" % name, text, re.S)
    if not m:
        raise SystemExit(f"table {name} not found")
    count = int(m.group(1)); rows = []
    for r in ROW.finditer(m.group(2)):
        albedo = [float(v) for v in r.group(1).split(",")]
        emission = [float(v) for v in r.group(2).split(",")]
        rows.append((albedo, emission, float(r.group(3)), float(r.group(4)), float(r.group(5)), float(r.group(6))))
    if len(rows) != count:
        raise SystemExit(f"{name}: declared {count} rows, parsed {len(rows)}")
    return rows

def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8").read()
    out = []
    for name, transparent in (("MatTableOpaque", False), ("MatTableTransparent", True)):
        for i, (albedo, emission, smooth, metal, ior, ext) in enumerate(parse_table(text, name)):
            out.append({"id": i, "transparent": transparent, "albedo": albedo, "emission": emission,
                        "smoothness": smooth, "metallic": metal, "ior": ior, "extinction": ext})
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {len(out)} entries to {dst}")

if __name__ == "__main__":
    main()
