"""Converts a host materials JSON of the experiment era to the production format.

    python -I tools/convert_materials_json.py <in.json> <out.json>  (in and out may be the same file)

- Drops the `sggx` object (closed experiment; the production host ignores unknown keys, but the
  file should not carry them).
- Sets `sss_model: 1` (CAELIX_SSS_MODEL_HG) on every entry whose sss_probability > 0 and
  sss_steps > 0. Entries with sss keys all zero lose them (they are the defaults).
- Keeps every other key and value as it is; writes one entry per line in a fixed key order.
- Checks the library ranges (caelix.h "Materials") and the id ranges of the host parser.
"""
import json
import math
import sys

ORDER = ["id", "transparent", "albedo", "smoothness", "metallic", "emission", "ior", "extinction",
         "sss_model", "sss_probability", "sss_albedo", "sss_anisotropy", "sss_steps", "sss_params",
         "texture_set", "connect_group"]
SSS_KEYS = ["sss_model", "sss_probability", "sss_albedo", "sss_anisotropy", "sss_steps", "sss_params"]


def check(entry):
    i = entry["id"]
    assert i <= (511 if entry.get("transparent") else 32767), f"id {i} out of range"
    for k in ("albedo", "emission", "sss_albedo"):
        if k in entry:
            assert len(entry[k]) == 3 and all(math.isfinite(v) and abs(v) <= 65504 for v in entry[k]), (i, k)
    assert entry.get("ior", 1.5) > 0, (i, "ior")
    assert entry.get("extinction", 0.0) >= 0, (i, "extinction")
    assert 0 <= entry.get("sss_probability", 0.0) <= 1, (i, "sss_probability")
    assert all(0 <= v <= 1 for v in entry.get("sss_albedo", [0, 0, 0])), (i, "sss_albedo")
    assert -1 <= entry.get("sss_anisotropy", 0.0) <= 1, (i, "sss_anisotropy")
    assert 0 <= entry.get("sss_steps", 0) <= 255, (i, "sss_steps")
    assert entry.get("sss_model", 0) <= 255, (i, "sss_model")


def convert(entry):
    out = dict(entry)
    dropped = []
    if "sggx" in out:
        dropped.append("sggx")
        del out["sggx"]
    unknown = [k for k in out if k not in ORDER]
    assert not unknown, f"id {out['id']}: unknown keys {unknown}"
    sss = out.get("sss_probability", 0.0) > 0 and out.get("sss_steps", 0) > 0
    if sss:
        out["sss_model"] = 1
    else:
        for k in SSS_KEYS:
            out.pop(k, None)
    check(out)
    return {k: out[k] for k in ORDER if k in out}, sss, dropped


def main():
    src, dst = sys.argv[1], sys.argv[2]
    entries = json.load(open(src, encoding="utf-8"))
    ids = [(bool(e.get("transparent")), e["id"]) for e in entries]
    assert len(ids) == len(set(ids)), "duplicate ids"
    result, n_sss, n_dropped = [], 0, 0
    for e in entries:
        c, sss, dropped = convert(e)
        result.append(c)
        n_sss += sss
        n_dropped += bool(dropped)
    lines = ",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in result)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write("[\n" + lines + "\n]\n")
    print(f"{src}: {len(result)} entries, {n_sss} SSS entries set to sss_model 1, "
          f"{n_dropped} sggx objects dropped -> {dst}")


if __name__ == "__main__":
    main()
