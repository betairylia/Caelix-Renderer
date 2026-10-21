"""Runs the GI experiment matrix with the Rust host and tabulates the metrics.

    python tools/gi_bench.py --config tools/gi_bench.json --host <caelix-app.exe> --out <dir>
        [--only <scene>] [--runs a,b,c] [--dry-run] [--force]

Per scene and run: host command line -> <out>/<scene>/<run>.{png,log}; metrics of every PNG
against the scene's reference PNG (tools/image_metrics.py); <out>/README.md + results.json.
The host prints `timings: trace_ms=<f> gi_ms=<f> light_ms=<f> denoise_ms=<f> composite_ms=<f>
upscale_ms=<f>`; when absent, the status line `trace X ms | denoise ...` gives trace_ms, else "n/a".
A scene's "pose" may be null: the host then starts at the pose stored in the world file (no --pose).
A scene's optional "reference" (a string of host arguments) replaces RUNS["reference"] for that scene,
e.g. a light NEE reference where the plain path tracer does not converge.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import image_metrics  # noqa: E402

RUNS = {
    "reference": "--mode ref --accumulate 2048 --gi pt",
    "restir_converged": "--mode ref --accumulate 512 --gi restir",
    "pt": "--mode pt --no-denoise --warmup 64 --gi pt",
    "restir": "--mode pt --no-denoise --warmup 64 --gi restir",
    "restir_visibility": "--mode pt --no-denoise --warmup 64 --gi restir --gi-visibility",
    "restir_unbiased": "--mode pt --no-denoise --warmup 64 --gi restir --gi-unbiased",
    "restir_temporal_only": "--mode pt --no-denoise --warmup 64 --gi restir --gi-no-spatial",
    "restir_spatial_only": "--mode pt --no-denoise --warmup 64 --gi restir --gi-no-temporal",
    # Light list (docs/LIGHT_LIST.md): next-event estimation on list lights.
    "pt_nee": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee",
    "pt_nee_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee",
    "pt_nee_nomis": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-no-mis",
    "pt_nee_nomis_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-no-mis",
    # Per-pixel ReSTIR DI on the light list (docs/RESTIR_DI.md).
    "di": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --gi-restir-di",
    "di_unbiased": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --gi-restir-di --gi-unbiased",
    "di_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --gi-restir-di",
    # Temporal reuse in the light cell fill (docs/LIGHT_LIST.md section 2.2; static scenes).
    "pt_nee_temporal": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal",
    "pt_nee_temporal_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal",
    "di_temporal": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --gi-restir-di --light-cell-temporal",
    # Push-back of shadow ray results into the light cells (docs/LIGHT_LIST.md section 2.2).
    "pt_nee_push": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal --light-cell-push",
    "pt_nee_push_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal --light-cell-push",
    "di_push": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --gi-restir-di --light-cell-temporal "
    "--light-cell-push",
    "di_push_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --gi-restir-di --light-cell-temporal "
    "--light-cell-push",
    # Per-vertex resampling of the light NEE (docs/LIGHT_LIST.md section 2.6) and the vertex MIS
    # proxy (docs/LIGHT_MIS.md "Vertex proxy").
    "pt_nee_rs1": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-nee-candidates 1",
    "pt_nee_rs4": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-nee-candidates 4",
    "pt_nee_rs4_vp": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-nee-candidates 4 --light-proxy-vertex",
    "pt_nee_rs16_vp": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-nee-candidates 16 --light-proxy-vertex",
    "pt_nee_rs1_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-nee-candidates 1",
    "pt_nee_rs4_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-nee-candidates 4",
    "pt_nee_rs4_vp_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-nee-candidates 4 --light-proxy-vertex",
    "pt_nee_rs16_vp_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-nee-candidates 16 --light-proxy-vertex",
    # Oriented light cells (docs/LIGHT_LIST.md section 2.7); since then a cell holds 8 reservoirs, so
    # pt_nee_rs1 and pt_nee_rs1_conv are the unoriented 8-reservoir runs.
    "pt_nee_or": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-cell-oriented",
    "pt_nee_or_k32": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-cell-oriented --light-candidates 32",
    "pt_nee_or_rs4": "--mode pt --no-denoise --warmup 64 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-cell-oriented --light-nee-candidates 4",
    "pt_nee_or_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-cell-oriented",
    "pt_nee_or_k32_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-cell-oriented --light-candidates 32",
    "pt_nee_or_rs4_conv": "--mode ref --accumulate 512 --gi pt --gi-light-nee --light-cell-temporal "
    "--light-cell-oriented --light-nee-candidates 4",
}

TIMINGS_RE = re.compile(r"timings:((?:\s+\w+=[-+0-9.eE]+)+)")
STATUS_RE = re.compile(r"trace\s+([0-9.]+)\s*ms")


def build_command(host, scene, run, size, out_dir):
    png = out_dir / scene["name"] / f"{run}.png"
    cmd = [str(host), "--world", scene["world"]]
    if scene.get("pose") is not None:
        cmd += ["--pose", scene["pose"]]
    cmd += ["--fov", str(scene["fov"]), "--size", size, "--no-input", "--exit-after-screenshot",
            "--screenshot", str(png)]
    args = scene["reference"] if run == "reference" and scene.get("reference") else RUNS[run]
    cmd += args.split()
    cmd += [str(a) for a in scene.get("extra", [])]
    return cmd, png


def parse_timings(log_text):
    """Last `timings:` line as a dict; falls back to the status line's trace time."""
    found = TIMINGS_RE.findall(log_text)
    if found:
        return {k: float(v) for k, v in (kv.split("=") for kv in found[-1].split())}
    status = STATUS_RE.findall(log_text)
    if status:
        return {"trace_ms": float(status[-1])}
    return {}


def fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only")
    p.add_argument("--runs", help="comma list of run names (the reference always runs first)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="rerun even when the PNG exists")
    args = p.parse_args()

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    host = pathlib.Path(args.host).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    size = config.get("size", "1920x1080")
    wanted = args.runs.split(",") if args.runs else list(RUNS)
    for r in wanted:
        if r not in RUNS:
            print(f"unknown run '{r}', known: {', '.join(RUNS)}", file=sys.stderr)
            return 1
    runs = ["reference"] + [r for r in wanted if r != "reference"]
    scenes = [s for s in config["scenes"] if not args.only or s["name"] == args.only]
    if not scenes:
        print("no scene selected", file=sys.stderr)
        return 1

    env = dict(os.environ, DISABLE_VULKAN_OW_OVERLAY_LAYER="1", DISABLE_VULKAN_OW_OBS_CAPTURE="1")
    results = {}
    commands = []
    for scene in scenes:
        if not args.dry_run:
            (out_dir / scene["name"]).mkdir(parents=True, exist_ok=True)
        ref_png = out_dir / scene["name"] / "reference.png"
        scene_res = {}
        for run in runs:
            cmd, png = build_command(host, scene, run, size, out_dir)
            line = " ".join(f'"{c}"' if " " in c else c for c in cmd)
            commands.append((scene["name"], run, line))
            if args.dry_run:
                print(f"[{scene['name']}/{run}] {line}")
                continue
            log = out_dir / scene["name"] / f"{run}.log"
            if png.exists() and not args.force:
                print(f"[{scene['name']}/{run}] skip, PNG exists")
                text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            else:
                print(f"[{scene['name']}/{run}] running")
                proc = subprocess.run(cmd, cwd=host.parent, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, errors="replace")
                text = proc.stdout
                log.write_text(text, encoding="utf-8")
                if proc.returncode != 0:
                    print(f"  host exit code {proc.returncode}, see {log}", file=sys.stderr)
            entry = {"png": str(png), "timings": parse_timings(text)}
            if png.exists() and ref_png.exists():
                entry["metrics"] = image_metrics.compute_files(ref_png, png)
            scene_res[run] = entry
        results[scene["name"]] = scene_res
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = ["# GI bench results", "", f"Host: `{host}`, size {size}.", ""]
    for scene in scenes:
        extra = " ".join(map(str, scene.get("extra", [])))
        md += [f"## {scene['name']}", "",
               f"World `{scene['world']}`, pose `{scene.get('pose') or 'stored in the file'}`, fov {scene['fov']}, "
               f"extra `{extra}`.", "",
               "| run | PSNR dB | mean ratio | firefly % | trace ms | gi ms | light ms |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for run in runs:
            e = results[scene["name"]].get(run, {})
            m, t = e.get("metrics"), e.get("timings", {})
            md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                run,
                fmt(m and m["psnr_db"], ".2f"), fmt(m and m["mean_ratio"], ".4f"),
                fmt(m and m["firefly_fraction"] * 100, ".3f"),
                fmt(t.get("trace_ms"), ".2f"), fmt(t.get("gi_ms"), ".2f"), fmt(t.get("light_ms"), ".2f")))
        md += ["", "Command lines:", ""]
        md += [f"- `{run}`: `{line}`" for s, run, line in commands if s == scene["name"]]
        md.append("")
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
