"""Builds the shareable package of the Rust host (caelix-app) as a folder and a zip.

The package holds caelix-app.exe, caelix_renderer.dll, the release DLSS DLLs, README.txt and
NOTICE.txt (from tools/package/, CRLF), and licenses/: the NVIDIA, C++ and Rust crate license
texts. The Rust list comes from cargo-about (`cargo install cargo-about --locked --features cli`)
with host-rs/licenses/about.toml and about.hbs.

The exe and the DLLs are taken from an existing release build; this script does not build them.
It refuses DLSS DLLs that are not the SDK's release (rel) files: the dev files show a watermark
and must never ship.

Usage: python tools/package_host.py [--bin DIR] [--deps DIR] [--ngx DIR] [--out DIR] [--name NAME]
"""

import argparse
import datetime
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST = ROOT / "host-rs"
NRD_VERSION = "4.18.0"
NGX_VERSION = "310.9.16"
BINARIES = ["caelix-app.exe", "caelix_renderer.dll", "nvngx_dlss.dll", "nvngx_dlssd.dll"]
DLSS_DLLS = ["nvngx_dlss.dll", "nvngx_dlssd.dll"]
# (source path under the dependency directory, name in licenses/)
CPP_LICENSES = [
    ("nrd-src/LICENSE.txt", "NVIDIA-NRD-LICENSE.txt"),
    ("shadermake-src/LICENSE.txt", "ShaderMake-LICENSE.txt"),
    ("mathlib-src/LICENSE.txt", "MathLib-LICENSE.txt"),
    ("volk-src/LICENSE.md", "volk-LICENSE.md"),
    ("vulkanmemoryallocator-src/LICENSE.txt", "VulkanMemoryAllocator-LICENSE.txt"),
    ("unordered_dense-src/LICENSE", "unordered_dense-LICENSE.txt"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_crlf(text: str, path: Path) -> None:
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\r\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=HOST / "target" / "gi" / "release",
                        help="directory of the built exe and DLLs")
    parser.add_argument("--deps", type=Path, default=ROOT / "build-gi" / "_deps",
                        help="CMake _deps directory of the library build (NRD and the MIT dependencies)")
    parser.add_argument("--ngx", type=Path, default=ROOT / "build" / "_deps" / "ngx-src",
                        help="the DLSS SDK source (LICENSE.txt, lib/Windows_x86_64/rel)")
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="output directory")
    parser.add_argument("--name", default=f"caelix-viewer-{datetime.date.today():%Y%m%d}",
                        help="package folder and zip name")
    args = parser.parse_args()

    for name in BINARIES:
        if not (args.bin / name).is_file():
            sys.exit(f"missing {args.bin / name}: build the host first")
    for name in DLSS_DLLS:
        release = args.ngx / "lib" / "Windows_x86_64" / "rel" / name
        if sha256(args.bin / name) != sha256(release):
            sys.exit(f"{args.bin / name} is not the SDK release file {release} (a dev DLL must never ship)")

    package = args.out / args.name
    if package.exists():
        shutil.rmtree(package)
    licenses = package / "licenses"
    licenses.mkdir(parents=True)

    for name in BINARIES:
        shutil.copy2(args.bin / name, package / name)
    for source, name in CPP_LICENSES:
        shutil.copy2(args.deps / source, licenses / name)
    shutil.copy2(args.ngx / "LICENSE.txt", licenses / "NVIDIA-DLSS-LICENSE.txt")
    shutil.copy2(ROOT / "LICENSE", licenses / "Caelix-Renderer-LICENSE.txt")

    # cargo-about refuses a redirected stdout on Windows: it writes the file itself.
    crates = licenses / "rust-crates.txt"
    subprocess.run(
        ["cargo", "about", "generate", "-c", "licenses/about.toml", "-m", "caelix-app/Cargo.toml",
         "-o", str(crates), "licenses/about.hbs"],
        cwd=HOST, check=True, capture_output=True,
    )
    write_crlf(crates.read_text(encoding="utf-8"), crates)

    template = ROOT / "tools" / "package"
    write_crlf((template / "README.txt").read_text(encoding="utf-8"), package / "README.txt")
    notice = (template / "NOTICE.txt").read_text(encoding="utf-8")
    notice = notice.replace("{nrd_version}", NRD_VERSION).replace("{ngx_version}", NGX_VERSION)
    write_crlf(notice, package / "NOTICE.txt")

    archive = shutil.make_archive(str(args.out / args.name), "zip", root_dir=args.out, base_dir=args.name)
    print(f"package: {package}")
    print(f"zip:     {archive} ({Path(archive).stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
