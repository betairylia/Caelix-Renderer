"""Builds the shareable package of the Unity native plugin as a folder and a zip.

The package holds caelix_unity.dll, caelix_renderer.dll, the release DLSS DLLs, README.txt
and NOTICE.txt (from tools/package/README-plugin.txt and NOTICE.txt, CRLF), and licenses/:
the Caelix-Renderer MIT license and the NVIDIA and C++ dependency license texts.

The plugin DLLs are taken from an existing release build (CMake preset `release`); this
script does not build them. The DLSS DLLs come from the SDK's release (rel) directory and
are checked by hash against the SDK; the dev files show a watermark and must never ship.

Usage: python tools/package_plugin.py [--bin DIR] [--deps DIR] [--ngx DIR] [--out DIR] [--name NAME]
"""

import argparse
import datetime
import shutil
import sys
from pathlib import Path

from package_host import CPP_LICENSES, NGX_VERSION, NRD_VERSION, DLSS_DLLS, sha256, write_crlf

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_BINARIES = ["caelix_unity.dll", "caelix_renderer.dll"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=ROOT / "build" / "bin" / "Release",
                        help="directory of the built plugin DLLs")
    parser.add_argument("--deps", type=Path, default=ROOT / "build" / "_deps",
                        help="CMake _deps directory of the library build (NRD and the MIT dependencies)")
    parser.add_argument("--ngx", type=Path, default=ROOT / "build" / "_deps" / "ngx-src",
                        help="the DLSS SDK source (LICENSE.txt, lib/Windows_x86_64/rel)")
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="output directory")
    parser.add_argument("--name", default=f"caelix-unity-plugin-{datetime.date.today():%Y%m%d}",
                        help="package folder and zip name")
    args = parser.parse_args()

    package = args.out / args.name
    if package.exists():
        shutil.rmtree(package)
    licenses = package / "licenses"
    licenses.mkdir(parents=True)

    for name in PLUGIN_BINARIES:
        source = args.bin / name
        if not source.is_file():
            print(f"missing: {source}", file=sys.stderr)
            return 1
        shutil.copy2(source, package / name)

    rel = args.ngx / "lib" / "Windows_x86_64" / "rel"
    for name in DLSS_DLLS:
        source = rel / name
        if not source.is_file():
            print(f"missing: {source}", file=sys.stderr)
            return 1
        shutil.copy2(source, package / name)
        if sha256(package / name) != sha256(source):
            print(f"hash mismatch after copy: {name}", file=sys.stderr)
            return 1

    for source, name in CPP_LICENSES:
        shutil.copy2(args.deps / source, licenses / name)
    shutil.copy2(args.ngx / "LICENSE.txt", licenses / "NVIDIA-DLSS-LICENSE.txt")
    shutil.copy2(ROOT / "LICENSE", licenses / "Caelix-Renderer-LICENSE.txt")

    write_crlf((ROOT / "tools" / "package" / "README-plugin.txt").read_text(encoding="utf-8"),
               package / "README.txt")
    notice = (ROOT / "tools" / "package" / "NOTICE.txt").read_text(encoding="utf-8")
    notice = notice.replace("{nrd_version}", NRD_VERSION).replace("{ngx_version}", NGX_VERSION)
    # The viewer-only paragraph (Rust crates) does not apply to the plugin.
    cut = notice.find("caelix-app.exe contains")
    if cut > 0:
        notice = notice[:cut].rstrip() + "\n"
    write_crlf(notice, package / "NOTICE.txt")

    archive = shutil.make_archive(str(args.out / args.name), "zip", root_dir=args.out, base_dir=args.name)
    print(f"package: {package}")
    print(f"zip:     {archive} ({Path(archive).stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
