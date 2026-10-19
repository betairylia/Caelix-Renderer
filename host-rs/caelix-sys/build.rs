use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = strip_verbatim(
        manifest_dir
            .join("..")
            .join("..")
            .canonicalize()
            .expect("cannot resolve the repository root from CARGO_MANIFEST_DIR"),
    );

    // Native library location: CAELIX_LIB_DIR, or the CMake Release output.
    let lib_dir = match env::var_os("CAELIX_LIB_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => repo_root.join("build").join("bin").join("Release"),
    };
    let import_lib = lib_dir.join("caelix_renderer.lib");
    if !import_lib.exists() {
        panic!(
            "caelix_renderer.lib not found in {}.\n\
             Build the native library first, from the repository root:\n\
             \x20   cmake --preset vs2022\n\
             \x20   cmake --build --preset release\n\
             or set CAELIX_LIB_DIR to the directory that contains caelix_renderer.lib.",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=caelix_renderer");

    let header = repo_root.join("include").join("caelix").join("caelix.h");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", import_lib.display());
    println!("cargo:rerun-if-env-changed=CAELIX_LIB_DIR");

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_function("caelix_.*")
        .allowlist_type("Caelix.*")
        .allowlist_var("CAELIX_.*")
        .use_core()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate bindings for caelix.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("cannot write bindings.rs");

    // Cargo does not manage runtime DLLs. The usual workaround: copy the DLL
    // next to the executables of the current profile, so `cargo run` and
    // `cargo test` find it. OUT_DIR is <target>/<profile>/build/<pkg>-<hash>/out,
    // so three levels up is <target>/<profile>. The NGX feature DLLs (DLSS SR
    // and RR) go along when the library was built with them: NGX loads them
    // from the directory of caelix_renderer.dll.
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout")
        .to_path_buf();
    for name in [
        "caelix_renderer.dll",
        "caelix_renderer.pdb",
        "nvngx_dlss.dll",
        "nvngx_dlssd.dll",
    ] {
        let src = lib_dir.join(name);
        if src.exists() {
            println!("cargo:rerun-if-changed={}", src.display());
            fs::copy(&src, profile_dir.join(name))
                .unwrap_or_else(|e| panic!("cannot copy {}: {e}", src.display()));
        }
    }
}

/// `canonicalize` on Windows returns a verbatim path (`\\?\F:\...`). Strip the
/// prefix so the paths passed to cargo, rustc and libclang stay plain.
fn strip_verbatim(path: PathBuf) -> PathBuf {
    match path.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(plain) => PathBuf::from(plain),
        None => path,
    }
}
