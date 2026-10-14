use std::ffi::CStr;
use std::process::ExitCode;

use caelix_sys::{caelix_get_version, caelix_version_string, CaelixVersion, CAELIX_ABI_VERSION};

fn main() -> ExitCode {
    let mut version = CaelixVersion { major: 0, minor: 0, patch: 0, abi: 0 };
    // SAFETY: `version` is a valid, writable CaelixVersion. The returned string
    // is a static NUL-terminated string owned by the library.
    let version_string = unsafe {
        caelix_get_version(&mut version);
        CStr::from_ptr(caelix_version_string())
    };

    println!(
        "caelix_renderer {} (abi {})",
        version_string.to_string_lossy(),
        version.abi
    );

    if version.abi != CAELIX_ABI_VERSION {
        eprintln!(
            "error: ABI mismatch: library reports abi {}, bindings expect abi {}",
            version.abi, CAELIX_ABI_VERSION
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
