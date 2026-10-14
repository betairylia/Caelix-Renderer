//! Raw FFI bindings to the `caelix_renderer` C ABI (`include/caelix/caelix.h`).
//! Generated at build time by bindgen; see `build.rs`.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
