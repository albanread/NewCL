// build.rs — ncl-driver
//
// When the `gui-app` feature is active (i.e. we are building the
// packaged release binary), embed a Win32 application manifest that
// declares:
//   • Per-Monitor DPI awareness (V2)
//   • Windows Vista / 7 / 8 / 8.1 / 10 / 11 compatibility
//   • asInvoker execution level (no UAC elevation prompt)
//   • UTF-8 active code page
//
// On non-Windows hosts the embed-manifest crate is a no-op, so this
// build script is safe to compile everywhere.

fn main() {
    #[cfg(feature = "gui-app")]
    embed_manifest::embed_manifest(embed_manifest::new_manifest("NewCormanLisp"))
        .expect("failed to embed application manifest");
}
