fn main() {
    // Link libvlc. Prefer pkg-config (portable); otherwise fall back to the
    // libs extracted into vendor/vlc/lib (see scripts: no system VLC install
    // needed) and the usual Homebrew prefixes.
    if pkg_config::Config::new().probe("libvlc").is_err() {
        println!("cargo:rustc-link-lib=vlc");
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        let vendor = format!("{manifest}/vendor/vlc/lib");
        for p in [vendor.as_str(), "/opt/homebrew/lib", "/usr/local/lib"] {
            if std::path::Path::new(p).exists() {
                println!("cargo:rustc-link-search=native={p}");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
            }
        }
    }

    // The Apple Event + run-loop calls used for "Open With" file associations.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=CoreServices");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        // CATransform3DMakeRotation / CATransaction for rotating the video view.
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=UniformTypeIdentifiers");
    }
}
