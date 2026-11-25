fn main() {
    println!("cargo:rerun-if-changed=src/seh_copy.c");
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        cc::Build::new()
            .file("src/seh_copy.c")
            .opt_level(3)
            .compile("victory_seh_copy");
    }
}
