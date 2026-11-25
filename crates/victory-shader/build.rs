use std::fs;
use std::path::{Path, PathBuf};

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs" || e == "toml") {
            out.push(path);
        }
    }
}

fn fnv1a_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = manifest_dir.join("src");
    let cargo_toml = manifest_dir.join("Cargo.toml");

    println!("cargo:rerun-if-changed={}", src_dir.display());
    println!("cargo:rerun-if-changed={}", cargo_toml.display());

    let mut files = vec![cargo_toml];
    collect_files(&src_dir, &mut files);
    files.sort();

    // Two independent 64-bit FNV-1a lanes (128-bit fingerprint) over sorted relative paths + bytes.
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h2: u64 = 0x8422_2325_cbf2_9ce4;
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let rel = path
            .strip_prefix(&manifest_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        h1 = fnv1a_update(h1, rel.as_bytes());
        h1 = fnv1a_update(h1, &[0]);
        h2 = fnv1a_update(h2, &[0xff]);
        h2 = fnv1a_update(h2, rel.as_bytes());
        if let Ok(bytes) = fs::read(&path) {
            h1 = fnv1a_update(h1, &bytes);
            h1 = fnv1a_update(h1, &[0xfe]);
            h2 = fnv1a_update(h2, &(bytes.len() as u64).to_le_bytes());
            h2 = fnv1a_update(h2, &bytes);
        }
    }
    println!("cargo:rustc-env=VICTORY_SHADER_FINGERPRINT={h1:016x}{h2:016x}");
}
