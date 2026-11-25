fn main() {
    #[cfg(target_os = "windows")]
    {
        // Export DirectX 12 Agility SDK symbols so Windows 10 (e.g. 19044 LTSC)
        // loads D3D12Core.dll from .\D3D12\ for Shader Model 6.6+ & dynamic resources.
        println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKPath,DATA");
    }
}
