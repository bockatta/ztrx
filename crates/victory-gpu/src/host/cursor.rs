//! Host on-screen arrow cursor rendered onto the swapchain backbuffer.
//!
//! Reads cursor coordinates and visibility from the per-process Win32 shared
//! memory mapping `Local\victory-cursor-<pid>` populated by `victory-hle`
//! (`gameinput.rs`). When `victory-gpu` runs standalone without `victory-hle`,
//! `VICTORY_CURSOR=1` enables a direct Win32 client-cursor fallback, and
//! `VICTORY_CURSOR=0` acts as a global kill-switch.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows::Win32::Foundation::RECT as WinRect;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, GetCursorPos};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE,
    UnmapViewOfFile,
};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_LBUTTON};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::super::{Host, HostList, retire_resource, transition};

const CURSOR_MAGIC: u32 = 0x5643_5552; // "VCUR"
const CURSOR_VERSION: u32 = 1;
const SHARED_SIZE: u32 = 64;

pub(super) const CURSOR_W: u32 = 12;
pub(super) const CURSOR_H: u32 = 18;

/// Classic crisp 12x18 arrow cursor (`#` = black outline, `@` = interior fill, `.` = transparent).
const CURSOR_ROWS: [&str; CURSOR_H as usize] = [
    "#...........",
    "##..........",
    "#@#.........",
    "#@@#........",
    "#@@@#.......",
    "#@@@@#......",
    "#@@@@@#.....",
    "#@@@@@@#....",
    "#@@@@@@@#...",
    "#@@@@@@@@#..",
    "#@@@@@@@@@#.",
    "#@@@@@@#####",
    "#@@@#@@#....",
    "#@@#.#@@#...",
    "#@#..#@@#...",
    "##....#@@#..",
    "#.....#@@#..",
    ".......##...",
];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct CursorSnapshot {
    pub(super) u: f32,
    pub(super) v: f32,
    pub(super) visible: bool,
    pub(super) pressed: bool,
    pub(super) cursor_mode: bool,
    pub(super) initialized: bool,
    pub(super) seq: u32,
}

#[repr(C)]
struct SharedCursor {
    magic: AtomicU32,
    version: AtomicU32,
    packed: AtomicU64,
}

const _: () = assert!(size_of::<SharedCursor>() <= SHARED_SIZE as usize);

#[cfg(test)]
fn pack_cursor(s: CursorSnapshot) -> u64 {
    let x_q16 = (s.u.clamp(0.0, 1.0) * 65535.0).round() as u64;
    let y_q16 = (s.v.clamp(0.0, 1.0) * 65535.0).round() as u64;
    let seq = u64::from(s.seq & 0x0FFF_FFFF);
    x_q16
        | (y_q16 << 16)
        | (u64::from(s.visible) << 32)
        | (u64::from(s.pressed) << 33)
        | (u64::from(s.cursor_mode) << 34)
        | (u64::from(s.initialized) << 35)
        | (seq << 36)
}

fn unpack_cursor(packed: u64) -> CursorSnapshot {
    let x_q16 = (packed & 0xFFFF) as f32;
    let y_q16 = ((packed >> 16) & 0xFFFF) as f32;
    CursorSnapshot {
        u: x_q16 / 65535.0,
        v: y_q16 / 65535.0,
        visible: (packed & (1 << 32)) != 0,
        pressed: (packed & (1 << 33)) != 0,
        cursor_mode: (packed & (1 << 34)) != 0,
        initialized: (packed & (1 << 35)) != 0,
        seq: (packed >> 36) as u32,
    }
}

struct SharedMapping {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
}

// SAFETY: `SharedCursor` contains only atomics in a process-lifetime pagefile mapping.
unsafe impl Send for SharedMapping {}
// SAFETY: as above.
unsafe impl Sync for SharedMapping {}

impl SharedMapping {
    fn open(name: &str) -> Option<Self> {
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: creating/opening a named 64-byte pagefile mapping with null-terminated UTF-16 name.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                SHARED_SIZE,
                wide.as_ptr(),
            )
        };
        if handle.is_null() {
            return None;
        }
        // SAFETY: mapping the 64-byte view we just opened/created.
        let view = unsafe { MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, SHARED_SIZE as usize) };
        if view.Value.is_null() {
            // SAFETY: valid handle from `CreateFileMappingW`.
            unsafe { CloseHandle(handle) };
            return None;
        }
        Some(Self { handle, view })
    }

    fn header(&self) -> &SharedCursor {
        // SAFETY: `view.Value` is non-null, aligned (>=4KB page), and at least 64 bytes.
        unsafe { &*self.view.Value.cast::<SharedCursor>() }
    }
}

impl Drop for SharedMapping {
    fn drop(&mut self) {
        // SAFETY: `view` and `handle` were successfully acquired in `SharedMapping::open`.
        unsafe {
            UnmapViewOfFile(self.view);
            CloseHandle(self.handle);
        }
    }
}

fn shared_mapping() -> Option<&'static SharedMapping> {
    static MAPPING: OnceLock<Option<SharedMapping>> = OnceLock::new();
    MAPPING
        .get_or_init(|| {
            let name = format!("Local\\victory-cursor-{}", std::process::id());
            SharedMapping::open(&name)
        })
        .as_ref()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CursorEnv {
    Default,
    Disabled,
    StandaloneOn,
}

fn cursor_env() -> CursorEnv {
    static ENV: OnceLock<CursorEnv> = OnceLock::new();
    *ENV.get_or_init(|| match std::env::var("VICTORY_CURSOR").ok().as_deref() {
        Some("0") => CursorEnv::Disabled,
        Some("1") => CursorEnv::StandaloneOn,
        _ => CursorEnv::Default,
    })
}

fn resolve_cursor_snapshot(
    env: CursorEnv,
    shared: Option<CursorSnapshot>,
    standalone_fallback: Option<(f32, f32, bool)>,
) -> CursorSnapshot {
    if env == CursorEnv::Disabled {
        return CursorSnapshot::default();
    }
    if let Some(snap) = shared
        && snap.initialized
    {
        return snap;
    }
    if env == CursorEnv::StandaloneOn
        && let Some((u, v, pressed)) = standalone_fallback
    {
        return CursorSnapshot {
            u: u.clamp(0.0, 1.0),
            v: v.clamp(0.0, 1.0),
            visible: true,
            pressed,
            cursor_mode: true,
            initialized: false,
            seq: 0,
        };
    }
    CursorSnapshot::default()
}

fn read_shared_snapshot(mapping: &SharedMapping) -> Option<CursorSnapshot> {
    let hdr = mapping.header();
    if hdr.magic.load(Ordering::Acquire) != CURSOR_MAGIC
        || hdr.version.load(Ordering::Acquire) != CURSOR_VERSION
    {
        return None;
    }
    Some(unpack_cursor(hdr.packed.load(Ordering::Acquire)))
}

fn query_standalone_cursor() -> Option<(f32, f32, bool)> {
    // SAFETY: plain Win32 query on foreground window ownership.
    let fg = unsafe { GetForegroundWindow() };
    if fg.is_null() {
        return None;
    }
    let mut pid = 0u32;
    // SAFETY: valid `fg` and output pointer.
    unsafe { GetWindowThreadProcessId(fg, &mut pid) };
    if pid != std::process::id() {
        return None;
    }
    let fg_win = windows::Win32::Foundation::HWND(fg);
    let mut pt = windows::Win32::Foundation::POINT::default();
    let mut rect = WinRect::default();
    // SAFETY: valid window handle and output structs.
    unsafe {
        if GetCursorPos(&mut pt).is_err() {
            return None;
        }
        if !ScreenToClient(fg_win, &mut pt).as_bool() {
            return None;
        }
        if GetClientRect(fg_win, &mut rect).is_err() {
            return None;
        }
    }
    let w = rect.right - rect.left;
    let h = rect.bottom - rect.top;
    if w <= 0 || h <= 0 || pt.x < 0 || pt.y < 0 || pt.x >= w || pt.y >= h {
        return None;
    }
    let u = (pt.x as f32 / w as f32).clamp(0.0, 1.0);
    let v = (pt.y as f32 / h as f32).clamp(0.0, 1.0);
    // SAFETY: plain key state query.
    let pressed = unsafe { GetAsyncKeyState(i32::from(VK_LBUTTON)) } < 0;
    Some((u, v, pressed))
}

fn current_snapshot() -> CursorSnapshot {
    let env = cursor_env();
    if env == CursorEnv::Disabled {
        return CursorSnapshot::default();
    }
    let shared = shared_mapping().and_then(read_shared_snapshot);
    let fallback = if shared.as_ref().is_some_and(|s| s.initialized) || env != CursorEnv::StandaloneOn {
        None
    } else {
        query_standalone_cursor()
    };
    resolve_cursor_snapshot(env, shared, fallback)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    y: u32,
    x0: u32,
    x1: u32,
}

fn logical_spans() -> &'static [Span] {
    static SPANS: OnceLock<Vec<Span>> = OnceLock::new();
    SPANS.get_or_init(|| {
        let mut out = Vec::new();
        for (y, row) in CURSOR_ROWS.iter().enumerate() {
            let bytes = row.as_bytes();
            let mut x = 0;
            while x < bytes.len() {
                if bytes[x] == b'.' {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < bytes.len() && bytes[x] != b'.' {
                    x += 1;
                }
                out.push(Span {
                    y: y as u32,
                    x0: start as u32,
                    x1: x as u32,
                });
            }
        }
        out
    })
}

/// `[outline, normal_fill, pressed_fill]` byte slices for each swapchain format.
fn palette(format: DXGI_FORMAT) -> Option<[&'static [u8]; 3]> {
    Some(match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => [
            &[0x00, 0x00, 0x00, 0xFF],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            &[0xFF, 0xD0, 0x20, 0xFF],
        ],
        DXGI_FORMAT_B8G8R8A8_UNORM => [
            &[0x00, 0x00, 0x00, 0xFF],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            &[0x20, 0xD0, 0xFF, 0xFF],
        ],
        DXGI_FORMAT_R10G10B10A2_UNORM => [
            &[0x00, 0x00, 0x00, 0xC0],
            &[0xFF, 0xFF, 0xFF, 0xFF],
            &[0xFF, 0x03, 0x0D, 0xC8], // A=3, B=128, G=832, R=1023 -> 0xC80D03FF LE
        ],
        DXGI_FORMAT_R16G16B16A16_FLOAT => [
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3C],
            &[0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C],
            &[0x00, 0x3C, 0x1A, 0x3A, 0x00, 0x30, 0x00, 0x3C],
        ],
        _ => return None,
    })
}

fn sprite_scale(height: u32) -> u32 {
    (height / 540).max(1)
}

/// Rasterizes both normal (`y: 0..sprite_h`) and pressed (`y: sprite_h..2*sprite_h`)
/// cursor variants into row-major rows of `pitch` bytes.
fn rasterize_atlas(scale: u32, pal: [&[u8]; 3], pitch: usize) -> Vec<u8> {
    let bpp = pal[0].len();
    let sprite_h = (CURSOR_H * scale) as usize;
    let mut out = vec![0u8; pitch * sprite_h * 2];
    for (variant, fill_idx) in [(0usize, 1usize), (1usize, 2usize)] {
        let base_row = variant * sprite_h;
        for (ly, row) in CURSOR_ROWS.iter().enumerate() {
            let row_top = (base_row + ly * scale as usize) * pitch;
            for (lx, &ch) in row.as_bytes().iter().enumerate() {
                let pixel = match ch {
                    b'#' => pal[0],
                    b'@' => pal[fill_idx],
                    _ => continue,
                };
                for sx in 0..scale as usize {
                    let off = row_top + (lx * scale as usize + sx) * bpp;
                    out[off..off + bpp].copy_from_slice(pixel);
                }
            }
            for sy in 1..scale as usize {
                out.copy_within(row_top..row_top + pitch, row_top + sy * pitch);
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlitRect {
    dst_x: u32,
    dst_y: u32,
    src_x: u32,
    src_y: u32,
    w: u32,
    h: u32,
}

fn plan_blits(u: f32, v: f32, pressed: bool, (width, height): (u32, u32), scale: u32) -> Vec<BlitRect> {
    if width == 0 || height == 0 || scale == 0 {
        return Vec::new();
    }
    let anchor_x = ((u.clamp(0.0, 1.0) * (width - 1) as f32).round() as u32).min(width - 1);
    let anchor_y = ((v.clamp(0.0, 1.0) * (height - 1) as f32).round() as u32).min(height - 1);
    let sprite_h = CURSOR_H * scale;
    let y_base = if pressed { sprite_h } else { 0 };
    let mut blits = Vec::with_capacity(logical_spans().len());
    for span in logical_spans() {
        let dst_x = anchor_x.saturating_add(span.x0 * scale);
        let dst_y = anchor_y.saturating_add(span.y * scale);
        if dst_x >= width || dst_y >= height {
            continue;
        }
        let copy_w = ((span.x1 - span.x0) * scale).min(width - dst_x);
        let copy_h = scale.min(height - dst_y);
        if copy_w == 0 || copy_h == 0 {
            continue;
        }
        blits.push(BlitRect {
            dst_x,
            dst_y,
            src_x: span.x0 * scale,
            src_y: y_base + span.y * scale,
            w: copy_w,
            h: copy_h,
        });
    }
    blits
}

struct Atlas {
    format: DXGI_FORMAT,
    scale: u32,
    pitch: u32,
    card: ID3D12Resource,
}

// SAFETY: D3D12 resources are thread-safe and mutated only under the presenter's lock.
unsafe impl Send for Atlas {}

impl Atlas {
    fn create_and_upload(
        host: &Host,
        list: &HostList,
        format: DXGI_FORMAT,
        scale: u32,
    ) -> Option<Self> {
        let pal = palette(format)?;
        let sprite_w = CURSOR_W * scale;
        let sprite_h = CURSOR_H * scale;
        let bpp = pal[0].len() as u32;
        let pitch = (sprite_w * bpp).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(sprite_h * 2);
        let pixels = rasterize_atlas(scale, pal, pitch as usize);
        let (upload, mapped) = host.upload_buffer(bytes)?;
        // SAFETY: `upload` is a freshly mapped upload buffer of `bytes >= pixels.len()`.
        unsafe { std::ptr::copy_nonoverlapping(pixels.as_ptr(), mapped, pixels.len()) };

        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: creating a `DEFAULT` buffer for the cached cursor atlas.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut card,
                )
                .ok()?;
        }
        let card = card?;
        // SAFETY: `list` is open, `card` and `upload` are live buffers; `upload` is retained via `list.keep`.
        unsafe {
            list.list.CopyResource(&card, &upload);
            list.list.ResourceBarrier(&[transition(
                &card,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        list.keep(&upload);
        Some(Self {
            format,
            scale,
            pitch,
            card,
        })
    }
}

pub(super) struct CursorOverlay {
    atlas: Option<Atlas>,
    failed: bool,
}

impl Drop for CursorOverlay {
    fn drop(&mut self) {
        if let Some(atlas) = self.atlas.take() {
            retire_resource(atlas.card);
        }
    }
}

impl CursorOverlay {
    pub(super) fn new() -> Self {
        Self {
            atlas: None,
            failed: false,
        }
    }

    pub(super) fn draw(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        size: (u32, u32),
        format: DXGI_FORMAT,
    ) {
        let snap = current_snapshot();
        self.draw_state(host, list, back, size, format, snap);
    }

    pub(super) fn draw_state(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        snap: CursorSnapshot,
    ) {
        if !snap.visible || width == 0 || height == 0 || self.failed {
            return;
        }
        let scale = sprite_scale(height);
        if self
            .atlas
            .as_ref()
            .is_none_or(|a| (a.format, a.scale) != (format, scale))
        {
            if let Some(old) = self.atlas.take() {
                retire_resource(old.card);
            }
            self.atlas = Atlas::create_and_upload(host, list, format, scale);
            if self.atlas.is_none() {
                self.failed = true;
                return;
            }
        }
        let Some(atlas) = &self.atlas else {
            return;
        };
        let blits = plan_blits(snap.u, snap.v, snap.pressed, (width, height), atlas.scale);
        if blits.is_empty() {
            return;
        }
        let sprite_w = CURSOR_W * atlas.scale;
        let sprite_h = CURSOR_H * atlas.scale;
        // SAFETY: `list` is open; `atlas.card` is in `COPY_SOURCE`, `back` is in `COPY_DEST`,
        // and all `CopyTextureRegion` boxes are clipped strictly inside both resources.
        unsafe {
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&atlas.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: format,
                            Width: sprite_w,
                            Height: sprite_h * 2,
                            Depth: 1,
                            RowPitch: atlas.pitch,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            for blit in blits {
                let src_box = D3D12_BOX {
                    left: blit.src_x,
                    top: blit.src_y,
                    front: 0,
                    right: blit.src_x + blit.w,
                    bottom: blit.src_y + blit.h,
                    back: 1,
                };
                list.list
                    .CopyTextureRegion(&to, blit.dst_x, blit.dst_y, 0, &from, Some(&src_box));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use super::*;

    fn cpu_blit(
        buf: &mut [u8],
        pitch: usize,
        bpp: usize,
        atlas: &[u8],
        atlas_pitch: usize,
        blits: &[BlitRect],
    ) {
        for b in blits {
            for dy in 0..b.h as usize {
                let src_row = (b.src_y as usize + dy) * atlas_pitch + b.src_x as usize * bpp;
                let dst_row = (b.dst_y as usize + dy) * pitch + b.dst_x as usize * bpp;
                let len = b.w as usize * bpp;
                buf[dst_row..dst_row + len].copy_from_slice(&atlas[src_row..src_row + len]);
            }
        }
    }

    #[test]
    fn cursor_shape_has_outline_fill_and_transparent_margin() {
        assert_eq!(CURSOR_ROWS.len(), CURSOR_H as usize);
        let mut outline = 0usize;
        let mut fill = 0usize;
        let mut transparent = 0usize;
        for row in CURSOR_ROWS {
            assert_eq!(row.len(), CURSOR_W as usize);
            for b in row.bytes() {
                match b {
                    b'#' => outline += 1,
                    b'@' => fill += 1,
                    b'.' => transparent += 1,
                    other => panic!("unexpected bitmap char {other}"),
                }
            }
        }
        assert_eq!(CURSOR_ROWS[0].as_bytes()[0], b'#');
        assert!(outline > 20 && fill > 20 && transparent > 20);

        let spans = logical_spans();
        assert!(!spans.is_empty());
        let mut covered = vec![vec![false; CURSOR_W as usize]; CURSOR_H as usize];
        for s in spans {
            assert!(s.x0 < s.x1 && s.x1 <= CURSOR_W && s.y < CURSOR_H);
            for x in s.x0..s.x1 {
                let ch = CURSOR_ROWS[s.y as usize].as_bytes()[x as usize];
                assert_ne!(ch, b'.', "span ({},{})..{} includes transparent pixel", s.y, s.x0, s.x1);
                assert!(!covered[s.y as usize][x as usize], "overlapping spans");
                covered[s.y as usize][x as usize] = true;
            }
        }
        for (y, row) in CURSOR_ROWS.iter().enumerate() {
            for (x, ch) in row.bytes().enumerate() {
                assert_eq!(covered[y][x], ch != b'.');
            }
        }
    }

    #[test]
    fn rasterize_and_blit_supports_all_swapchain_formats_and_scales() {
        let formats = [
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
        ];
        // Verify R10G10B10A2 pressed color matches exact 10-bit packing.
        let expected_10b = ((3u32 << 30) | (0x080u32 << 20) | (0x340u32 << 10) | 0x3FFu32).to_le_bytes();
        assert_eq!(palette(DXGI_FORMAT_R10G10B10A2_UNORM).unwrap()[2], &expected_10b);

        for format in formats {
            let pal = palette(format).unwrap();
            let bpp = pal[0].len();
            let sentinel: Vec<u8> = (0..bpp).map(|i| 0x42 + i as u8).collect();
            for scale in [1u32, 2, 4] {
                let atlas_pitch = ((CURSOR_W * scale) as usize * bpp).next_multiple_of(256);
                let atlas = rasterize_atlas(scale, pal, atlas_pitch);
                let (w, h) = (160u32, 160u32);
                let dst_pitch = w as usize * bpp;

                for pressed in [false, true] {
                    let mut dst = vec![0u8; dst_pitch * h as usize];
                    for px in dst.chunks_exact_mut(bpp) {
                        px.copy_from_slice(&sentinel);
                    }
                    let blits = plan_blits(0.25, 0.25, pressed, (w, h), scale);
                    assert!(!blits.is_empty());
                    cpu_blit(&mut dst, dst_pitch, bpp, &atlas, atlas_pitch, &blits);

                    let ax = ((0.25f32 * (w - 1) as f32).round() as u32) as usize;
                    let ay = ((0.25f32 * (h - 1) as f32).round() as u32) as usize;
                    let px_at = |x: usize, y: usize| &dst[y * dst_pitch + x * bpp..][..bpp];

                    // Tip (0,0) -> outline.
                    assert_eq!(px_at(ax, ay), pal[0]);
                    // Interior (1, 2) -> fill or pressed accent.
                    let expected_fill = if pressed { pal[2] } else { pal[1] };
                    assert_eq!(
                        px_at(ax + scale as usize, ay + 2 * scale as usize),
                        expected_fill
                    );
                    // Transparent pixel (4, 0) inside bounding box must remain sentinel!
                    assert_eq!(px_at(ax + 4 * scale as usize, ay), sentinel.as_slice());
                    // Far corner untouched.
                    assert_eq!(px_at(0, 0), sentinel.as_slice());
                }
            }
        }
    }

    #[test]
    fn clamping_at_screen_bounds_never_panics_or_overflows() {
        for (w, h) in [(32, 32), (1, 1), (0, 0)] {
            for scale in [1, 2, 4] {
                let sprite_w = CURSOR_W * scale;
                let sprite_h = CURSOR_H * scale;
                for (u, v) in [
                    (-0.5, -0.5),
                    (0.0, 0.0),
                    (0.999, 0.999),
                    (1.0, 1.0),
                    (1.5, 2.0),
                ] {
                    let blits = plan_blits(u, v, true, (w, h), scale);
                    if w == 0 || h == 0 {
                        assert!(blits.is_empty());
                        continue;
                    }
                    assert!(!blits.is_empty());
                    for b in blits {
                        assert!(b.dst_x < w && b.dst_y < h);
                        assert!(b.dst_x + b.w <= w && b.dst_y + b.h <= h);
                        assert!(b.src_x + b.w <= sprite_w);
                        assert!(b.src_y >= sprite_h && b.src_y + b.h <= sprite_h * 2);
                    }
                }
            }
        }
    }

    #[test]
    fn shared_cursor_pack_unpack_and_visibility_toggle_ipc() {
        static TEST_SEQ: AtomicU32 = AtomicU32::new(1);
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let name = format!("Local\\victory-cursor-test-{}-{seq}", std::process::id());
        let writer = SharedMapping::open(&name).expect("writer mapping");
        let reader = SharedMapping::open(&name).expect("reader mapping");

        // Uninitialized header returns None; standalone fallback obeys VICTORY_CURSOR env.
        assert_eq!(read_shared_snapshot(&reader), None);
        assert!(
            !resolve_cursor_snapshot(CursorEnv::Default, None, Some((0.5, 0.5, false))).visible
        );
        assert!(
            resolve_cursor_snapshot(CursorEnv::StandaloneOn, None, Some((0.5, 0.5, true)))
                .visible
        );

        let hdr_w = writer.header();
        hdr_w.magic.store(CURSOR_MAGIC, Ordering::Release);
        hdr_w.version.store(CURSOR_VERSION, Ordering::Release);

        let cases = [
            CursorSnapshot {
                u: 0.0,
                v: 0.0,
                visible: true,
                pressed: false,
                cursor_mode: true,
                initialized: true,
                seq: 1,
            },
            CursorSnapshot {
                u: 0.5,
                v: 0.5,
                visible: false,
                pressed: false,
                cursor_mode: false,
                initialized: true,
                seq: 2,
            },
            CursorSnapshot {
                u: 1.0,
                v: 1.0,
                visible: true,
                pressed: true,
                cursor_mode: true,
                initialized: true,
                seq: 0x0FFF_FFFF,
            },
        ];
        for s in cases {
            hdr_w.packed.store(pack_cursor(s), Ordering::Release);
            let got = read_shared_snapshot(&reader).expect("valid header");
            assert!((got.u - s.u).abs() < 1e-4);
            assert!((got.v - s.v).abs() < 1e-4);
            assert_eq!(got.visible, s.visible);
            assert_eq!(got.pressed, s.pressed);
            assert_eq!(got.cursor_mode, s.cursor_mode);
            assert_eq!(got.initialized, s.initialized);
            assert_eq!(got.seq, s.seq);

            // Kill-switch always overrides to hidden.
            assert!(!resolve_cursor_snapshot(CursorEnv::Disabled, Some(got), None).visible);
            assert_eq!(
                resolve_cursor_snapshot(CursorEnv::Default, Some(got), None).visible,
                s.visible
            );
        }
    }

    #[test]
    fn d3d12_gpu_cursor_draw_and_readback() {
        let Some(host) = super::super::super::host() else {
            return;
        };
        let (w, h) = (128u32, 128u32);
        let bpp = 4usize;
        let row_pitch = (w as usize * bpp).next_multiple_of(256);
        let total_bytes = (row_pitch * h as usize) as u64;

        let tex_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(w),
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        let mut tex: Option<ID3D12Resource> = None;
        // SAFETY: creating a 128x128 RGBA8 texture on the host D3D12 device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &tex_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &tex_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut tex,
                )
                .expect("CreateCommittedResource texture");
        }
        let tex = tex.expect("texture");

        let blue = [0x00u8, 0x00, 0xFF, 0xFF];
        let (clear_up, clear_ptr) = host.upload_buffer(total_bytes).expect("clear upload");
        // SAFETY: `clear_ptr` points to `total_bytes` mapped memory.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(clear_ptr, total_bytes as usize);
            for row in 0..h as usize {
                for col in 0..w as usize {
                    let off = row * row_pitch + col * bpp;
                    slice[off..off + bpp].copy_from_slice(&blue);
                }
            }
        }
        let readback = host.read_back_buffer(total_bytes).expect("readback");

        let run_case = |overlay: &mut CursorOverlay, snap: CursorSnapshot| -> Vec<u8> {
            let list = host.new_list().expect("new_list");
            // SAFETY: recording clear -> cursor draw -> readback on open command list.
            unsafe {
                let clear_src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&clear_up),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: w,
                                Height: h,
                                Depth: 1,
                                RowPitch: row_pitch as u32,
                            },
                        },
                    },
                };
                let tex_loc = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.list.CopyTextureRegion(&tex_loc, 0, 0, 0, &clear_src, None);
                overlay.draw_state(host, &list, &tex, (w, h), DXGI_FORMAT_R8G8B8A8_UNORM, snap);
                list.list.ResourceBarrier(&[transition(
                    &tex,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let rb_dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: w,
                                Height: h,
                                Depth: 1,
                                RowPitch: row_pitch as u32,
                            },
                        },
                    },
                };
                list.list.CopyTextureRegion(&rb_dst, 0, 0, 0, &tex_loc, None);
                list.list.ResourceBarrier(&[transition(
                    &tex,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            }
            list.close();
            host.execute(&[&list]);

            let mut ptr = std::ptr::null_mut::<c_void>();
            // SAFETY: GPU execution finished; reading back `total_bytes`.
            unsafe {
                readback.Map(0, None, Some(&mut ptr)).expect("readback Map");
                let out = std::slice::from_raw_parts(ptr.cast::<u8>(), total_bytes as usize).to_vec();
                readback.Unmap(0, None);
                out
            }
        };

        let mut overlay = CursorOverlay::new();
        let px_at = |data: &[u8], x: usize, y: usize| -> [u8; 4] {
            let off = y * row_pitch + x * bpp;
            data[off..off + 4].try_into().unwrap()
        };

        // Case 1: visible = true, normal fill at (0.5, 0.5)
        let normal_img = run_case(
            &mut overlay,
            CursorSnapshot {
                u: 0.5,
                v: 0.5,
                visible: true,
                pressed: false,
                cursor_mode: true,
                initialized: true,
                seq: 1,
            },
        );
        let ax = (0.5f32 * (w - 1) as f32).round() as usize;
        let ay = (0.5f32 * (h - 1) as f32).round() as usize;
        assert_eq!(px_at(&normal_img, ax, ay), [0x00, 0x00, 0x00, 0xFF]);
        assert_eq!(px_at(&normal_img, ax + 1, ay + 2), [0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(px_at(&normal_img, 0, 0), blue);

        // Case 2: visible = true, pressed = true -> gold interior
        let pressed_img = run_case(
            &mut overlay,
            CursorSnapshot {
                u: 0.5,
                v: 0.5,
                visible: true,
                pressed: true,
                cursor_mode: true,
                initialized: true,
                seq: 2,
            },
        );
        assert_eq!(px_at(&pressed_img, ax, ay), [0x00, 0x00, 0x00, 0xFF]);
        assert_eq!(px_at(&pressed_img, ax + 1, ay + 2), [0xFF, 0xD0, 0x20, 0xFF]);

        // Case 3: visible = false -> all pixels untouched blue
        let hidden_img = run_case(
            &mut overlay,
            CursorSnapshot {
                u: 0.5,
                v: 0.5,
                visible: false,
                pressed: false,
                cursor_mode: false,
                initialized: true,
                seq: 3,
            },
        );
        for y in 0..h as usize {
            for x in 0..w as usize {
                assert_eq!(px_at(&hidden_img, x, y), blue);
            }
        }

        retire_resource(tex);
        retire_resource(clear_up);
        retire_resource(readback);
    }
}
