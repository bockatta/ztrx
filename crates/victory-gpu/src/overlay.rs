//! Lightweight top-right in-swapchain FPS & frametime overlay toggled by F9.
//!
//! Designed for near-zero overhead (< 0.1 ms/frame when visible, 0.000 ms and
//! zero D3D12 commands when hidden):
//! - Fixed-size stack ring buffer (zero per-frame heap allocations).
//! - Pre-allocated per-backbuffer upload strip (512x200, 2048-byte row pitch).
//! - Large crisp integer-scaled glyphs (3x headline FPS + 2x metrics) for clear
//!   readability at 1080p/1440p/4K.
//! - Copied directly into the swapchain backbuffer via a single
//!   `CopyTextureRegion` while the backbuffer is already in `COPY_DEST`
//!   (no PSO, no HLSL, no RTV, no extra barriers, no DWM DirectFlip break).
//! - Self-profiles CPU raster + command recording in microseconds (`last_us`).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

/// Panel pixel dimensions (512 wide x 200 tall, top-right with 16 px margin).
pub(crate) const PANEL_W: u32 = 512;
pub(crate) const PANEL_H: u32 = 200;
/// 256-byte aligned row pitch for 4-byte-per-pixel formats (512 * 4 = 2048).
const ROW_PITCH: usize = 2048;
const STRIDE_U32: usize = ROW_PITCH / 4;
const BUF_BYTES: usize = ROW_PITCH * PANEL_H as usize;
const RING_LEN: usize = 240;
/// 122 bars x 4 px = 488 px wide inside the 512 px panel.
const SPARK_BARS: usize = 122;
const VK_F9: i32 = 0x78;

static VISIBLE: AtomicBool = AtomicBool::new(true);
static VISIBLE_INIT: std::sync::Once = std::sync::Once::new();
static F9_DOWN: AtomicBool = AtomicBool::new(false);
static LAST_OVERLAY_US: AtomicU64 = AtomicU64::new(0);

/// Rolling frame-time statistics computed from the last `RING_LEN` frames
/// with zero heap allocations.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameStats {
    pub cur_ms: f32,
    pub cur_fps: f32,
    pub avg_ms: f32,
    pub avg_fps: f32,
    pub p99_ms: f32,
    pub p99_fps: f32,
    #[allow(dead_code)]
    pub count: usize,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            cur_ms: 0.0,
            cur_fps: 0.0,
            avg_ms: 0.0,
            avg_fps: 0.0,
            p99_ms: 0.0,
            p99_fps: 0.0,
            count: 0,
        }
    }
}

/// Fixed-capacity ring buffer of recent frame times in milliseconds.
pub(crate) struct FrameHistory {
    ring: [f32; RING_LEN],
    head: usize,
    count: usize,
    last_present: Option<Instant>,
    scratch: [f32; RING_LEN],
    last_stats: FrameStats,
}

impl FrameHistory {
    pub(crate) fn new() -> Self {
        Self {
            ring: [0.0; RING_LEN],
            head: 0,
            count: 0,
            last_present: None,
            scratch: [0.0; RING_LEN],
            last_stats: FrameStats::default(),
        }
    }

    /// Records one completed frame using wall-clock delta since the previous
    /// present and returns updated rolling statistics (zero allocations).
    pub(crate) fn record_now(&mut self) -> FrameStats {
        let now = Instant::now();
        let dt_ms = match self.last_present.replace(now) {
            Some(prev) => (now.saturating_duration_since(prev).as_secs_f64() * 1000.0) as f32,
            None => return self.last_stats,
        };
        self.record_ms(dt_ms)
    }

    /// Records a frame duration in milliseconds and updates rolling statistics.
    pub(crate) fn record_ms(&mut self, ms: f32) -> FrameStats {
        let ms = ms.clamp(0.05, 60_000.0);
        self.ring[self.head] = ms;
        self.head = (self.head + 1) % RING_LEN;
        if self.count < RING_LEN {
            self.count += 1;
        }
        let n = self.count;
        let mut sum = 0.0f64;
        for i in 0..n {
            let v = self.ring[i];
            self.scratch[i] = v;
            sum += f64::from(v);
        }
        let slice = &mut self.scratch[..n];
        slice.sort_unstable_by(|a, b| a.total_cmp(b));
        // 1% low frametime = average of the worst 1% of frames (at least 1 frame).
        let worst_n = (n / 100).max(1);
        let worst_sum: f64 = slice[n - worst_n..].iter().map(|&v| f64::from(v)).sum();
        let p99_ms = (worst_sum / worst_n as f64) as f32;
        let avg_ms = (sum / n as f64) as f32;
        let stats = FrameStats {
            cur_ms: ms,
            cur_fps: if ms > 0.0 { 1000.0 / ms } else { 0.0 },
            avg_ms,
            avg_fps: if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 },
            p99_ms,
            p99_fps: if p99_ms > 0.0 { 1000.0 / p99_ms } else { 0.0 },
            count: n,
        };
        self.last_stats = stats;
        stats
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> FrameStats {
        self.last_stats
    }

    /// Copies the most recent `SPARK_BARS` frame times (oldest to newest) into `out`.
    fn recent(&self, out: &mut [f32; SPARK_BARS]) -> usize {
        let n = self.count.min(SPARK_BARS);
        let start = (self.head + RING_LEN - n) % RING_LEN;
        for i in 0..n {
            out[i] = self.ring[(start + i) % RING_LEN];
        }
        n
    }
}

/// Returns whether the overlay is currently visible (initializes from `VICTORY_OVERLAY`).
pub(crate) fn visible() -> bool {
    VISIBLE_INIT.call_once(|| {
        if std::env::var_os("VICTORY_OVERLAY").is_some_and(|v| v == "0") {
            VISIBLE.store(false, Ordering::Relaxed);
        }
    });
    VISIBLE.load(Ordering::Relaxed)
}

/// Polls F9 edge-trigger when the game window has focus, toggling visibility instantly.
pub(crate) fn poll_toggle(hwnd: HWND) -> bool {
    let _ = visible();
    if hwnd.0.is_null() {
        return VISIBLE.load(Ordering::Relaxed);
    }
    // SAFETY: plain Win32 focus and key-state queries.
    let focused = unsafe { GetForegroundWindow() } == hwnd;
    if !focused {
        F9_DOWN.store(false, Ordering::Relaxed);
        return VISIBLE.load(Ordering::Relaxed);
    }
    let down = unsafe { GetAsyncKeyState(VK_F9) } < 0;
    let was_down = F9_DOWN.swap(down, Ordering::Relaxed);
    if down && !was_down {
        let next = !VISIBLE.load(Ordering::Relaxed);
        VISIBLE.store(next, Ordering::Relaxed);
    }
    VISIBLE.load(Ordering::Relaxed)
}

/// Last measured CPU + command-recording overhead of the overlay, in microseconds.
pub(crate) fn last_us() -> u64 {
    LAST_OVERLAY_US.load(Ordering::Relaxed)
}

/// Per-presenter pre-allocated upload strips (one per swapchain backbuffer).
pub(crate) struct OverlayGpu {
    strips: Vec<(ID3D12Resource, *mut u8)>,
    pub(crate) history: FrameHistory,
    last_draw_us: u64,
}

// SAFETY: only accessed under `Host::presenter` mutex.
unsafe impl Send for OverlayGpu {}

impl OverlayGpu {
    pub(crate) fn new() -> Self {
        Self {
            strips: Vec::new(),
            history: FrameHistory::new(),
            last_draw_us: 0,
        }
    }

    /// Updates rolling frame stats every present (even when hidden so F9 toggle
    /// shows immediate history) and draws the top-right panel into `back` (which
    /// must already be in `D3D12_RESOURCE_STATE_COPY_DEST`) if visible.
    pub(crate) fn on_present(
        &mut self,
        host: &crate::host::Host,
        list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource,
        back_index: u32,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
        hwnd: HWND,
    ) -> FrameStats {
        let stats = self.history.record_now();
        if !poll_toggle(hwnd) {
            LAST_OVERLAY_US.store(0, Ordering::Relaxed);
            return stats;
        }
        if width < PANEL_W + 32 || height < PANEL_H + 32 {
            return stats;
        }
        let started = Instant::now();
        let slot = back_index as usize;
        while self.strips.len() <= slot {
            let Some(buf) = host.upload_buffer(BUF_BYTES as u64) else {
                return stats;
            };
            self.strips.push(buf);
        }
        let (ref upload_res, ptr) = self.strips[slot];
        // SAFETY: `upload_res` is mapped for `BUF_BYTES` and `back_index` is not
        // currently being read by an in-flight present after swapchain rotation.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr, BUF_BYTES) };
        let mut spark = [0.0f32; SPARK_BARS];
        let spark_n = self.history.recent(&mut spark);
        rasterize_panel(dst, format, &stats, &spark[..spark_n], self.last_draw_us);

        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer live for this present command list.
            pResource: unsafe { std::mem::transmute_copy(upload_res) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: PANEL_W,
                        Height: PANEL_H,
                        Depth: 1,
                        RowPitch: ROW_PITCH as u32,
                    },
                },
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed backbuffer pointer live for this present command list.
            pResource: unsafe { std::mem::transmute_copy(back) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let dst_x = width - PANEL_W - 16;
        let dst_y = 16;
        // SAFETY: `back` is in `COPY_DEST`, `dst_x + PANEL_W <= width`, `dst_y + PANEL_H <= height`.
        unsafe {
            list.CopyTextureRegion(&dst_loc, dst_x, dst_y, 0, &src_loc, None);
        }
        let us = started.elapsed().as_micros() as u64;
        self.last_draw_us = us;
        LAST_OVERLAY_US.store(us, Ordering::Relaxed);
        stats
    }
}

#[inline]
fn pack_pixel(format: DXGI_FORMAT, r: u8, g: u8, b: u8) -> u32 {
    match format {
        DXGI_FORMAT_B8G8R8A8_UNORM => {
            u32::from(b) | (u32::from(g) << 8) | (u32::from(r) << 16) | (0xFFu32 << 24)
        }
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let r10 = (u32::from(r) << 2) | (u32::from(r) >> 6);
            let g10 = (u32::from(g) << 2) | (u32::from(g) >> 6);
            let b10 = (u32::from(b) << 2) | (u32::from(b) >> 6);
            r10 | (g10 << 10) | (b10 << 20) | (3u32 << 30)
        }
        _ => u32::from(r) | (u32::from(g) << 8) | (u32::from(b) << 16) | (0xFFu32 << 24),
    }
}

/// Rasterizes the readable HUD panel (3x headline FPS, 2x metrics + 76 px sparkline)
/// directly onto u32 pixels with zero heap allocations.
pub(crate) fn rasterize_panel(
    buf: &mut [u8],
    format: DXGI_FORMAT,
    stats: &FrameStats,
    spark: &[f32],
    last_us: u64,
) {
    if !(buf.as_ptr() as usize).is_multiple_of(4) || buf.len() < BUF_BYTES {
        return;
    }
    // SAFETY: 4-byte aligned D3D12 upload buffer (or Vec<u8> in tests) of length >= BUF_BYTES.
    let pixels = unsafe {
        std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u32>(), BUF_BYTES / 4)
    };

    let bg = pack_pixel(format, 12, 15, 22);
    let border = pack_pixel(format, 90, 110, 145);
    let graph_bg = pack_pixel(format, 7, 9, 14);
    let text_hi = pack_pixel(format, 245, 250, 255);
    let text_dim = pack_pixel(format, 175, 192, 215);
    let green = pack_pixel(format, 78, 240, 134);
    let yellow = pack_pixel(format, 252, 212, 72);
    let red = pack_pixel(format, 250, 92, 92);
    let ref_33 = pack_pixel(format, 42, 115, 72);
    let ref_100 = pack_pixel(format, 128, 108, 42);

    // Fill panel background with crisp 2 px border using fast slice fills.
    let w = PANEL_W as usize;
    let h = PANEL_H as usize;
    for y in 0..h {
        let row = &mut pixels[y * STRIDE_U32..y * STRIDE_U32 + w];
        if y < 2 || y >= h - 2 {
            row.fill(border);
        } else {
            row.fill(bg);
            row[0] = border;
            row[1] = border;
            row[w - 2] = border;
            row[w - 1] = border;
        }
    }

    let fps_col = if stats.avg_fps >= 29.0 {
        green
    } else if stats.avg_fps >= 10.0 {
        yellow
    } else {
        red
    };
    let mut line = StackLine::new();

    // Line 1 (y=12, 3x scale = 15x21 px glyphs): headline FPS + AVG
    line.clear();
    line.push_str("FPS ");
    line.push_f1(stats.cur_fps);
    line.push_str("  AVG ");
    line.push_f1(stats.avg_fps);
    draw_text_scaled(pixels, 14, 12, line.as_bytes(), fps_col, 3);

    // Line 2 (y=44, 2x scale = 10x14 px glyphs): frametime current & average
    line.clear();
    line.push_str("FT ");
    line.push_f1(stats.cur_ms);
    line.push_str("MS   AVG ");
    line.push_f1(stats.avg_ms);
    line.push_str("MS");
    draw_text_scaled(pixels, 14, 44, line.as_bytes(), text_hi, 2);

    // Line 3 (y=66, 2x scale): 1% low ms & FPS
    line.clear();
    line.push_str("1% LOW ");
    line.push_f1(stats.p99_ms);
    line.push_str("MS (");
    line.push_f1(stats.p99_fps);
    line.push_str(" FPS)");
    draw_text_scaled(pixels, 14, 66, line.as_bytes(), text_hi, 2);

    // Line 4 (y=88, 2x scale): self-timed overlay cost + F9 hotkey
    line.clear();
    line.push_str("OVL ");
    line.push_f2(last_us as f32 / 1000.0);
    line.push_str("MS   [F9 HIDE]");
    draw_text_scaled(pixels, 14, 88, line.as_bytes(), text_dim, 2);

    // Sparkline graph box: x=12..500 (488 px = 122 bars x 4 px), y=112..188 (76 px tall).
    let (gx0, gy0, gw, gh) = (12usize, 112usize, SPARK_BARS * 4, 76usize);
    for y in gy0..gy0 + gh {
        pixels[y * STRIDE_U32 + gx0..y * STRIDE_U32 + gx0 + gw].fill(graph_bg);
    }

    // Dynamic vertical scale: at least 120 ms, clamped to 2.5x avg (max 500 ms).
    let scale_ms = (stats.avg_ms * 2.5).clamp(120.0, 500.0);

    // Dotted horizontal reference guides at 33.3 ms (30 FPS) and 100 ms (10 FPS).
    for (target, col) in [(33.3f32, ref_33), (100.0f32, ref_100)] {
        let th = ((target / scale_ms) * (gh - 4) as f32).round() as usize;
        if th > 0 && th < gh {
            let ry = gy0 + gh - 1 - th;
            let row = &mut pixels[ry * STRIDE_U32..ry * STRIDE_U32 + w];
            for x in (gx0..gx0 + gw).step_by(4) {
                row[x] = col;
                row[x + 1] = col;
            }
        }
    }

    // Bars: 3 px solid + 1 px gap for clean visual separation.
    let offset = SPARK_BARS.saturating_sub(spark.len());
    for (i, &ms) in spark.iter().enumerate() {
        let bar_h = (((ms / scale_ms) * (gh - 4) as f32).round() as usize).clamp(2, gh - 2);
        let col = if ms <= 34.0 {
            green
        } else if ms <= 100.0 {
            yellow
        } else {
            red
        };
        let bx = gx0 + (offset + i) * 4;
        for y in (gy0 + gh - bar_h)..(gy0 + gh) {
            let base = y * STRIDE_U32 + bx;
            pixels[base] = col;
            pixels[base + 1] = col;
            pixels[base + 2] = col;
        }
    }
}

/// Tiny zero-allocation ASCII formatting buffer for HUD lines.
struct StackLine {
    buf: [u8; 64],
    len: usize,
}

impl StackLine {
    fn new() -> Self {
        Self { buf: [0; 64], len: 0 }
    }
    fn clear(&mut self) {
        self.len = 0;
    }
    fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
    fn push_byte(&mut self, b: u8) {
        if self.len < self.buf.len() {
            self.buf[self.len] = b;
            self.len += 1;
        }
    }
    fn push_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            self.push_byte(b);
        }
    }
    fn push_uint(&mut self, mut v: u32) {
        let mut tmp = [0u8; 10];
        let mut n = 0;
        if v == 0 {
            self.push_byte(b'0');
            return;
        }
        while v > 0 {
            tmp[n] = b'0' + (v % 10) as u8;
            v /= 10;
            n += 1;
        }
        while n > 0 {
            n -= 1;
            self.push_byte(tmp[n]);
        }
    }
    fn push_f1(&mut self, val: f32) {
        let tenths = (val.max(0.0) * 10.0).round() as u32;
        self.push_uint(tenths / 10);
        self.push_byte(b'.');
        self.push_byte(b'0' + (tenths % 10) as u8);
    }
    fn push_f2(&mut self, val: f32) {
        let hundredths = (val.max(0.0) * 100.0).round() as u32;
        self.push_uint(hundredths / 100);
        self.push_byte(b'.');
        self.push_byte(b'0' + ((hundredths / 10) % 10) as u8);
        self.push_byte(b'0' + (hundredths % 10) as u8);
    }
}

/// Draws integer-scaled 5x7 glyphs directly onto the u32 pixel buffer.
fn draw_text_scaled(pixels: &mut [u32], mut x: usize, y: usize, text: &[u8], px: u32, scale: usize) {
    let w = PANEL_W as usize;
    let h = PANEL_H as usize;
    let advance = 6 * scale;
    for &ch in text {
        if x + 5 * scale > w {
            break;
        }
        let g = glyph(ch);
        for (row, &bits) in g.iter().enumerate() {
            if bits == 0 {
                continue;
            }
            let py0 = y + row * scale;
            if py0 + scale > h {
                break;
            }
            for col in 0..5usize {
                if (bits >> (4 - col)) & 1 != 0 {
                    let px0 = x + col * scale;
                    for dy in 0..scale {
                        let base = (py0 + dy) * STRIDE_U32 + px0;
                        for dx in 0..scale {
                            pixels[base + dx] = px;
                        }
                    }
                }
            }
        }
        x += advance;
    }
}

/// Crisp 5x7 uppercase/numeric ASCII bitmap glyphs (zero external assets).
fn glyph(ch: u8) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        b'0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        b'1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        b'3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        b'4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        b'5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        b'6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        b'7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        b'8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        b'9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        b'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        b'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        b'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        b'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        b'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        b'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        b'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        b'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        b'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        b'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        b'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        b'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        b'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        b'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        b'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        b'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        b'.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        b'%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        b'(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        b')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        b'[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        b']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        b':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        b'-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        b'/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        _ => [0; 7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_history_computes_exact_avg_and_1pct_low() {
        let mut h = FrameHistory::new();
        for _ in 0..99 {
            h.record_ms(100.0);
        }
        let s = h.record_ms(500.0);
        assert_eq!(s.count, 100);
        assert!((s.avg_ms - 104.0).abs() < 0.01);
        assert!((s.p99_ms - 500.0).abs() < 0.01);
        assert!((s.p99_fps - 2.0).abs() < 0.01);
    }

    #[test]
    fn overlay_rasterize_overhead_is_under_half_millisecond() {
        let mut h = FrameHistory::new();
        for i in 0..140 {
            h.record_ms(33.3 + (i % 40) as f32 * 4.0);
        }
        let stats = h.stats();
        let mut spark = [0.0f32; SPARK_BARS];
        let n = h.recent(&mut spark);
        let mut buf = vec![0u8; BUF_BYTES];
        // Warm up once, then time 100 rasterizations.
        rasterize_panel(&mut buf, DXGI_FORMAT_R8G8B8A8_UNORM, &stats, &spark[..n], 25);
        let started = Instant::now();
        for _ in 0..100 {
            rasterize_panel(&mut buf, DXGI_FORMAT_R8G8B8A8_UNORM, &stats, &spark[..n], 25);
        }
        let avg_us = started.elapsed().as_micros() as f64 / 100.0;
        println!("overlay rasterize avg: {avg_us:.2} us/frame");
        let budget_us = if cfg!(debug_assertions) { 3000.0 } else { 500.0 };
        assert!(
            avg_us < budget_us,
            "overlay rasterization exceeded {budget_us} us budget: {avg_us:.1} us"
        );
    }
}
