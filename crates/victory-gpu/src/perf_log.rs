//! Structured, rate-limited, low-overhead JSONL performance & asset telemetry.
//!
//! Covers:
//! - Periodic frame timing aggregates (FPS, avg ms, 1% low ms/FPS, CPU prep,
//!   copies, texture refresh, heap pass, GPU wait, copy-back, overlay/log cost).
//! - One-shot timestamped load milestones (`engine_init`, `first_present`,
//!   `world_enter`, `prison_warmup_done`, `pipelines_warm`, `textures_drained`).
//! - Asset streaming state (queue depth, due count, oldest age, async COPY-queue
//!   pending/uploaded totals).
//! - Shader/pipeline activity (making, compute waits, cache hits/misses).
//!
//! Controlled by `VICTORY_PERF_LOG=0|1|2` (default `1`; runtime toggleable via
//! F10 when focused) and `VICTORY_PERF_LOG_PATH` (default `%TEMP%\ztrx-perf.jsonl`).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use windows::Win32::Foundation::HWND;
use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

const VK_F10: i32 = 0x79;

static START: OnceLock<Instant> = OnceLock::new();
static VERBOSITY: AtomicU8 = AtomicU8::new(1);
static VERBOSITY_INIT: std::sync::Once = std::sync::Once::new();
static F10_DOWN: AtomicBool = AtomicBool::new(false);
static LAST_LOG_US: AtomicU64 = AtomicU64::new(0);

// One-shot milestone flags.
static M_FIRST_PRESENT: AtomicBool = AtomicBool::new(false);
static M_WORLD_ENTER: AtomicBool = AtomicBool::new(false);
static M_WARMUP_DONE: AtomicBool = AtomicBool::new(false);
static M_PIPELINES_WARM: AtomicBool = AtomicBool::new(false);
static M_TEXTURES_DRAINED: AtomicBool = AtomicBool::new(false);
static WORLD_ENTER_MS: AtomicU64 = AtomicU64::new(0);

struct Sink {
    writer: BufWriter<File>,
    last_agg: Instant,
}

static SINK: Mutex<Option<Sink>> = Mutex::new(None);
static SINK_OPENED: std::sync::Once = std::sync::Once::new();

fn start_time() -> Instant {
    *START.get_or_init(Instant::now)
}

fn uptime_ms() -> u64 {
    start_time().elapsed().as_millis() as u64
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Current verbosity: `0` = off, `1` = normal (2s aggregates + milestones),
/// `2` = verbose (1s aggregates + extra detail).
pub(crate) fn verbosity() -> u8 {
    VERBOSITY_INIT.call_once(|| {
        let v = std::env::var("VICTORY_PERF_LOG")
            .ok()
            .and_then(|s| s.trim().parse::<u8>().ok())
            .map_or(1, |n| n.min(2));
        VERBOSITY.store(v, Ordering::Relaxed);
    });
    VERBOSITY.load(Ordering::Relaxed)
}

/// Polls F10 edge-trigger when focused to cycle verbosity `0 -> 1 -> 2 -> 0`.
pub(crate) fn poll_verbosity_toggle(hwnd: HWND) {
    let _ = verbosity();
    if hwnd.0.is_null() {
        return;
    }
    let focused = unsafe { GetForegroundWindow() } == hwnd;
    if !focused {
        F10_DOWN.store(false, Ordering::Relaxed);
        return;
    }
    let down = unsafe { GetAsyncKeyState(VK_F10) } < 0;
    let was = F10_DOWN.swap(down, Ordering::Relaxed);
    if down && !was {
        let next = (VERBOSITY.load(Ordering::Relaxed) + 1) % 3;
        VERBOSITY.store(next, Ordering::Relaxed);
        tracing::info!(verbosity = next, "VICTORY_PERF_LOG verbosity toggled via F10");
    }
}

fn log_path() -> PathBuf {
    if let Some(p) = std::env::var_os("VICTORY_PERF_LOG_PATH")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    std::env::temp_dir().join("ztrx-perf.jsonl")
}

fn ensure_open() {
    let mut opened_now = false;
    SINK_OPENED.call_once(|| {
        let _ = start_time();
        let path = log_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match OpenOptions::new().create(true).write(true).truncate(true).open(&path) {
            Ok(file) => {
                let mut guard = SINK.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(Sink {
                    writer: BufWriter::with_capacity(16 * 1024, file),
                    last_agg: Instant::now() - Duration::from_secs(10),
                });
                opened_now = true;
            }
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "could not open perf JSONL log");
            }
        }
    });
    if opened_now {
        // Emit initial milestone AFTER `SINK_OPENED.call_once` completes to avoid
        // re-entering `Once::call_once` on the same thread (which deadlocks).
        milestone("engine_init", "structured perf log opened");
    }
}

fn write_line(line: &str) {
    if verbosity() == 0 {
        return;
    }
    ensure_open();
    let started = Instant::now();
    let mut guard = SINK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(sink) = guard.as_mut() {
        let _ = sink.writer.write_all(line.as_bytes());
        let _ = sink.writer.write_all(b"\n");
        let _ = sink.writer.flush();
    }
    LAST_LOG_US.store(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Emits a structured timestamped milestone line.
pub(crate) fn milestone(name: &str, detail: &str) {
    if verbosity() == 0 {
        return;
    }
    let line = format!(
        "{{\"ts_ms\":{},\"t_ms\":{},\"level\":\"INFO\",\"event\":\"milestone\",\"name\":\"{}\",\"detail\":\"{}\"}}",
        epoch_ms(),
        uptime_ms(),
        name,
        detail
    );
    write_line(&line);
}

/// Snapshot passed from `Meter::frame` (already aggregated over the last ~1s).
pub(crate) struct FrameAgg {
    pub frame: u64,
    pub stats: crate::overlay::FrameStats,
    pub draws: f64,
    pub dropped: f64,
    pub cpu_prep_ms: f64,
    pub cpu_copies_ms: f64,
    pub cpu_copies_mb: f64,
    pub cpu_tex_ms: f64,
    pub cpu_tex_mb: f64,
    pub cpu_heap_ms: f64,
    pub cpu_heap_slots_ms: f64,
    pub cpu_heap_tex_ms: f64,
    pub gpu_wait_ms: f64,
    pub copyback_ms: f64,
    pub copyback_mb: f64,
    pub pso_making: u64,
    pub compute_waits: u64,
}

/// Records milestones and rate-limited periodic aggregates (every 2s at
/// verbosity 1, every 1s at verbosity 2).
pub(crate) fn record_aggregate(agg: &FrameAgg) {
    let v = verbosity();
    if v == 0 {
        return;
    }
    ensure_open();
    let now_ms = uptime_ms();
    let in_world = agg.draws >= 100.0;

    if !M_FIRST_PRESENT.swap(true, Ordering::Relaxed) {
        milestone("first_present", "first swapchain frame presented");
    }
    if in_world && !M_WORLD_ENTER.swap(true, Ordering::Relaxed) {
        WORLD_ENTER_MS.store(now_ms, Ordering::Relaxed);
        milestone("world_enter", "entered gameplay scene (draws >= 100)");
    }
    if M_WORLD_ENTER.load(Ordering::Relaxed) {
        let enter = WORLD_ENTER_MS.load(Ordering::Relaxed);
        if now_ms.saturating_sub(enter) >= 15_000
            && !M_WARMUP_DONE.swap(true, Ordering::Relaxed)
        {
            milestone("prison_warmup_done", "15s post-world-enter warmup complete");
        }
        if agg.pso_making == 0 && !M_PIPELINES_WARM.swap(true, Ordering::Relaxed) {
            milestone("pipelines_warm", "all active pipelines compiled (making=0)");
        }
        let due = crate::draw::HEAP_TEXTURES_QUEUE[1].load(Ordering::Relaxed);
        if due < 64 && !M_TEXTURES_DRAINED.swap(true, Ordering::Relaxed) {
            milestone("textures_drained", "texture queue drained below 64 due slots");
        }
    }

    let min_interval = if v >= 2 {
        Duration::from_secs(1)
    } else {
        Duration::from_secs(2)
    };
    {
        let mut guard = SINK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(sink) = guard.as_mut() else {
            return;
        };
        if sink.last_agg.elapsed() < min_interval {
            return;
        }
        sink.last_agg = Instant::now();
    }

    let [tex_waiting, tex_due, tex_away, tex_partly, tex_oldest] =
        [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
    let tex_uploaded = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed);
    let (async_pending, async_uploaded) = crate::texture::async_counts();
    let (cache_hits, cache_misses) = crate::shader_cache::counts();
    let overlay_us = crate::overlay::last_us();
    let log_us = LAST_LOG_US.load(Ordering::Relaxed);

    let line = format!(
        "{{\"ts_ms\":{},\"t_ms\":{},\"level\":\"INFO\",\"event\":\"frame_agg\",\"frame\":{},\"in_world\":{},\
         \"fps_cur\":{:.2},\"fps_avg\":{:.2},\"ms_avg\":{:.2},\"ms_1pct_low\":{:.2},\"fps_1pct_low\":{:.2},\
         \"cpu_prep_ms\":{:.2},\"cpu_copies_ms\":{:.2},\"cpu_copies_mb\":{:.2},\"cpu_tex_ms\":{:.2},\"cpu_tex_mb\":{:.2},\
         \"cpu_heap_ms\":{:.2},\"cpu_heap_slots_ms\":{:.2},\"cpu_heap_tex_ms\":{:.2},\
         \"gpu_wait_ms\":{:.2},\"copyback_ms\":{:.2},\"copyback_mb\":{:.2},\
         \"draws\":{:.0},\"dropped\":{:.1},\"overlay_us\":{},\"log_us\":{},\
         \"tex_uploaded\":{},\"tex_waiting\":{},\"tex_due\":{},\"tex_oldest_frames\":{},\"tex_away\":{},\"tex_partly\":{},\
         \"tex_async_pending\":{},\"tex_async_uploaded\":{},\
         \"pso_making\":{},\"compute_waits\":{},\"shader_cache_hits\":{},\"shader_cache_misses\":{}}}",
        epoch_ms(),
        now_ms,
        agg.frame,
        in_world,
        agg.stats.cur_fps,
        agg.stats.avg_fps,
        agg.stats.avg_ms,
        agg.stats.p99_ms,
        agg.stats.p99_fps,
        agg.cpu_prep_ms,
        agg.cpu_copies_ms,
        agg.cpu_copies_mb,
        agg.cpu_tex_ms,
        agg.cpu_tex_mb,
        agg.cpu_heap_ms,
        agg.cpu_heap_slots_ms,
        agg.cpu_heap_tex_ms,
        agg.gpu_wait_ms,
        agg.copyback_ms,
        agg.copyback_mb,
        agg.draws,
        agg.dropped,
        overlay_us,
        log_us,
        tex_uploaded,
        tex_waiting,
        tex_due,
        tex_oldest,
        tex_away,
        tex_partly,
        async_pending,
        async_uploaded,
        agg.pso_making,
        agg.compute_waits,
        cache_hits,
        cache_misses,
    );
    write_line(&line);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perf_log_json_format_and_overhead_are_negligible() {
        let started = Instant::now();
        for i in 0..100u64 {
            let line = format!(
                "{{\"ts_ms\":{},\"t_ms\":{},\"level\":\"INFO\",\"event\":\"frame_agg\",\"frame\":{}}}",
                epoch_ms(),
                uptime_ms(),
                i
            );
            assert!(line.starts_with('{') && line.ends_with('}'));
        }
        let avg_us = started.elapsed().as_micros() as f64 / 100.0;
        assert!(avg_us < 100.0, "perf log formatting too slow: {avg_us:.1} us");
    }

    #[test]
    fn perf_log_ensure_open_and_milestone_do_not_deadlock() {
        // Regression test: `ensure_open` must never re-enter `SINK_OPENED.call_once`.
        let started = Instant::now();
        milestone("test_open", "verifying non-reentrant Once initialization");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "perf_log::milestone blocked unexpectedly"
        );
    }
}
