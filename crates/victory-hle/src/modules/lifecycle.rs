//! The title's life cycle. A console title is never told it is closing:
//! quitting it or launching a new game first constrains it (its
//! constrained-change routines run), then suspends it (its suspend routines
//! run, and it has a second to save) and then ends the process. Closing the
//! title's window does the same here; a new launch then starts `victory`
//! again, which waits for this process to be gone.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use windows_sys::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
use windows_sys::Win32::UI::WindowsAndMessaging::{self as wm, HMENU};
use windows_sys::core::{PCSTR, PCWSTR, w};

use crate::{HleModule, hle_export};

/// A suspend routine (`quiesced`) or a constrained-change routine
/// (`constrained`): both take a flag and the title's context.
type Routine = extern "system" fn(flag: u8, context: *mut c_void);

/// Registration id, routine and context.
type Registration = (usize, Routine, usize);

/// Each suspend routine.
static ROUTINES: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

/// Each constrained-change routine.
static CONSTRAINED: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

fn routines() -> MutexGuard<'static, Vec<Registration>> {
    ROUTINES.lock().unwrap_or_else(|e| e.into_inner())
}

fn constrained() -> MutexGuard<'static, Vec<Registration>> {
    CONSTRAINED.lock().unwrap_or_else(|e| e.into_inner())
}

/// How long the console lets suspend routines run before it ends the title.
const SUSPEND_DEADLINE: Duration = Duration::from_secs(1);

const ERROR_INVALID_PARAMETER: u32 = 87;

/// Adds `routine` to `list` and writes its registration id.
fn register(
    mut list: MutexGuard<'static, Vec<Registration>>,
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    let Some(routine) = routine else {
        return ERROR_INVALID_PARAMETER;
    };
    if registration.is_null() {
        return ERROR_INVALID_PARAMETER;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    list.push((id, routine, context as usize));
    // SAFETY: checked non-null; the title passes room for its registration.
    unsafe { registration.write(id) };
    0
}

pub extern "system" fn RegisterAppStateChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(routines(), routine, context, registration)
}

pub extern "system" fn UnregisterAppStateChangeNotification(registration: usize) {
    routines().retain(|&(id, ..)| id != registration);
}

/// The host's own version never calls the routine (a desktop app is never
/// constrained), so a title waiting for it would never suspend.
pub extern "system" fn RegisterAppConstrainedChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(constrained(), routine, context, registration)
}

pub extern "system" fn UnregisterAppConstrainedChangeNotification(registration: usize) {
    constrained().retain(|&(id, ..)| id != registration);
}

/// The window procedure a title window had before ours.
const TITLE_PROC: PCWSTR = w!("victory.wndproc");

/// Top-level title window handle tracked from `CreateWindowExW`/`CreateWindowExA`.
static TITLE_HWND: AtomicUsize = AtomicUsize::new(0);
/// True while the user is dragging/resizing the title window.
static IN_SIZE_MOVE: AtomicBool = AtomicBool::new(false);
/// Accumulated horizontal/vertical mouse wheel deltas (`WHEEL_DELTA = 120` per notch).
static WHEEL_X: AtomicI64 = AtomicI64::new(0);
static WHEEL_Y: AtomicI64 = AtomicI64::new(0);
/// Incremented on `WM_SETFOCUS` / `WM_KILLFOCUS`.
static FOCUS_SERIAL: AtomicU64 = AtomicU64::new(0);
/// Packed latest client-area pointer state from `WM_MOUSEMOVE`:
/// bits 0..=15 = u_q16, bits 16..=31 = v_q16, bit 32 = in_client, bit 33 = valid.
static CLIENT_CURSOR: AtomicU64 = AtomicU64::new(0);
/// Virtual-key bitset updated by `WM_KEYDOWN`/`WM_KEYUP` (`WM_SYSKEYDOWN`/`WM_SYSKEYUP`)
/// and cleared on `WM_KILLFOCUS` so posted window key messages work even when unfocused.
static WINDOW_KEYS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

struct CursorUiState {
    visible: AtomicBool,
    cursor_mode: AtomicBool,
}

fn cursor_ui_state() -> &'static CursorUiState {
    static STATE: OnceLock<CursorUiState> = OnceLock::new();
    STATE.get_or_init(|| {
        let initial_cursor = std::env::var_os("VICTORY_CURSOR").is_some_and(|v| v == "1");
        CursorUiState {
            visible: AtomicBool::new(initial_cursor),
            cursor_mode: AtomicBool::new(initial_cursor),
        }
    })
}

pub(crate) fn set_cursor_ui_state(visible: bool, cursor_mode: bool) {
    let state = cursor_ui_state();
    state.visible.store(visible, Ordering::Relaxed);
    state.cursor_mode.store(cursor_mode, Ordering::Relaxed);
}

fn should_hide_os_cursor_with(os_cursor_override: bool, visible: bool, cursor_mode: bool) -> bool {
    !os_cursor_override && (visible || !cursor_mode)
}

fn should_hide_os_cursor() -> bool {
    let os_override = std::env::var_os("VICTORY_CURSOR_OS").is_some_and(|v| v == "1");
    let state = cursor_ui_state();
    should_hide_os_cursor_with(
        os_override,
        state.visible.load(Ordering::Relaxed),
        state.cursor_mode.load(Ordering::Relaxed),
    )
}

pub(crate) fn title_hwnd() -> usize {
    TITLE_HWND.load(Ordering::Acquire)
}

pub(crate) fn in_size_move() -> bool {
    IN_SIZE_MOVE.load(Ordering::Acquire)
}

pub(crate) fn take_wheel() -> (i64, i64) {
    (
        WHEEL_X.swap(0, Ordering::AcqRel),
        WHEEL_Y.swap(0, Ordering::AcqRel),
    )
}

pub(crate) fn focus_serial() -> u64 {
    FOCUS_SERIAL.load(Ordering::Acquire)
}

pub(crate) fn client_cursor() -> Option<(f32, f32, bool)> {
    let packed = CLIENT_CURSOR.load(Ordering::Acquire);
    if (packed & (1 << 33)) == 0 {
        return None;
    }
    let u = (packed & 0xFFFF) as f32 / 65535.0;
    let v = ((packed >> 16) & 0xFFFF) as f32 / 65535.0;
    let in_client = (packed & (1 << 32)) != 0;
    Some((u, v, in_client))
}

fn set_client_cursor(u: f32, v: f32, in_client: bool) {
    let x_q16 = (u.clamp(0.0, 1.0) * 65535.0).round() as u64;
    let y_q16 = (v.clamp(0.0, 1.0) * 65535.0).round() as u64;
    let packed = x_q16 | (y_q16 << 16) | (u64::from(in_client) << 32) | (1u64 << 33);
    CLIENT_CURSOR.store(packed, Ordering::Release);
}

fn clear_client_in_client() {
    CLIENT_CURSOR.fetch_and(!(1u64 << 32), Ordering::AcqRel);
}

fn normalize_window_vk(vk: u8) -> u8 {
    match vk {
        0x10 => 0xA0, // VK_SHIFT -> VK_LSHIFT
        0x11 => 0xA2, // VK_CONTROL -> VK_LCONTROL
        0x12 => 0xA4, // VK_MENU -> VK_LMENU
        _ => vk,
    }
}

fn set_window_key(vk: u8, down: bool) {
    let vk = normalize_window_vk(vk);
    let word = usize::from(vk >> 6);
    let bit = 1u64 << (vk & 63);
    if down {
        WINDOW_KEYS[word].fetch_or(bit, Ordering::Release);
    } else {
        WINDOW_KEYS[word].fetch_and(!bit, Ordering::Release);
    }
}

fn clear_window_keys() {
    for word in &WINDOW_KEYS {
        word.store(0, Ordering::Release);
    }
}

pub(crate) fn window_keys() -> Vec<u8> {
    let mut keys = Vec::new();
    for (word, slot) in WINDOW_KEYS.iter().enumerate() {
        let mut bits = slot.load(Ordering::Acquire);
        while bits != 0 {
            let bit = bits.trailing_zeros() as u8;
            keys.push((word as u8) * 64 + bit);
            bits &= bits - 1;
        }
    }
    keys
}

fn track_top_level(hwnd: HWND, parent: HWND, style: u32) {
    if parent.is_null() && !hwnd.is_null() && (style & wm::WS_CHILD) == 0 {
        TITLE_HWND.store(hwnd as usize, Ordering::Release);
    }
}

pub extern "system" fn CreateWindowExW(
    ex_style: u32,
    class: PCWSTR,
    name: PCWSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let (x, y, width, height) = fit(parent, x, y, width, height);
    // SAFETY: the title's own arguments, forwarded (the size fitted).
    let hwnd = unsafe {
        wm::CreateWindowExW(
            ex_style, class, name, style, x, y, width, height, parent, menu, instance, param,
        )
    };
    track_top_level(hwnd, parent, style);
    hook(hwnd);
    hwnd
}

/// See [`CreateWindowExW`].
pub extern "system" fn CreateWindowExA(
    ex_style: u32,
    class: PCSTR,
    name: PCSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let (x, y, width, height) = fit(parent, x, y, width, height);
    // SAFETY: the title's own arguments, forwarded (the size fitted).
    let hwnd = unsafe {
        wm::CreateWindowExA(
            ex_style, class, name, style, x, y, width, height, parent, menu, instance, param,
        )
    };
    track_top_level(hwnd, parent, style);
    hook(hwnd);
    hwnd
}

/// A console title sizes its window to its render target (
/// the title: 3840x2160 on a Series X), which can exceed the host's screen.
/// A top-level window larger than the work area is shrunk to fit, keeping
/// its proportions, and centred; what it shows is scaled to it (the swap
/// chain stretches). `VICTORY_WINDOW_MAX=WIDTHxHEIGHT` applies a smaller
/// limit without changing the title's render targets.
fn fit(parent: HWND, x: i32, y: i32, width: i32, height: i32) -> (i32, i32, i32, i32) {
    let mut area = RECT::default();
    // SAFETY: `area` receives the primary monitor's work area.
    let known =
        unsafe { wm::SystemParametersInfoW(wm::SPI_GETWORKAREA, 0, (&raw mut area).cast(), 0) };
    let limit = window_limit();
    let fitted = fit_with_limit(&area, width, height, limit);
    if !parent.is_null() || known == 0 || fitted == (width, height) {
        return (x, y, width, height);
    }
    tracing::info!(width, height, to = ?fitted, ?limit, "title window fitted to the screen");
    let (w, h) = fitted;
    let left = area.left + (area.right - area.left - w) / 2;
    let top = area.top + (area.bottom - area.top - h) / 2;
    (left, top, w, h)
}

fn parse_window_limit(value: &str) -> Option<(i32, i32)> {
    let (width, height) = value.trim().split_once(['x', 'X'])?;
    let (width, height) = (width.trim().parse().ok()?, height.trim().parse().ok()?);
    (width > 0 && height > 0).then_some((width, height))
}

fn window_limit() -> Option<(i32, i32)> {
    static LIMIT: std::sync::OnceLock<Option<(i32, i32)>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let Some(value) = std::env::var_os("VICTORY_WINDOW_MAX") else {
            // Default to a clean <=1920x1080 centered window (never overflows a 1440p monitor).
            return Some((1920, 1080));
        };
        if value.to_str().is_some_and(|s| matches!(s.trim(), "0" | "off" | "none")) {
            return None;
        }
        let parsed = value.to_str().and_then(parse_window_limit);
        if parsed.is_none() {
            tracing::warn!(
                ?value,
                "ignoring invalid VICTORY_WINDOW_MAX; expected positive WIDTHxHEIGHT"
            );
        }
        parsed
    })
}

fn fit_with_limit(
    area: &RECT,
    width: i32,
    height: i32,
    limit: Option<(i32, i32)>,
) -> (i32, i32) {
    let Some((max_width, max_height)) = limit else {
        return fit_into(area, width, height);
    };
    let bounded = RECT {
        left: 0,
        top: 0,
        right: (area.right - area.left).min(max_width),
        bottom: (area.bottom - area.top).min(max_height),
    };
    fit_into(&bounded, width, height)
}

/// `width` x `height` scaled down to fit `area`, or unchanged if it fits
/// (or is `CW_USEDEFAULT`).
fn fit_into(area: &RECT, width: i32, height: i32) -> (i32, i32) {
    let (aw, ah) = (area.right - area.left, area.bottom - area.top);
    if width <= 0 || height <= 0 || aw <= 0 || ah <= 0 || (width <= aw && height <= ah) {
        return (width, height);
    }
    let scale = f64::min(
        f64::from(aw) / f64::from(width),
        f64::from(ah) / f64::from(height),
    );
    (
        (f64::from(width) * scale) as i32,
        (f64::from(height) * scale) as i32,
    )
}

/// Puts our procedure in front of the title's.
fn hook(hwnd: HWND) {
    if hwnd.is_null() {
        return;
    }
    // SAFETY: a window this thread just created. The title's procedure is
    // stored before ours can run.
    unsafe {
        let title = wm::GetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC);
        wm::SetPropW(hwnd, TITLE_PROC, title as *mut c_void);
        wm::SetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC, window_proc as *const () as isize);
    }
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // A user close (close button, Alt+F4, window or taskbar menu); a
    // console title never sees one, nor a WM_CLOSE from it.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_CLOSE as usize {
        end_title("window closed", Box::new(|| {}));
        return 0;
    }
    // Alt or F10 puts a window into menu mode: the system runs a modal menu
    // loop on the thread that owns the window, which is the title's. The
    // title then stops (no frames, no input) until a key ends the loop, and
    // nothing shows it, since the window has no menu bar;
    // the title aborts itself about half a minute later. A console has no window
    // menu. The mouse still opens it from the title bar.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_KEYMENU as usize {
        return 0;
    }
    match message {
        wm::WM_MOUSEMOVE => {
            let x = (lparam as isize & 0xFFFF) as i16;
            let y = ((lparam as isize >> 16) & 0xFFFF) as i16;
            let mut rect = RECT::default();
            // SAFETY: `hwnd` is the window receiving `WM_MOUSEMOVE`.
            if unsafe { wm::GetClientRect(hwnd, &mut rect) } != 0 {
                let w = rect.right - rect.left;
                let h = rect.bottom - rect.top;
                if w > 0 && h > 0 {
                    let u = (f32::from(x) / w as f32).clamp(0.0, 1.0);
                    let v = (f32::from(y) / h as f32).clamp(0.0, 1.0);
                    let in_client =
                        i32::from(x) >= 0 && i32::from(y) >= 0 && i32::from(x) < w && i32::from(y) < h;
                    set_client_cursor(u, v, in_client);
                }
            }
        }
        wm::WM_NCMOUSEMOVE => {
            clear_client_in_client();
        }
        wm::WM_MOUSEWHEEL => {
            let dy = i64::from(((wparam >> 16) & 0xFFFF) as i16);
            WHEEL_Y.fetch_add(dy, Ordering::AcqRel);
        }
        wm::WM_MOUSEHWHEEL => {
            let dx = i64::from(((wparam >> 16) & 0xFFFF) as i16);
            WHEEL_X.fetch_add(dx, Ordering::AcqRel);
        }
        wm::WM_ENTERSIZEMOVE => {
            IN_SIZE_MOVE.store(true, Ordering::Release);
        }
        wm::WM_EXITSIZEMOVE => {
            IN_SIZE_MOVE.store(false, Ordering::Release);
        }
        wm::WM_KEYDOWN | wm::WM_SYSKEYDOWN => {
            set_window_key((wparam & 0xFF) as u8, true);
        }
        wm::WM_KEYUP | wm::WM_SYSKEYUP => {
            set_window_key((wparam & 0xFF) as u8, false);
        }
        wm::WM_SETFOCUS => {
            FOCUS_SERIAL.fetch_add(1, Ordering::AcqRel);
        }
        wm::WM_KILLFOCUS => {
            FOCUS_SERIAL.fetch_add(1, Ordering::AcqRel);
            IN_SIZE_MOVE.store(false, Ordering::Release);
            clear_client_in_client();
            clear_window_keys();
        }
        wm::WM_SETCURSOR => {
            if (lparam as isize & 0xFFFF) as u32 == wm::HTCLIENT && should_hide_os_cursor() {
                // SAFETY: `SetCursor(null)` hides the OS arrow inside `HTCLIENT` while the
                // swapchain cursor or camera-look mode is active; non-client hit-tests
                // (`HTCAPTION`, borders) still reach the default window procedure.
                unsafe { wm::SetCursor(std::ptr::null_mut()) };
                return 1;
            }
        }
        _ => {}
    }
    // SAFETY: the procedure `hook` stored, called the way the system would.
    unsafe {
        let title = wm::GetPropW(hwnd, TITLE_PROC);
        if message == wm::WM_NCDESTROY {
            let _ = TITLE_HWND.compare_exchange(hwnd as usize, 0, Ordering::AcqRel, Ordering::Acquire);
            wm::RemovePropW(hwnd, TITLE_PROC);
        }
        let title = std::mem::transmute::<*mut c_void, wm::WNDPROC>(title);
        wm::CallWindowProcW(title, hwnd, message, wparam, lparam)
    }
}

/// Suspends the title as the console does when it ends it, runs `then`
/// (once), then ends the process. Returns at once; later calls do nothing.
pub(crate) fn end_title(why: &'static str, then: Box<dyn FnOnce() + Send>) {
    static ENDING: AtomicBool = AtomicBool::new(false);
    if ENDING.swap(true, Ordering::SeqCst) {
        return;
    }
    let constrain = constrained().clone();
    let suspend = routines().clone();
    tracing::info!(
        constrained = constrain.len(),
        routines = suspend.len(),
        "{why}: suspending the title"
    );
    // Whoever comes first (the routines or the deadline) runs `then`; the
    // other waits for it, so the process never ends halfway through.
    let then = std::sync::Arc::new(Mutex::new(Some(then)));
    let finish = move |how: &str| {
        let mut then = then.lock().unwrap_or_else(|e| e.into_inner());
        tracing::info!("{how}; ending the title");
        if let Some(then) = then.take() {
            then();
        }
        end();
    };
    let late = finish.clone();
    // Not on the calling thread: titles hand the suspend to their window
    // thread and wait for it inside the routine.
    std::thread::spawn(move || {
        constrain_then_suspend(&constrain, &suspend);
        finish("title suspended");
    });
    std::thread::spawn(move || {
        std::thread::sleep(SUSPEND_DEADLINE);
        late("title did not finish suspending in time");
    });
}

/// Runs the constrained-change routines (constrained), then the suspend
/// routines (quiesced), as the console does when it suspends a title.
/// Grand Theft Auto VI's suspend routine waits for a state that only its
/// constrained routine brings about; called alone it never returns.
fn constrain_then_suspend(constrain: &[Registration], suspend: &[Registration]) {
    for &(_, routine, context) in constrain.iter().chain(suspend) {
        routine(1, context as *mut c_void);
    }
}

/// Set in the environment of a relaunched `victory`: an inherited handle
/// of the process it replaces.
const RELAUNCHED_FROM: &str = "VICTORY_RELAUNCHED_FROM";

/// Starts `victory` again with `args` (after the title has suspended); the
/// new process waits until this one has ended.
pub(crate) fn relaunch(args: Vec<std::ffi::OsString>) {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
    let mut me: HANDLE = std::ptr::null_mut();
    // SAFETY: an inheritable duplicate of our own process handle.
    let duplicated = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            GetCurrentProcess(),
            GetCurrentProcess(),
            &mut me,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        )
    } != 0;
    let spawned = std::env::current_exe().and_then(|exe| {
        let mut command = std::process::Command::new(exe);
        command.args(&args);
        if duplicated {
            command.env(RELAUNCHED_FROM, (me as usize).to_string());
        }
        command.spawn()
    });
    match spawned {
        Ok(child) => tracing::info!(pid = child.id(), ?args, "relaunched"),
        Err(e) => tracing::warn!(%e, "relaunch failed; the title just ends"),
    }
}

/// In a relaunched `victory`: waits (30 s at most) for the process it
/// replaces to be gone, so the two never share the title's files.
pub fn wait_for_previous() {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    let Some(handle) = std::env::var(RELAUNCHED_FROM)
        .ok()
        .and_then(|h| h.parse::<usize>().ok())
    else {
        return;
    };
    // SAFETY: removing our own variable before any other thread reads it.
    unsafe { std::env::remove_var(RELAUNCHED_FROM) };
    let handle = handle as HANDLE;
    // SAFETY: the inherited process handle the previous process gave us.
    unsafe {
        WaitForSingleObject(handle, 30_000);
        CloseHandle(handle);
    }
}

fn end() {
    // Compiled host pipelines kept for the next run.
    #[cfg(windows)]
    victory_gpu::pipeline::save_pipeline_cache();
    // SAFETY: ends this process, as the console ends a suspended title.
    unsafe { TerminateProcess(GetCurrentProcess(), 0) };
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "lifecycle",
        dlls: &[
            "api-ms-win-core-psm-appnotify-*",
            "ext-ms-win-rtcore-ntuser-window-*",
        ],
        exports: vec![
            hle_export!(RegisterAppStateChangeNotification),
            hle_export!(UnregisterAppStateChangeNotification),
            hle_export!(RegisterAppConstrainedChangeNotification),
            hle_export!(UnregisterAppConstrainedChangeNotification),
            hle_export!(CreateWindowExW),
            hle_export!(CreateWindowExA),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "system" fn noop(_: u8, _: *mut c_void) {}

    #[test]
    fn registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        UnregisterAppStateChangeNotification(a);
        let left: Vec<_> = routines().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    #[test]
    fn constrained_registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        assert_eq!(
            RegisterAppConstrainedChangeNotification(None, std::ptr::null_mut(), &mut a),
            ERROR_INVALID_PARAMETER
        );
        UnregisterAppConstrainedChangeNotification(a);
        let left: Vec<_> = constrained().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    /// What `recording_routine` saw: (context, flag) per call, in order.
    static CALLS: Mutex<Vec<(usize, u8)>> = Mutex::new(Vec::new());

    extern "system" fn recording_routine(flag: u8, context: *mut c_void) {
        CALLS.lock().unwrap().push((context as usize, flag));
    }

    #[test]
    fn constrained_routines_run_before_suspend_routines() {
        let r: Routine = recording_routine;
        constrain_then_suspend(&[(1, r, 10), (2, r, 11)], &[(3, r, 20), (4, r, 21)]);
        assert_eq!(*CALLS.lock().unwrap(), [(10, 1), (11, 1), (20, 1), (21, 1)]);
    }

    #[test]
    fn oversized_windows_shrink_to_the_screen() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(fit_into(&area, 3840, 2160), (1848, 1040));
        assert_eq!(fit_into(&area, 1280, 720), (1280, 720));
        assert_eq!(
            fit_into(&area, wm::CW_USEDEFAULT, 0),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn configured_window_limit_fits_4k_into_1080p() {
        let area = RECT {
            left: 100,
            top: 50,
            right: 2660,
            bottom: 1490,
        };
        assert_eq!(fit_with_limit(&area, 3840, 2160, Some((1920, 1080))), (1920, 1080));
        assert_eq!(fit_with_limit(&area, 640, 480, Some((1920, 1080))), (640, 480));
        assert_eq!(fit_with_limit(&area, 3840, 2160, None), (2560, 1440));
    }

    #[test]
    fn configured_window_limit_still_respects_the_work_area() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(fit_with_limit(&area, 3840, 2160, Some((2560, 1440))), (1848, 1040));
        assert_eq!(
            fit_with_limit(&area, wm::CW_USEDEFAULT, 0, Some((1920, 1080))),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn configured_window_limit_preserves_non_widescreen_proportions() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 2560,
            bottom: 1440,
        };
        assert_eq!(fit_with_limit(&area, 1600, 1200, Some((1920, 1080))), (1440, 1080));
    }

    #[test]
    fn window_limit_requires_two_positive_dimensions() {
        assert_eq!(parse_window_limit("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_window_limit(" 1920 X 1080 "), Some((1920, 1080)));
        for bad in [
            "",
            "1920",
            "0x1080",
            "1920x0",
            "-1x1080",
            "1920x-1",
            "1920x1080x2",
            "2147483648x1080",
        ] {
            assert_eq!(parse_window_limit(bad), None, "{bad}");
        }
    }

    #[test]
    fn hooked_windows_still_reach_the_title_procedure() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("before"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        let mut text = [0u16; 16];
        // SAFETY: our window; the buffer holds 16 characters.
        unsafe {
            wm::SetWindowTextW(hwnd, w!("after"));
            let n = wm::GetWindowTextW(hwnd, text.as_mut_ptr(), 16);
            assert_eq!(String::from_utf16_lossy(&text[..n as usize]), "after");
            assert!(!wm::GetPropW(hwnd, TITLE_PROC).is_null());
            wm::DestroyWindow(hwnd);
        }
    }

    thread_local! {
        /// The system commands `recording` received, low four bits cleared.
        static COMMANDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Stands in for a title's procedure: notes system commands and acts
    /// on none of them (no menu loop, no minimising).
    extern "system" fn recording(
        hwnd: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if message == wm::WM_SYSCOMMAND {
            COMMANDS.with(|c| c.borrow_mut().push(wparam & 0xFFF0));
            return 0;
        }
        // SAFETY: the system's default procedure, called as the system would.
        unsafe { wm::DefWindowProcW(hwnd, message, wparam, lparam) }
    }

    #[test]
    fn the_keyboard_menu_never_reaches_the_title() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("menu"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        // SAFETY: our window, owned by this thread, so each message is
        // handled before SendMessageW returns. `recording` stands in for the
        // title's procedure and the original is put back before the window
        // is destroyed.
        unsafe {
            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, recording as *const () as *mut c_void);
            let command = |c: u32, key: u8| {
                wm::SendMessageW(hwnd, wm::WM_SYSCOMMAND, c as usize, LPARAM::from(key))
            };
            command(wm::SC_KEYMENU, 0); // Alt or F10
            command(wm::SC_KEYMENU, b' '); // Alt+Space
            command(wm::SC_KEYMENU | 1, 0); // the low bits are the system's
            command(wm::SC_MINIMIZE, 0);
            wm::SetPropW(hwnd, TITLE_PROC, title);
            wm::DestroyWindow(hwnd);
        }
        let seen = COMMANDS.with(|c| c.borrow().clone());
        assert_eq!(seen, [wm::SC_MINIMIZE as usize]);
    }

    static CURSOR_TEST_LOCK: Mutex<()> = Mutex::new(());

    thread_local! {
        static RECORDED_MESSAGES: std::cell::RefCell<Vec<(u32, LPARAM)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    extern "system" fn record_messages(
        hwnd: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        RECORDED_MESSAGES.with(|m| m.borrow_mut().push((message, lparam)));
        if message == wm::WM_SETCURSOR {
            return 42;
        }
        // SAFETY: system default proc for synthetic test window.
        unsafe { wm::DefWindowProcW(hwnd, message, wparam, lparam) }
    }

    #[test]
    fn window_proc_tracks_mouse_move_wheel_sizemove_focus_and_forwards() {
        let _guard = CURSOR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = take_wheel();
        IN_SIZE_MOVE.store(false, Ordering::Release);
        RECORDED_MESSAGES.with(|m| m.borrow_mut().clear());

        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("input-test"),
            0,
            0,
            0,
            200,
            100,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        assert_eq!(title_hwnd(), hwnd as usize);

        // Creating a child window must not overwrite the tracked top-level HWND.
        let child = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("child"),
            wm::WS_CHILD,
            0,
            0,
            10,
            10,
            hwnd,
            null,
            null,
            std::ptr::null(),
        );
        assert_eq!(title_hwnd(), hwnd as usize);

        // SAFETY: `hwnd` is owned by this thread; original title proc is restored before destroy.
        unsafe {
            let mut rect = RECT::default();
            assert_ne!(wm::GetClientRect(hwnd, &mut rect), 0);
            let cw = rect.right - rect.left;
            let ch = rect.bottom - rect.top;
            assert!(cw > 0 && ch > 0);

            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, record_messages as *const () as *mut c_void);

            let mx = cw / 2;
            let my = ch / 4;
            let move_lp = ((my & 0xFFFF) << 16) | (mx & 0xFFFF);
            wm::SendMessageW(hwnd, wm::WM_MOUSEMOVE, 0, move_lp as LPARAM);
            let (u, v, in_client) = client_cursor().expect("WM_MOUSEMOVE recorded cursor");
            assert!(in_client);
            assert!((u - mx as f32 / cw as f32).abs() < 0.01);
            assert!((v - my as f32 / ch as f32).abs() < 0.01);

            wm::SendMessageW(hwnd, wm::WM_NCMOUSEMOVE, 0, 0);
            assert!(!client_cursor().unwrap().2);

            wm::SendMessageW(hwnd, wm::WM_MOUSEMOVE, 0, move_lp as LPARAM);
            assert!(client_cursor().unwrap().2);

            wm::SendMessageW(hwnd, wm::WM_MOUSEWHEEL, (120i16 as u16 as usize) << 16, 0);
            wm::SendMessageW(hwnd, wm::WM_MOUSEWHEEL, ((-240i16) as u16 as usize) << 16, 0);
            wm::SendMessageW(hwnd, wm::WM_MOUSEHWHEEL, (120i16 as u16 as usize) << 16, 0);
            assert_eq!(take_wheel(), (120, -120));
            assert_eq!(take_wheel(), (0, 0));

            wm::SendMessageW(hwnd, wm::WM_ENTERSIZEMOVE, 0, 0);
            assert!(in_size_move());
            wm::SendMessageW(hwnd, wm::WM_EXITSIZEMOVE, 0, 0);
            assert!(!in_size_move());

            let f0 = focus_serial();
            wm::SendMessageW(hwnd, wm::WM_SETFOCUS, 0, 0);
            let f1 = focus_serial();
            assert!(f1 > f0);
            wm::SendMessageW(hwnd, wm::WM_KILLFOCUS, 0, 0);
            let f2 = focus_serial();
            assert!(f2 > f1);
            assert!(!client_cursor().unwrap().2);

            wm::SetPropW(hwnd, TITLE_PROC, title);
            if !child.is_null() {
                wm::DestroyWindow(child);
            }
            wm::DestroyWindow(hwnd);
        }
        assert_eq!(title_hwnd(), 0);
        let seen = RECORDED_MESSAGES.with(|m| m.borrow().clone());
        assert!(seen.iter().any(|(m, _)| *m == wm::WM_MOUSEMOVE));
        assert!(seen.iter().any(|(m, _)| *m == wm::WM_MOUSEWHEEL));
        assert!(seen.iter().any(|(m, _)| *m == wm::WM_ENTERSIZEMOVE));
    }

    #[test]
    fn window_proc_setcursor_handles_htclient_and_htcaption() {
        let _guard = CURSOR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert!(should_hide_os_cursor_with(false, true, true));
        assert!(should_hide_os_cursor_with(false, false, false));
        assert!(!should_hide_os_cursor_with(false, false, true));
        assert!(!should_hide_os_cursor_with(true, true, true));
        assert!(!should_hide_os_cursor_with(true, false, false));

        if std::env::var_os("VICTORY_CURSOR_OS").is_some_and(|v| v == "1") {
            return;
        }

        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("cursor-test"),
            0,
            0,
            0,
            40,
            40,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());

        // SAFETY: `hwnd` is owned by this thread; original title proc restored before destroy.
        unsafe {
            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, record_messages as *const () as *mut c_void);

            // Cursor mode visible -> HTCLIENT intercepted (returns 1), HTCAPTION forwarded (returns 42).
            set_cursor_ui_state(true, true);
            RECORDED_MESSAGES.with(|m| m.borrow_mut().clear());
            assert_eq!(wm::SendMessageW(hwnd, wm::WM_SETCURSOR, hwnd as usize, wm::HTCLIENT as LPARAM), 1);
            assert_eq!(wm::SendMessageW(hwnd, wm::WM_SETCURSOR, hwnd as usize, wm::HTCAPTION as LPARAM), 42);
            let seen = RECORDED_MESSAGES.with(|m| m.borrow().clone());
            assert_eq!(seen, [(wm::WM_SETCURSOR, wm::HTCAPTION as LPARAM)]);

            // Camera-look mode (!cursor_mode) -> HTCLIENT intercepted (returns 1).
            set_cursor_ui_state(false, false);
            assert_eq!(wm::SendMessageW(hwnd, wm::WM_SETCURSOR, hwnd as usize, wm::HTCLIENT as LPARAM), 1);

            // Cursor mode with swapchain cursor hidden -> HTCLIENT forwarded to title/OS.
            set_cursor_ui_state(false, true);
            assert_eq!(wm::SendMessageW(hwnd, wm::WM_SETCURSOR, hwnd as usize, wm::HTCLIENT as LPARAM), 42);

            wm::SetPropW(hwnd, TITLE_PROC, title);
            wm::DestroyWindow(hwnd);
        }
    }

    #[test]
    fn window_proc_bridges_keydown_keyup_and_clears_on_killfocus() {
        let _guard = CURSOR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_window_keys();
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("key-bridge-test"),
            0,
            0,
            0,
            40,
            40,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());

        // SAFETY: `hwnd` is owned by this thread; original title proc restored before destroy.
        unsafe {
            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, record_messages as *const () as *mut c_void);

            wm::SendMessageW(hwnd, wm::WM_KEYDOWN, 0x0D, 0x001C_0001);
            wm::SendMessageW(hwnd, wm::WM_SYSKEYDOWN, 0x10, 0);
            assert_eq!(window_keys(), vec![0x0D, 0xA0]);

            wm::SendMessageW(hwnd, wm::WM_KEYUP, 0x0D, 0xC01C_0001);
            assert_eq!(window_keys(), vec![0xA0]);

            wm::SendMessageW(hwnd, wm::WM_KILLFOCUS, 0, 0);
            assert!(window_keys().is_empty());

            wm::SetPropW(hwnd, TITLE_PROC, title);
            wm::DestroyWindow(hwnd);
        }
    }
}
