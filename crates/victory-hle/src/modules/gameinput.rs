//! `gameinput.dll`: input devices from the host. Gamepads come from XInput
//! (host slots 0-3); while none is connected, the keyboard drives one
//! stand-in gamepad (`FALLBACK_BUTTONS`). The keyboard is a device of its own
//! too. A poll thread turns host state into readings (a new one per change)
//! and runs device callbacks on connect and disconnect.
#![allow(non_snake_case)]
// Title code calls these through the objects, passing pointers the real API
// would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::xasync::{E_INVALIDARG, E_NOTIMPL};
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleExport, HleModule};

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const MOUSE_KIND: u32 = 0x20;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// GDK `GameInputMouseButtons` flags.
const MOUSE_LEFT: u32 = 0x1;
const MOUSE_RIGHT: u32 = 0x2;
const MOUSE_MIDDLE: u32 = 0x4;
const MOUSE_BUTTON4: u32 = 0x8;
const MOUSE_BUTTON5: u32 = 0x10;

/// Devices: host pad slots, the KBM stand-in pad, the keyboard, and the mouse.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const MOUSE: usize = PADS + 2;
const DEVICES: usize = PADS + 3;

/// Readings kept per device for next/previous lookups.
const HISTORY: usize = 256;
const POLL: Duration = Duration::from_millis(8);

/// Standard PC GTA keyboard bindings onto the stand-in gamepad (virtual key, button).
/// Left/right modifier VKs (`0xA0..=0xA3`) match what `Host::keys()` reports.
/// Sticks: W/A/S/D left, Mouse delta + I/J/K/L right.
/// Mouse in Camera Mode: LMB=RT (Shoot/Gas), RMB=LT (Aim/Brake), MMB=R3, XBtn1=LB, XBtn2=RB.
/// Press F8 to toggle between Camera-Look Mode and on-screen Cursor Mode.
const FALLBACK_BUTTONS: [(u8, u32); 25] = [
    (0x1B, MENU),           // Esc (Pause Menu)
    (b'P', MENU),           // P (Pause Menu)
    (b'V', VIEW),           // V (Camera View)
    (b'M', VIEW),           // M (Interaction Menu)
    (0xA0, A),              // Left Shift (Sprint)
    (0xA1, A),              // Right Shift (Sprint)
    (0x0D, A),              // Enter (Confirm)
    (b'R', B),              // R (Reload / Melee)
    (0x08, B),              // Backspace (Back)
    (0x20, X),              // Space (Jump / Climb)
    (b'X', X),              // X (Jump alias)
    (b'F', Y),              // F (Enter / Exit Vehicle)
    (b'Y', Y),              // Y (Enter alias)
    (0x26, UP),             // Up Arrow (Phone / D-Pad Up)
    (0x28, DOWN),           // Down Arrow (D-Pad Down)
    (b'Z', DOWN),           // Z (Radar / D-Pad Down)
    (0x25, LEFT),           // Left Arrow (D-Pad Left)
    (0x27, RIGHT),          // Right Arrow (D-Pad Right)
    (b'E', RIGHT),          // E (Context / Interact)
    (0x09, LEFT_SHOULDER),  // Tab (Weapon Wheel)
    (b'Q', RIGHT_SHOULDER), // Q (Cover / Handbrake)
    (0xA2, LEFT_STICK),     // Left Ctrl (Crouch / Stealth / Horn)
    (0xA3, LEFT_STICK),     // Right Ctrl (Crouch / Stealth)
    (b'G', LEFT_STICK),     // G (Alternate L3)
    (b'C', RIGHT_STICK),    // C (Look Behind / Special)
];

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PadState {
    buttons: u32,
    left_trigger: f32,
    right_trigger: f32,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

/// GDK `GameInputMouseState` (40 bytes): buttons mask and cumulative deltas.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MouseState {
    buttons: u32,
    position_x: i64,
    position_y: i64,
    wheel_x: i64,
    wheel_y: i64,
}

const _: () = assert!(
    size_of::<KeyState>() == 12
        && size_of::<PadState>() == 28
        && size_of::<MouseState>() == 40
);

/// Host input, or a fake in tests.
trait Source: Send {
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Keys (and gated mouse VKs `0x01..=0x06`) held down; none without focus.
    fn keys(&mut self) -> Vec<KeyState>;
    /// Latest cumulative mouse state for `MOUSE_KIND` (`0x20`).
    fn mouse(&mut self) -> MouseState {
        MouseState::default()
    }
    /// Mouse-look stick `(right_x, right_y)`, wheel D-Pad pulse bits, and whether
    /// free on-screen Cursor Mode is active (`F8` toggle).
    fn pad_extras(&mut self) -> ((f32, f32), u32, bool) {
        ((0.0, 0.0), 0, false)
    }
    /// Motor speeds: low frequency (left), high frequency (right).
    fn rumble(&mut self, slot: usize, low: u16, high: u16);
}

fn further(x: f32, y: f32) -> f32 {
    if y.abs() > x.abs() { y } else { x }
}

fn fallback_pad(
    keys: &[KeyState],
    mouse_look: (f32, f32),
    wheel_buttons: u32,
    cursor_mode: bool,
) -> PadState {
    let down = |vk: u8| keys.iter().any(|k| k.virtual_key == vk);
    let on = |vk: u8| if down(vk) { 1.0 } else { 0.0 };
    let mut buttons = FALLBACK_BUTTONS
        .iter()
        .filter(|(vk, _)| down(*vk))
        .fold(0, |all, (_, bit)| all | bit)
        | wheel_buttons;
    let (left_trigger, right_trigger, mx, my) = if cursor_mode {
        // In Cursor Mode, clicks and mouse movement drive the on-screen cursor
        // and `MouseState` rather than firing weapons or spinning the camera.
        (0.0, 0.0, 0.0, 0.0)
    } else {
        if down(0x04) {
            buttons |= RIGHT_STICK; // Middle Mouse -> R3
        }
        if down(0x05) {
            buttons |= LEFT_SHOULDER; // XButton1 -> LB
        }
        if down(0x06) {
            buttons |= RIGHT_SHOULDER; // XButton2 -> RB
        }
        (on(0x02), on(0x01), mouse_look.0, mouse_look.1)
    };
    let kb_rx = on(b'L') - on(b'J');
    let kb_ry = on(b'I') - on(b'K');
    PadState {
        buttons,
        left_trigger,
        right_trigger,
        left_x: on(b'D') - on(b'A'),
        left_y: on(b'W') - on(b'S'),
        right_x: further(kb_rx, mx),
        right_y: further(kb_ry, my),
    }
}

/// Drift-resistant axis merge: preserves raw physical controller values when KBM
/// is idle (`kbm == 0.0`), lets KBM micro-aim beat resting pad drift (`< 0.08`),
/// and otherwise picks whichever is pushed further.
fn merge_axis(kbm: f32, pad: f32) -> f32 {
    if kbm == 0.0 {
        pad
    } else if pad.abs() < 0.08 {
        kbm
    } else {
        further(kbm, pad)
    }
}

/// Two pads as one (`a` = KBM/accumulated, `b` = physical host pad): buttons of
/// both, and drift-resistant merge of each axis and trigger.
fn merged(a: PadState, b: PadState) -> PadState {
    PadState {
        buttons: a.buttons | b.buttons,
        left_trigger: merge_axis(a.left_trigger, b.left_trigger),
        right_trigger: merge_axis(a.right_trigger, b.right_trigger),
        left_x: merge_axis(a.left_x, b.left_x),
        left_y: merge_axis(a.left_y, b.left_y),
        right_x: merge_axis(a.right_x, b.right_x),
        right_y: merge_axis(a.right_y, b.right_y),
    }
}

/// Smooths raw mouse deltas into a right-stick vector with radial clamping and
/// deadzone snap so resting mouse produces exact `0.0` (no reading churn).
#[derive(Clone, Copy, Debug, Default)]
struct MouseLookFilter {
    vel_x: f32,
    vel_y: f32,
}

impl MouseLookFilter {
    fn reset(&mut self) {
        self.vel_x = 0.0;
        self.vel_y = 0.0;
    }

    fn update(&mut self, dx: i32, dy: i32, sensitivity: f32, invert_y: bool) -> (f32, f32) {
        let sens = sensitivity.max(1.0);
        let y_sign = if invert_y { 1.0 } else { -1.0 };
        self.vel_x = self.vel_x * 0.45 + (dx as f32) / sens;
        self.vel_y = self.vel_y * 0.45 + (dy as f32) * y_sign / sens;
        if self.vel_x.abs() < 0.005 {
            self.vel_x = 0.0;
        }
        if self.vel_y.abs() < 0.005 {
            self.vel_y = 0.0;
        }
        let len = self.vel_x.hypot(self.vel_y);
        if len > 1.0 {
            (self.vel_x / len, self.vel_y / len)
        } else {
            (self.vel_x, self.vel_y)
        }
    }
}

/// Converts mouse-wheel notches (`120` per detent) into `48 ms` D-Pad pulses
/// separated by `16 ms` gaps so 20-30 FPS title polling sees every notch cleanly.
#[derive(Clone, Copy, Debug, Default)]
struct WheelPulser {
    remainder: i64,
    queued: i32,
    active_button: u32,
    phase_until_ms: u64,
    in_gap: bool,
}

impl WheelPulser {
    const HOLD_MS: u64 = 48;
    const GAP_MS: u64 = 16;

    fn clear(&mut self) {
        *self = Self::default();
    }

    fn push(&mut self, delta: i64) {
        self.remainder += delta;
        let steps = (self.remainder / 120) as i32;
        if steps != 0 {
            self.remainder -= i64::from(steps) * 120;
            self.queued = (self.queued + steps).clamp(-16, 16);
        }
    }

    fn tick(&mut self, now_ms: u64) -> u32 {
        if self.active_button != 0 {
            if now_ms < self.phase_until_ms {
                return self.active_button;
            }
            self.active_button = 0;
            self.in_gap = true;
            self.phase_until_ms = now_ms.saturating_add(Self::GAP_MS);
            return 0;
        }
        if self.in_gap {
            if now_ms < self.phase_until_ms {
                return 0;
            }
            self.in_gap = false;
        }
        if self.queued > 0 {
            self.queued -= 1;
            self.active_button = UP;
            self.phase_until_ms = now_ms.saturating_add(Self::HOLD_MS);
            return UP;
        }
        if self.queued < 0 {
            self.queued += 1;
            self.active_button = DOWN;
            self.phase_until_ms = now_ms.saturating_add(Self::HOLD_MS);
            return DOWN;
        }
        0
    }
}

/// Whether the title sees one pad that is always there, fed by the keyboard
/// and by every host pad (on by default; `VICTORY_ONE_PAD=0` shows each host
/// pad as a device of its own, and the stand-in only while there is none).
/// The title keeps to the pad it started with: with the stand-in taken away
/// when a pad was switched on, it showed its controller-disconnected screen
/// until the pad was off again. It also keeps a pad the host lists twice (its
/// own slot and a remapper's) from being two players.
fn one_pad() -> bool {
    std::env::var_os("VICTORY_ONE_PAD").is_none_or(|v| v != "0")
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself. The title reads `id` (+0x20); the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(std::mem::offset_of!(DeviceInfo, id) == 0x20);

fn kind(index: usize) -> u32 {
    match index {
        KEYBOARD => KEYBOARD_KIND,
        MOUSE => MOUSE_KIND,
        _ => GAMEPAD_KIND,
    }
}

impl Device {
    fn new(index: usize) -> Device {
        let mut id = [0; 32];
        id[..7].copy_from_slice(b"victory");
        id[31] = index as u8 + 1;
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            // Low and high frequency.
            rumble_motors: if index <= FALLBACK { 0x3 } else { 0 },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
    Mouse(MouseState),
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
            Value::Mouse(_) => MOUSE_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> Event {
        Event {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct Event {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
}

struct State {
    slots: [Slot; DEVICES],
    callbacks: Vec<Registration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<Event>,
    /// Device callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Apart from the state: XInput can be slow, and the title reads
    /// readings every frame.
    source: Mutex<Box<dyn Source>>,
    /// The title sees one pad ([`one_pad`]).
    one_pad: bool,
    state: Mutex<State>,
    /// A device callback returned.
    returned: Condvar,
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind the mutex or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    fn new(source: Box<dyn Source>, one_pad: bool) -> &'static Hub {
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(Device::new),
            source: Mutex::new(source),
            one_pad,
            state: Mutex::new(State {
                slots: Default::default(),
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
            }),
            returned: Condvar::new(),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    /// A hub over the host's input, polled from now on.
    fn start(source: Box<dyn Source>) -> &'static Hub {
        let hub = Hub::new(source, one_pad());
        hub.poll();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(POLL);
                hub.poll();
            }
        });
        hub
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Reads the host once; device callbacks run on the calling thread.
    fn poll(&self) {
        let (pads, keys, mouse, extras) = {
            let mut source = lock(&self.source);
            let pads: [_; PADS] = std::array::from_fn(|slot| source.pad(slot));
            let keys = source.keys();
            let mouse = source.mouse();
            let extras = source.pad_extras();
            (pads, keys, mouse, extras)
        };
        let events = self
            .state()
            .poll(&self.devices, self.one_pad, pads, keys, mouse, extras);
        self.deliver(events);
    }

    /// Runs callbacks, never under our lock: they call back into us. One
    /// unregistered since the event was taken does not run, and
    /// `UnregisterCallback` waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        for e in events {
            let r = e.registration;
            {
                let mut state = self.state();
                if !state.callbacks.iter().any(|c| c.token == r.token) {
                    continue;
                }
                state.running.push((r.token, me));
            }
            (r.callback)(
                r.token,
                r.context as *mut c_void,
                &self.devices[e.device],
                e.timestamp,
                e.current,
                e.previous,
            );
            let mut state = self.state();
            if let Some(i) = state.running.iter().position(|&x| x == (r.token, me)) {
                state.running.swap_remove(i);
            }
            self.returned.notify_all();
        }
    }

    fn index(&self, device: *const Device) -> Option<usize> {
        self.devices.iter().position(|d| std::ptr::eq(d, device))
    }
}

impl State {
    fn poll(
        &mut self,
        devices: &[Device; DEVICES],
        one_pad: bool,
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        mouse: MouseState,
        (mouse_look, wheel_buttons, cursor_mode): ((f32, f32), u32, bool),
    ) -> Vec<Event> {
        let now = now();
        let mut events = std::mem::take(&mut self.pending);
        let kbm_pad = fallback_pad(&keys, mouse_look, wheel_buttons, cursor_mode);
        let kb_keys: Vec<KeyState> = keys
            .into_iter()
            .filter(|k| k.virtual_key >= 0x08)
            .collect();
        if one_pad {
            let pad = pads.into_iter().flatten().fold(kbm_pad, merged);
            self.update(devices, FALLBACK, Some(Value::Pad(pad)), now, &mut events);
            self.update(devices, KEYBOARD, Some(Value::Keys(kb_keys)), now, &mut events);
            self.update(devices, MOUSE, Some(Value::Mouse(mouse)), now, &mut events);
            return events;
        }
        let any_pad = pads.iter().any(Option::is_some);
        for (slot, pad) in pads.into_iter().enumerate() {
            self.update(devices, slot, pad.map(Value::Pad), now, &mut events);
        }
        let fallback = (!any_pad).then_some(Value::Pad(kbm_pad));
        self.update(devices, FALLBACK, fallback, now, &mut events);
        self.update(devices, KEYBOARD, Some(Value::Keys(kb_keys)), now, &mut events);
        self.update(devices, MOUSE, Some(Value::Mouse(mouse)), now, &mut events);
        events
    }

    /// Records a device's state: a reading when it changed, callbacks when
    /// the device came or went.
    fn update(
        &mut self,
        devices: &[Device; DEVICES],
        index: usize,
        value: Option<Value>,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let slot = &mut self.slots[index];
        let was = slot.connected;
        slot.connected = value.is_some();
        if let Some(value) = value
            && (!was || slot.history.back().is_none_or(|r| r.value != value))
        {
            slot.sequence += 1;
            if slot.history.len() == HISTORY {
                slot.history.pop_front();
            }
            slot.history.push_back(Ref::new(Reading {
                vtable: &READING,
                refs: AtomicU32::new(1),
                device: &devices[index],
                sequence: slot.sequence,
                timestamp: now,
                value,
            }));
        }
        if was != slot.connected {
            tracing::info!(device = index, connected = !was, "input device");
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, index) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(index, now, current, previous)),
            );
        }
    }
}

#[cfg(windows)]
fn host() -> Box<dyn Source> {
    Box::new(host::Host::default())
}

#[cfg(not(windows))]
fn host() -> Box<dyn Source> {
    struct Nothing;
    impl Source for Nothing {
        fn pad(&mut self, _: usize) -> Option<PadState> {
            None
        }
        fn keys(&mut self) -> Vec<KeyState> {
            Vec::new()
        }
        fn rumble(&mut self, _: usize, _: u16, _: u16) {}
    }
    Box::new(Nothing)
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| Hub::start(host()));

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    tracing::info!("GameInputCreate");
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

extern "system" fn no_interface(_: *const c_void, _iid: *const c_void, out: Out) -> HRESULT {
    tracing::warn!("input QueryInterface not implemented");
    fail(out, E_NOINTERFACE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    now()
}

extern "system" fn current_reading(
    this: This,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let state = hub.state();
    let newest = |i: usize| {
        state.slots[i]
            .history
            .back()
            .filter(|r| r.kind() & kind != 0)
    };
    let found = if device.is_null() {
        (0..DEVICES)
            .filter(|&i| state.slots[i].connected)
            .filter_map(newest)
            .max_by_key(|r| r.timestamp)
    } else {
        match hub.index(device) {
            None => return fail(out, E_INVALIDARG),
            Some(i) if !state.slots[i].connected => return fail(out, DEVICE_DISCONNECTED),
            Some(i) => newest(i),
        }
    };
    tracing::trace!(
        kind = format_args!("{kind:#x}"),
        device = ?hub.index(device),
        sequence = found.map(|r| r.sequence),
        "current reading"
    );
    hand(out, found)
}

/// The reading after (`forward`) or before `reference` on its device.
fn relative(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    // shortcut: with no device given only the reference's device is walked,
    // not every device's readings in time order; the title walks
    // only keyboard readings (one device) and the current pad's previous one.
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let state = hub.state();
    let history = &state.slots[index].history;
    if history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    let wanted = |r: &&Ref| r.kind() & kind != 0;
    let found = if forward {
        history
            .iter()
            .filter(|r| r.sequence > reference.sequence)
            .find(wanted)
    } else {
        history
            .iter()
            .rev()
            .filter(|r| r.sequence < reference.sequence)
            .find(wanted)
    };
    hand(out, found)
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let state = hub.state();
    let history = &state.slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

// shortcut: reading callbacks never run; the title registers none
// (it polls).
extern "system" fn register_reading(
    _: This,
    _device: *const c_void,
    _kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    tracing::warn!("input reading callbacks are never called");
    token(out)
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (the title registers holding a lock its callback
/// takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback else {
        return E_INVALIDARG;
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        let mut state = hub.state();
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| state.slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        tracing::info!(
            kind = format_args!("{kind:#x}"),
            enumeration,
            devices = found.len(),
            "input device callback"
        );
        state.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION {
            state.pending.append(&mut found);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    hub.deliver(found);
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    token(out)
}

fn forget(this: This, token: u64) {
    let mut state = hub(this).state();
    state.callbacks.retain(|r| r.token != token);
    state.pending.retain(|e| e.registration.token != token);
}
extern "system" fn stop_callback(this: This, token: u64) {
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut state = hub.state();
    loop {
        let running = state.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        state = match deadline {
            None => hub.returned.wait(state).unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(state, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    tracing::warn!("input dispatcher not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, _kind: u32, out: Out) -> HRESULT {
    tracing::warn!("aggregate input device not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    match hub(this).devices.iter().find(|d| d.info.id == *id) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device(_: This, _value: *const c_void, out: Out) -> HRESULT {
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {}

static INPUT: Table<22> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other as *const (),      // 0x50 guide button
    register_other as *const (),      // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device as *const (),         // 0x88 by object
    find_device as *const (),         // 0x90 by platform handle
    find_device as *const (),         // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.hub().state().slots[device.index].connected {
        STATUS_CONNECTED
    } else {
        0
    }
}
// shortcut: trigger motors are dropped (XInput has none). The host API that
// drives them without a redistributable is Windows.Gaming.Input (WinRT
// `Gamepad.Vibration`), which needs WinRT bindings this crate lacks outside
// diagnostics builds; the title sends trigger levels in slot 0x50.
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    let one_pad = device.hub().one_pad;
    if device.index >= PADS && !(one_pad && device.index == FALLBACK) {
        return;
    }
    let [low, high, ..] = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let speed = |level: f32| (level.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
    let mut source = lock(&device.hub().source);
    if one_pad {
        // The one pad is every host pad.
        for slot in 0..PADS {
            if source.pad(slot).is_some() {
                source.rumble(slot, speed(low), speed(high));
            }
        }
    } else {
        source.rumble(device.index, speed(low), speed(high));
    }
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    tracing::warn!(
        slot = format_args!("{SLOT:#x}"),
        "input device method not implemented"
    );
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let left = reading(this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, kind: u32) -> u64 {
    let reading = reading(this);
    if reading.kind() & kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        Value::Pad(_) | Value::Mouse(_) => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn mouse_state(this: *const Reading, out: *mut MouseState) -> bool {
    match &reading(this).value {
        Value::Mouse(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    match &reading(this).value {
        Value::Pad(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing(_: *const c_void) -> usize {
    0
}

static READING: Table<25> = Table([
    no_interface as *const (),      // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing as *const (),           // 0x38 raw report
    nothing as *const (),           // 0x40 controller axes, buttons, switches
    nothing as *const (),           // 0x48
    nothing as *const (),           // 0x50
    nothing as *const (),           // 0x58
    nothing as *const (),           // 0x60
    nothing as *const (),           // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    mouse_state as *const (),       // 0x80 mouse
    nothing as *const (),           // 0x88 touch
    nothing as *const (),           // 0x90
    nothing as *const (),           // 0x98 motion
    nothing as *const (),           // 0xa0 arcade stick
    nothing as *const (),           // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing as *const (),           // 0xb8 racing wheel
    nothing as *const (),           // 0xc0 UI navigation
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "gameinput",
        dlls: &["gameinput.dll"],
        // Titles import it by ordinal only.
        exports: vec![HleExport {
            name: "#1",
            address: GameInputCreate as *const () as usize,
        }],
    }
}

/// Host input: XInput pads, PC GTA KBM+mouse remapping, and on-screen cursor IPC.
#[cfg(windows)]
mod host {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, HWND, INVALID_HANDLE_VALUE, POINT, RECT};
    use windows_sys::Win32::Graphics::Gdi::{ClientToScreen, ScreenToClient};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
        PAGE_READWRITE, UnmapViewOfFile,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetClientRect, GetCursorPos, GetForegroundWindow, GetWindowThreadProcessId, SetCursorPos,
    };

    use super::*;
    use crate::modules::lifecycle;

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    /// Mouse virtual keys polled when focused inside client rect (skips 0x03 VK_CANCEL and 0x07).
    const MOUSE_VKS: [(u8, u32); 5] = [
        (0x01, MOUSE_LEFT),
        (0x02, MOUSE_RIGHT),
        (0x04, MOUSE_MIDDLE),
        (0x05, MOUSE_BUTTON4),
        (0x06, MOUSE_BUTTON5),
    ];
    const VK_F8: u8 = 0x77;

    /// Must match `crates/victory-gpu/src/host/cursor.rs` exactly.
    const CURSOR_MAGIC: u32 = 0x5643_5552; // "VCUR"
    const CURSOR_VERSION: u32 = 1;
    const CURSOR_SHARED_SIZE: u32 = 64;

    #[repr(C)]
    struct SharedCursor {
        magic: AtomicU32,
        version: AtomicU32,
        packed: AtomicU64,
    }

    const _: () = assert!(size_of::<SharedCursor>() <= CURSOR_SHARED_SIZE as usize);

    pub(super) fn pack_cursor(
        u: f32,
        v: f32,
        visible: bool,
        pressed: bool,
        cursor_mode: bool,
        initialized: bool,
        seq: u32,
    ) -> u64 {
        let x_q16 = (u.clamp(0.0, 1.0) * 65535.0).round() as u64;
        let y_q16 = (v.clamp(0.0, 1.0) * 65535.0).round() as u64;
        let seq = u64::from(seq & 0x0FFF_FFFF);
        x_q16
            | (y_q16 << 16)
            | (u64::from(visible) << 32)
            | (u64::from(pressed) << 33)
            | (u64::from(cursor_mode) << 34)
            | (u64::from(initialized) << 35)
            | (seq << 36)
    }

    struct SharedCursorMapping {
        handle: HANDLE,
        view: MEMORY_MAPPED_VIEW_ADDRESS,
    }

    // SAFETY: `SharedCursor` holds only atomics inside a process-lifetime pagefile mapping.
    unsafe impl Send for SharedCursorMapping {}
    // SAFETY: as above.
    unsafe impl Sync for SharedCursorMapping {}

    impl SharedCursorMapping {
        fn open() -> Option<Self> {
            let name = format!("Local\\victory-cursor-{}", std::process::id());
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            // SAFETY: creating/opening a 64-byte named pagefile mapping with null-terminated UTF-16 name.
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    CURSOR_SHARED_SIZE,
                    wide.as_ptr(),
                )
            };
            if handle.is_null() {
                return None;
            }
            // SAFETY: mapping the 64-byte view just opened.
            let view =
                unsafe { MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, CURSOR_SHARED_SIZE as usize) };
            if view.Value.is_null() {
                // SAFETY: valid handle from `CreateFileMappingW`.
                unsafe { CloseHandle(handle) };
                return None;
            }
            // SAFETY: `view.Value` is non-null, page-aligned, and >= 64 bytes.
            let hdr = unsafe { &*view.Value.cast::<SharedCursor>() };
            hdr.magic.store(CURSOR_MAGIC, Ordering::Release);
            hdr.version.store(CURSOR_VERSION, Ordering::Release);
            Some(Self { handle, view })
        }

        fn publish(&self, packed: u64) {
            // SAFETY: `view.Value` is non-null, page-aligned, and >= 64 bytes.
            let hdr = unsafe { &*self.view.Value.cast::<SharedCursor>() };
            hdr.magic.store(CURSOR_MAGIC, Ordering::Release);
            hdr.version.store(CURSOR_VERSION, Ordering::Release);
            hdr.packed.store(packed, Ordering::Release);
        }
    }

    impl Drop for SharedCursorMapping {
        fn drop(&mut self) {
            // SAFETY: `view` and `handle` came from `SharedCursorMapping::open`.
            unsafe {
                UnmapViewOfFile(self.view);
                CloseHandle(self.handle);
            }
        }
    }

    fn shared_cursor() -> Option<&'static SharedCursorMapping> {
        static MAPPING: OnceLock<Option<SharedCursorMapping>> = OnceLock::new();
        MAPPING.get_or_init(SharedCursorMapping::open).as_ref()
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum WheelPadMode {
        CursorOnly,
        Always,
        Off,
    }

    fn wheel_pad_mode() -> WheelPadMode {
        static MODE: OnceLock<WheelPadMode> = OnceLock::new();
        *MODE.get_or_init(|| {
            match std::env::var("VICTORY_WHEEL_PAD")
                .ok()
                .map(|s| s.to_ascii_lowercase())
                .as_deref()
            {
                Some("off" | "0") => WheelPadMode::Off,
                Some("always" | "1") => WheelPadMode::Always,
                _ => WheelPadMode::CursorOnly,
            }
        })
    }

    fn mouse_sensitivity() -> f32 {
        static SENS: OnceLock<f32> = OnceLock::new();
        *SENS.get_or_init(|| {
            std::env::var("VICTORY_MOUSE_SENSITIVITY")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(24.0)
        })
    }

    fn mouse_invert_y() -> bool {
        static INV: OnceLock<bool> = OnceLock::new();
        *INV.get_or_init(|| std::env::var_os("VICTORY_MOUSE_INVERT_Y").is_some_and(|v| v == "1"))
    }

    fn initial_cursor_mode() -> bool {
        std::env::var_os("VICTORY_CURSOR").is_some_and(|v| v == "1")
    }

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
        cursor_mode: bool,
        f8_down: bool,
        was_focused: bool,
        last_focus_serial: u64,
        suppress_mouse_mask: u8,
        recenter_pending: bool,
        look: MouseLookFilter,
        wheel: WheelPulser,
        mouse: MouseState,
        last_look: (f32, f32),
        last_wheel_buttons: u32,
        last_cursor_client: Option<(i32, i32)>,
        cursor_u: f32,
        cursor_v: f32,
        cursor_seq: u32,
    }

    impl Default for Host {
        fn default() -> Self {
            let cursor_mode = initial_cursor_mode();
            Self {
                empty_until: [None; PADS],
                cursor_mode,
                f8_down: false,
                was_focused: false,
                last_focus_serial: 0,
                suppress_mouse_mask: 0,
                recenter_pending: !cursor_mode,
                look: MouseLookFilter::default(),
                wheel: WheelPulser::default(),
                mouse: MouseState::default(),
                last_look: (0.0, 0.0),
                last_wheel_buttons: 0,
                last_cursor_client: None,
                cursor_u: 0.5,
                cursor_v: 0.5,
                cursor_seq: 0,
            }
        }
    }

    impl Host {
        fn publish_cursor(&mut self, visible: bool, pressed: bool) {
            self.cursor_seq = self.cursor_seq.wrapping_add(1);
            lifecycle::set_cursor_ui_state(visible, self.cursor_mode);
            if let Some(shared) = shared_cursor() {
                shared.publish(pack_cursor(
                    self.cursor_u,
                    self.cursor_v,
                    visible,
                    pressed,
                    self.cursor_mode,
                    true,
                    self.cursor_seq,
                ));
            }
        }
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            if self.empty_until[slot].is_some_and(|until| Instant::now() < until) {
                return None;
            }
            let mut state = XINPUT_STATE::default();
            // SAFETY: room for the state.
            if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
                self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
                return None;
            }
            self.empty_until[slot] = None;
            #[allow(unused_mut)]
            let mut pad = convert(&state.Gamepad);
            // The scripted keys (`VICTORY_KEYS`) also press the first
            // pad's buttons, by the stand-in pad's table: with a real pad
            // connected the stand-in is not there, and a title that listens
            // to the pad (the title's start screen) saw nothing.
            #[cfg(any(feature = "diagnostics", feature = "keys"))]
            if slot == 0 {
                for vk in scripted() {
                    if let Some(&(_, button)) =
                        super::FALLBACK_BUTTONS.iter().find(|(key, _)| *key == vk)
                    {
                        pad.buttons |= button;
                    }
                }
            }
            Some(pad)
        }

        fn keys(&mut self) -> Vec<KeyState> {
            #[cfg(any(feature = "diagnostics", feature = "keys"))]
            let mut down = scripted();
            #[cfg(not(any(feature = "diagnostics", feature = "keys")))]
            let mut down = Vec::new();
            for vk in lifecycle::window_keys() {
                if vk >= 0x08 && !down.contains(&vk) {
                    down.push(vk);
                }
            }

            let hwnd = focused_hwnd();
            if hwnd.is_null() {
                self.was_focused = false;
                self.recenter_pending = !self.cursor_mode;
                self.last_cursor_client = None;
                self.look.reset();
                self.wheel.clear();
                self.mouse.buttons = 0;
                self.last_look = (0.0, 0.0);
                self.last_wheel_buttons = 0;
                self.f8_down = false;
                self.publish_cursor(false, false);
                return down
                    .into_iter()
                    .map(|vk| KeyState {
                        // SAFETY: plain query.
                        scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
                        virtual_key: vk,
                        ..Default::default()
                    })
                    .collect();
            }

            let serial = lifecycle::focus_serial();
            if !self.was_focused || serial != self.last_focus_serial {
                self.was_focused = true;
                self.last_focus_serial = serial;
                self.recenter_pending = !self.cursor_mode;
                self.last_cursor_client = None;
                // Latch any mouse button already down during the focus click so
                // clicking into the window never fires a weapon or menu click.
                self.suppress_mouse_mask = 0;
                for (bit, &(vk, _)) in MOUSE_VKS.iter().enumerate() {
                    // SAFETY: plain key state query.
                    if unsafe { GetAsyncKeyState(i32::from(vk)) } < 0 {
                        self.suppress_mouse_mask |= 1 << bit;
                    }
                }
            }

            // F8 edge toggle between Camera-Look Mode and on-screen Cursor Mode.
            // SAFETY: plain key state query.
            let f8_now = unsafe { GetAsyncKeyState(i32::from(VK_F8)) } < 0;
            if f8_now && !self.f8_down {
                self.cursor_mode = !self.cursor_mode;
                self.look.reset();
                self.last_look = (0.0, 0.0);
                self.recenter_pending = !self.cursor_mode;
                self.last_cursor_client = None;
                tracing::info!(cursor_mode = self.cursor_mode, "toggled cursor mode (F8)");
            }
            self.f8_down = f8_now;

            let in_size_move = lifecycle::in_size_move();
            let client_info = query_client_cursor(hwnd);
            let in_client = !in_size_move
                && client_info
                    .map(|(_, _, _, _, inside)| inside)
                    .or_else(|| lifecycle::client_cursor().map(|(_, _, inside)| inside))
                    .unwrap_or(false);

            // Safe mouse VK polling: focused, inside client rect, not dragging titlebar,
            // and past any focus-gaining click latch.
            let mut mouse_buttons = 0u32;
            for (bit, &(vk, flag)) in MOUSE_VKS.iter().enumerate() {
                // SAFETY: plain key state query.
                let is_down = unsafe { GetAsyncKeyState(i32::from(vk)) } < 0;
                if !is_down {
                    self.suppress_mouse_mask &= !(1 << bit);
                    continue;
                }
                if (self.suppress_mouse_mask & (1 << bit)) != 0 || !in_client {
                    continue;
                }
                mouse_buttons |= flag;
                if !down.contains(&vk) {
                    down.push(vk);
                }
            }
            self.mouse.buttons = mouse_buttons;

            // Mouse wheel accumulation -> `MouseState` + deterministic 48ms D-Pad pulse.
            let (wx, wy) = lifecycle::take_wheel();
            self.mouse.wheel_x = self.mouse.wheel_x.wrapping_add(wx);
            self.mouse.wheel_y = self.mouse.wheel_y.wrapping_add(wy);
            let wheel_for_pad = match wheel_pad_mode() {
                WheelPadMode::Always => true,
                WheelPadMode::CursorOnly => self.cursor_mode,
                WheelPadMode::Off => false,
            };
            if wheel_for_pad {
                if wy != 0 {
                    self.wheel.push(wy);
                }
                self.last_wheel_buttons = self.wheel.tick(now() / 1000);
            } else {
                self.wheel.clear();
                self.last_wheel_buttons = 0;
            }

            // Cursor Mode vs Camera-Look Mode update + shared swapchain cursor publication.
            if self.cursor_mode {
                if let Some((px, py, cw, ch, inside)) = client_info {
                    if inside {
                        self.cursor_u = (px as f32 / cw as f32).clamp(0.0, 1.0);
                        self.cursor_v = (py as f32 / ch as f32).clamp(0.0, 1.0);
                    }
                    if let Some((lx, ly)) = self.last_cursor_client {
                        self.mouse.position_x = self.mouse.position_x.wrapping_add(i64::from(px - lx));
                        self.mouse.position_y = self.mouse.position_y.wrapping_add(i64::from(py - ly));
                    }
                    self.last_cursor_client = Some((px, py));
                } else if let Some((u, v, inside)) = lifecycle::client_cursor()
                    && inside
                {
                    self.cursor_u = u;
                    self.cursor_v = v;
                }
                self.last_look = (0.0, 0.0);
                self.publish_cursor(in_client, (mouse_buttons & MOUSE_LEFT) != 0);
            } else {
                let mut dx = 0i32;
                let mut dy = 0i32;
                if !in_size_move
                    && let Some((px, py, cw, ch, _)) = client_info
                {
                    let cx = cw / 2;
                    let cy = ch / 2;
                    if self.recenter_pending {
                        warp_client_center(hwnd, cx, cy);
                        self.recenter_pending = false;
                    } else {
                        dx = px - cx;
                        dy = py - cy;
                        if dx != 0 || dy != 0 {
                            warp_client_center(hwnd, cx, cy);
                        }
                    }
                }
                self.mouse.position_x = self.mouse.position_x.wrapping_add(i64::from(dx));
                self.mouse.position_y = self.mouse.position_y.wrapping_add(i64::from(dy));
                self.last_look =
                    self.look
                        .update(dx, dy, mouse_sensitivity(), mouse_invert_y());
                self.publish_cursor(false, false);
            }

            // Left/right modifier keys only, as devices report them; exclude F8 toggle key.
            let held: Vec<u8> = (0x08..=0xFE_u8)
                .filter(|vk| !(0x10..=0x12).contains(vk) && *vk != VK_F8 && !down.contains(vk))
                // SAFETY: plain queries.
                .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                .collect();
            down.extend(held);

            down.into_iter()
                .map(|vk| KeyState {
                    // SAFETY: plain query.
                    scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
                    virtual_key: vk,
                    // shortcut: no characters (code point 0);
                    // the title reads scan codes and virtual keys only.
                    ..Default::default()
                })
                .collect()
        }

        fn mouse(&mut self) -> MouseState {
            self.mouse
        }

        fn pad_extras(&mut self) -> ((f32, f32), u32, bool) {
            (self.last_look, self.last_wheel_buttons, self.cursor_mode)
        }

        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            let speeds = XINPUT_VIBRATION {
                wLeftMotorSpeed: low,
                wRightMotorSpeed: high,
            };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    fn query_client_cursor(hwnd: HWND) -> Option<(i32, i32, i32, i32, bool)> {
        let mut pt = POINT::default();
        let mut rect = RECT::default();
        // SAFETY: valid `hwnd` and output pointers.
        unsafe {
            if GetCursorPos(&mut pt) == 0
                || ScreenToClient(hwnd, &mut pt) == 0
                || GetClientRect(hwnd, &mut rect) == 0
            {
                return None;
            }
        }
        let cw = rect.right - rect.left;
        let ch = rect.bottom - rect.top;
        if cw <= 0 || ch <= 0 {
            return None;
        }
        let inside = pt.x >= 0 && pt.y >= 0 && pt.x < cw && pt.y < ch;
        Some((pt.x, pt.y, cw, ch, inside))
    }

    fn warp_client_center(hwnd: HWND, cx: i32, cy: i32) {
        let mut pt = POINT { x: cx, y: cy };
        // SAFETY: valid `hwnd` and `POINT`.
        unsafe {
            if ClientToScreen(hwnd, &mut pt) != 0 {
                SetCursorPos(pt.x, pt.y);
            }
        }
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts (`<from>-<to>:<key>`: all that
    /// time), with or without focus. A key is a
    /// letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Space`, `Shift`,
    /// `Ctrl`, `Up`, `Down`, `Left`, `Right`.
    #[cfg(any(feature = "diagnostics", feature = "keys"))]
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                let vk = match key.trim() {
                    "Enter" => 0x0D,
                    "Esc" => 0x1B,
                    "Tab" => 0x09,
                    "Back" => 0x08,
                    "Space" => 0x20,
                    "Shift" => 0xA0,
                    "Ctrl" => 0xA2,
                    "Up" => 0x26,
                    "Down" => 0x28,
                    "Left" => 0x25,
                    "Right" => 0x27,
                    k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
                        k.as_bytes()[0].to_ascii_uppercase()
                    }
                    _ => return None,
                };
                let time = |t: &str| Duration::try_from_secs_f64(t.trim().parse().ok()?).ok();
                let (from, to) = match at.split_once('-') {
                    Some((from, to)) => (time(from)?, time(to)?),
                    None => (time(at)?, time(at)? + Duration::from_millis(200)),
                };
                Some((from, to, vk))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        let now = start.elapsed();
        keys.iter()
            .filter(|&&(from, to, _)| (from..to).contains(&now))
            .map(|&(_, _, vk)| vk)
            .collect()
    }

    /// Foreground window if it belongs to our title (or fallback presenter window),
    /// rejecting modal dialogs when a main title window is tracked.
    fn focused_hwnd() -> HWND {
        // SAFETY: plain Win32 query.
        let fg = unsafe { GetForegroundWindow() };
        if fg.is_null() {
            return std::ptr::null_mut();
        }
        let tracked = lifecycle::title_hwnd();
        if tracked != 0 {
            return if fg as usize == tracked {
                fg
            } else {
                std::ptr::null_mut()
            };
        }
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            GetWindowThreadProcessId(fg, &mut process);
            if process == GetCurrentProcessId() {
                fg
            } else {
                std::ptr::null_mut()
            }
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }

    #[test]
    fn shared_cursor_bit_packing_matches_gpu_reader_contract() {
        let packed = pack_cursor(0.25, 0.75, true, true, true, true, 42);
        let u = (packed & 0xFFFF) as f32 / 65535.0;
        let v = ((packed >> 16) & 0xFFFF) as f32 / 65535.0;
        assert!((u - 0.25).abs() < 1e-4);
        assert!((v - 0.75).abs() < 1e-4);
        assert_ne!(packed & (1 << 32), 0); // visible
        assert_ne!(packed & (1 << 33), 0); // pressed
        assert_ne!(packed & (1 << 34), 0); // cursor_mode
        assert_ne!(packed & (1 << 35), 0); // initialized
        assert_eq!((packed >> 36) as u32, 42);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::HleResolver;
    use crate::HostPolicy;
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    #[derive(Default)]
    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        mouse: MouseState,
        mouse_look: (f32, f32),
        wheel_buttons: u32,
        cursor_mode: bool,
        rumble: Vec<(usize, u16, u16)>,
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn mouse(&mut self) -> MouseState {
            self.0.lock().unwrap().mouse
        }
        fn pad_extras(&mut self) -> ((f32, f32), u32, bool) {
            let h = self.0.lock().unwrap();
            (h.mouse_look, h.wheel_buttons, h.cursor_mode)
        }
        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            self.0.lock().unwrap().rumble.push((slot, low, high));
        }
    }

    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        fake_with(false)
    }

    fn fake_with(one_pad: bool) -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let hub = Hub::new(Box::new(Fake(Arc::clone(&host))), one_pad);
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
        result(hr, out)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn ordinal_one_resolves_to_create() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)),
            Some(Provenance::Hle)
        );
        assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(0x01)];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        // Mouse VKs (< 0x08) are excluded from the keyboard device reading.
        assert_eq!(count(r), 3);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 3);
        assert_eq!(got[..3], keys[..3]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn all_xbox_buttons_triggers_and_sticks_map_without_conflicts() {
        use std::collections::HashSet;
        let mut seen_vks = HashSet::new();
        for &(vk, _) in &FALLBACK_BUTTONS {
            assert!(seen_vks.insert(vk), "duplicate VK {vk:#x} in FALLBACK_BUTTONS");
            assert!(
                ![b'W', b'A', b'S', b'D', b'I', b'J', b'K', b'L'].contains(&vk),
                "stick key {vk:#x} collides with FALLBACK_BUTTONS"
            );
        }
        // Every single one of the 14 native Xbox buttons must have a working PC binding.
        let button_checks: [(u8, u32); 14] = [
            (0x1B, MENU),
            (b'V', VIEW),
            (0xA0, A),
            (b'R', B),
            (0x20, X),
            (b'F', Y),
            (0x26, UP),
            (0x28, DOWN),
            (0x25, LEFT),
            (0x27, RIGHT),
            (0x09, LEFT_SHOULDER),
            (b'Q', RIGHT_SHOULDER),
            (0xA2, LEFT_STICK),
            (b'C', RIGHT_STICK),
        ];
        for (vk, expected) in button_checks {
            let p = fallback_pad(&[key(vk)], (0.0, 0.0), 0, false);
            assert_eq!(p.buttons, expected, "VK {vk:#x} should map to {expected:#x}");
        }
        // Mouse buttons in Camera Mode: LMB=RT, RMB=LT, MMB=R3, XBtn1=LB, XBtn2=RB.
        let p = fallback_pad(
            &[key(0x01), key(0x02), key(0x04), key(0x05), key(0x06)],
            (0.4, -0.3),
            0,
            false,
        );
        assert_eq!(p.right_trigger, 1.0);
        assert_eq!(p.left_trigger, 1.0);
        assert_eq!(p.buttons, RIGHT_STICK | LEFT_SHOULDER | RIGHT_SHOULDER);
        assert_eq!((p.right_x, p.right_y), (0.4, -0.3));
    }

    #[test]
    fn cursor_mode_suppresses_combat_mouse_triggers_and_look() {
        let p = fallback_pad(
            &[key(0x01), key(0x02), key(0x04), key(0x05), key(0x06), key(b'W')],
            (0.8, 0.5),
            UP,
            true,
        );
        assert_eq!(p.left_trigger, 0.0);
        assert_eq!(p.right_trigger, 0.0);
        assert_eq!(p.right_x, 0.0);
        assert_eq!(p.right_y, 0.0);
        assert_eq!(p.left_y, 1.0);
        assert_eq!(p.buttons, UP);
    }

    #[test]
    fn one_pad_merge_ignores_physical_drift_under_kbm_and_keeps_raw_at_rest() {
        let resting_pad = PadState {
            right_x: 0.04,
            right_y: -0.05,
            ..Default::default()
        };
        // At rest (kbm == 0.0), physical pad passes through raw.
        let idle = merged(PadState::default(), resting_pad);
        assert_eq!((idle.right_x, idle.right_y), (0.04, -0.05));
        // Under micro mouse aim (kbm = -0.02), resting drift (< 0.08) yields to KBM.
        let micro_aim = merged(
            PadState {
                right_x: -0.02,
                right_y: 0.03,
                ..Default::default()
            },
            resting_pad,
        );
        assert_eq!((micro_aim.right_x, micro_aim.right_y), (-0.02, 0.03));
        // When physical stick is actively pushed (>= 0.08), further deflection wins.
        let active_pad = merged(
            PadState {
                right_x: 0.2,
                ..Default::default()
            },
            PadState {
                right_x: -0.9,
                buttons: B,
                ..Default::default()
            },
        );
        assert_eq!((active_pad.right_x, active_pad.buttons), (-0.9, B));
    }

    #[test]
    fn mouse_look_filter_inverts_y_clamps_circle_and_snaps_to_zero() {
        let mut f = MouseLookFilter::default();
        // Win32 dy < 0 (moving mouse up) maps to positive right_y by default.
        let (rx, ry) = f.update(12, -12, 24.0, false);
        assert!(rx > 0.0 && ry > 0.0);
        // Huge diagonal clamps smoothly to the unit circle.
        let (cx, cy) = f.update(1000, -1000, 24.0, false);
        assert!((cx.hypot(cy) - 1.0).abs() < 1e-5);
        // Decay with zero motion eventually snaps to exact (0.0, 0.0).
        let mut out = (cx, cy);
        for _ in 0..32 {
            out = f.update(0, 0, 24.0, false);
        }
        assert_eq!(out, (0.0, 0.0));
    }

    #[test]
    fn wheel_pulser_holds_across_slow_frames_and_separates_rapid_ticks() {
        let mut wp = WheelPulser::default();
        wp.push(240); // two upward notches
        assert_eq!(wp.tick(0), UP);
        assert_eq!(wp.tick(40), UP); // still held inside 48ms window
        assert_eq!(wp.tick(48), 0); // enters 16ms gap
        assert_eq!(wp.tick(60), 0); // still in gap
        assert_eq!(wp.tick(64), UP); // second detent fires cleanly
        wp.clear();
        assert_eq!(wp.tick(70), 0);
    }

    #[test]
    fn mouse_reading_reports_kind_and_cumulative_deltas_at_slot_0x80() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, MOUSE_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(MOUSE, STATUS_CONNECTED, 0)]);

        let first_state = MouseState {
            buttons: MOUSE_LEFT,
            position_x: 120,
            position_y: -45,
            wheel_x: 0,
            wheel_y: 120,
        };
        host.lock().unwrap().mouse = first_state;
        hub.poll();
        let r1 = current(hub, MOUSE_KIND, MOUSE).unwrap();
        let get_mouse =
            slot::<extern "system" fn(*const Reading, *mut MouseState) -> bool, _>(r1, 0x80);
        let mut got = MouseState::default();
        assert!(get_mouse(r1, &mut got));
        assert_eq!(got, first_state);

        let second_state = MouseState {
            buttons: MOUSE_LEFT | MOUSE_RIGHT,
            position_x: 200,
            position_y: -10,
            wheel_x: 0,
            wheel_y: 240,
        };
        host.lock().unwrap().mouse = second_state;
        hub.poll();
        let r2 = current(hub, MOUSE_KIND, MOUSE).unwrap();
        assert!(get_mouse(r2, &mut got));
        assert_eq!(got, second_state);
        assert_eq!(got.position_x - first_state.position_x, 80);
        release(r1);
        release(r2);
    }

    #[test]
    fn a_host_pad_feeds_the_one_pad_which_never_goes() {
        let (hub, host) = fake_with(true);
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);
        let pad = |hub: &Hub| match &hub.state().slots[FALLBACK].history.back().unwrap().value {
            Value::Pad(p) => *p,
            Value::Keys(_) | Value::Mouse(_) => unreachable!(),
        };
        host.lock().unwrap().keys = vec![key(b'D')];
        hub.poll();
        assert_eq!(pad(hub).left_x, 1.0);
        // A pad is switched on, in any slot: its input arrives, beside the
        // keyboard's, and no device comes or goes.
        host.lock().unwrap().pads[2] = Some(PadState { buttons: A, left_y: -0.5, ..Default::default() });
        hub.poll();
        let p = pad(hub);
        assert_eq!((p.buttons, p.left_x, p.left_y), (A, 1.0, -0.5));
        let device: *const Device = &hub.devices[FALLBACK];
        set_rumble(device, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(host.lock().unwrap().rumble, [(2, u16::MAX, 0)]);
        // And off again.
        host.lock().unwrap().pads[2] = None;
        host.lock().unwrap().keys.clear();
        hub.poll();
        assert_eq!(pad(hub), PadState::default());
        assert_eq!(log.lock().unwrap().len(), 1);
        assert!(!hub.state().slots[2].connected);
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        use std::sync::atomic::AtomicBool;
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(&hub.object, std::ptr::null(), GAMEPAD_KIND, 1, 0, context, Some(slow), &mut token),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState::default());
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(!unregister(&hub.object, token, 5000), "times out while it runs");
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        rumble(device, &[0.5, 1.0, 1.0, 1.0]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        assert_eq!(host.lock().unwrap().rumble, [(2, 32767, 65535), (2, 0, 0)]);
    }
}
