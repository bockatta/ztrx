//! Runtime-configurable graphics module for Series X lighting, shading,
//! ambient occlusion, contact shadows, and ceiling illumination.
//!
//! Resolution precedence (lowest -> highest):
//! 1. Preset defaults (`Balanced` by default)
//! 2. `[graphics]` section in `victory.toml` (`apply_file_overrides`)
//! 3. Environment variables (`VICTORY_GRAPHICS_PRESET`, `VICTORY_GRAPHICS_*`,
//!    and existing `VICTORY_AMBIENT*` flags)
//! 4. Live runtime hotkeys (`F4` cycle preset, `F5` cycle AO/shading taps,
//!    `F6` cycle contact shadow steps)
//!
//! All state is stored in lock-free atomics so `ambient.rs` and `panel.rs`
//! observe changes on the very next frame with zero pipeline recompilation.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// High-level graphics quality presets trading shading/lighting cost for FPS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Preset {
    /// Full dual-scale 20-tap GTAO + 8-step contact shadows + full SH1 gather.
    Cinema = 0,
    /// 8-tap contact AO + 4-step contact shadows + stride-2 SH1 gather (fast & crisp).
    Balanced = 1,
    /// Light 4-tap AO + no contact shadow marching + stride-4 SH1 gather.
    Performance = 2,
    /// Lowest shading cost: skips SH1 gather, AO, contact shadows, air, and reflect passes;
    /// keeps lightweight sky/hemisphere indoor fill so ceilings and NPCs stay lit.
    Minimal = 3,
}

impl Preset {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Cinema,
            1 => Self::Balanced,
            2 => Self::Performance,
            _ => Self::Minimal,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cinema" | "ultra" | "high" => Some(Self::Cinema),
            "balanced" | "medium" | "med" | "normal" => Some(Self::Balanced),
            "performance" | "perf" | "low" => Some(Self::Performance),
            "minimal" | "min" | "fast" | "flat" => Some(Self::Minimal),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cinema => "CINEMA",
            Self::Balanced => "BALANCED",
            Self::Performance => "PERF",
            Self::Minimal => "MINIMAL",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Cinema => Self::Balanced,
            Self::Balanced => Self::Performance,
            Self::Performance => Self::Minimal,
            Self::Minimal => Self::Cinema,
        }
    }

    pub fn settings(self) -> Settings {
        match self {
            Self::Cinema => Settings {
                preset: Self::Cinema,
                ambient_enabled: true,
                ambient_level: 0.90,
                ambient_constant: 0.0,
                sky_enabled: true,
                sky_gain: 0.52,
                sky_least: 0.062,
                sky_power: 0.42,
                ao_strength: 1.00,
                ao_radius: 0.72,
                ao_taps: 20,
                contact_steps: 8,
                gather_step: 1,
                reflect_cap: 0.18,
                reflect_floor: 0.015,
                air_cap: 1.2,
                ambient_most: 2.0,
                ambient_floor: 0.075,
                ceiling_floor: 0.56,
            },
            Self::Balanced => Settings {
                preset: Self::Balanced,
                ambient_enabled: true,
                ambient_level: 0.90,
                ambient_constant: 0.0,
                sky_enabled: true,
                sky_gain: 0.52,
                sky_least: 0.065,
                sky_power: 0.42,
                ao_strength: 0.85,
                ao_radius: 0.65,
                ao_taps: 8,
                contact_steps: 4,
                gather_step: 2,
                reflect_cap: 0.18,
                reflect_floor: 0.015,
                air_cap: 1.2,
                ambient_most: 2.0,
                ambient_floor: 0.080,
                ceiling_floor: 0.58,
            },
            Self::Performance => Settings {
                preset: Self::Performance,
                ambient_enabled: true,
                ambient_level: 0.90,
                ambient_constant: 0.0,
                sky_enabled: true,
                sky_gain: 0.52,
                sky_least: 0.068,
                sky_power: 0.42,
                ao_strength: 0.60,
                ao_radius: 0.50,
                ao_taps: 4,
                contact_steps: 0,
                gather_step: 4,
                reflect_cap: 0.18,
                reflect_floor: 0.015,
                air_cap: 1.2,
                ambient_most: 2.0,
                ambient_floor: 0.080,
                ceiling_floor: 0.62,
            },
            Self::Minimal => Settings {
                preset: Self::Minimal,
                ambient_enabled: true,
                ambient_level: 0.90,
                ambient_constant: 0.0,
                sky_enabled: true,
                sky_gain: 0.52,
                sky_least: 0.072,
                sky_power: 0.42,
                ao_strength: 0.0,
                ao_radius: 0.0,
                ao_taps: 0,
                contact_steps: 0,
                gather_step: 0,
                reflect_cap: 0.18,
                reflect_floor: 0.015,
                air_cap: 0.0,
                ambient_most: 2.0,
                ambient_floor: 0.085,
                ceiling_floor: 0.68,
            },
        }
    }
}

/// Complete snapshot of active graphics parameters consumed by `ambient.rs` and `panel.rs`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub preset: Preset,
    pub ambient_enabled: bool,
    pub ambient_level: f32,
    pub ambient_constant: f32,
    pub sky_enabled: bool,
    pub sky_gain: f32,
    pub sky_least: f32,
    pub sky_power: f32,
    pub ao_strength: f32,
    pub ao_radius: f32,
    /// GTAO sample budget: 20 (8 tight + 12 wide), 8 (tight only), 4 (fast), or 0 (off).
    pub ao_taps: u32,
    /// Directional contact shadow ray-march steps: 8, 4, or 0 (off).
    pub contact_steps: u32,
    /// Stride over the 64x36 radiance grid in `gather` (1=full, 2=half, 4=quarter, 0=skip gather).
    pub gather_step: u32,
    pub reflect_cap: f32,
    pub reflect_floor: f32,
    pub air_cap: f32,
    pub ambient_most: f32,
    pub ambient_floor: f32,
    /// Minimum hemisphere factor for downward-facing ceilings (prevents black ceilings indoors).
    pub ceiling_floor: f32,
}

/// Optional overrides loaded from `victory.toml [graphics]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FileOverrides {
    pub preset: Option<String>,
    pub lighting: Option<bool>,
    pub shading: Option<bool>,
    pub shadows: Option<bool>,
    pub ao_taps: Option<u32>,
    pub contact_steps: Option<u32>,
    pub gather_step: Option<u32>,
    pub ambient_level: Option<f32>,
    pub ceiling_floor: Option<f32>,
    pub air_cap: Option<f32>,
    pub reflect_cap: Option<f32>,
}

struct AtomicSettings {
    preset: AtomicU32,
    ambient_enabled: AtomicBool,
    ambient_level: AtomicU32,
    ambient_constant: AtomicU32,
    sky_enabled: AtomicBool,
    sky_gain: AtomicU32,
    sky_least: AtomicU32,
    sky_power: AtomicU32,
    ao_strength: AtomicU32,
    ao_radius: AtomicU32,
    ao_taps: AtomicU32,
    contact_steps: AtomicU32,
    gather_step: AtomicU32,
    reflect_cap: AtomicU32,
    reflect_floor: AtomicU32,
    air_cap: AtomicU32,
    ambient_most: AtomicU32,
    ambient_floor: AtomicU32,
    ceiling_floor: AtomicU32,
    generation: AtomicU64,
}

impl AtomicSettings {
    fn from_settings(s: Settings) -> Self {
        Self {
            preset: AtomicU32::new(s.preset as u32),
            ambient_enabled: AtomicBool::new(s.ambient_enabled),
            ambient_level: AtomicU32::new(s.ambient_level.to_bits()),
            ambient_constant: AtomicU32::new(s.ambient_constant.to_bits()),
            sky_enabled: AtomicBool::new(s.sky_enabled),
            sky_gain: AtomicU32::new(s.sky_gain.to_bits()),
            sky_least: AtomicU32::new(s.sky_least.to_bits()),
            sky_power: AtomicU32::new(s.sky_power.to_bits()),
            ao_strength: AtomicU32::new(s.ao_strength.to_bits()),
            ao_radius: AtomicU32::new(s.ao_radius.to_bits()),
            ao_taps: AtomicU32::new(s.ao_taps),
            contact_steps: AtomicU32::new(s.contact_steps),
            gather_step: AtomicU32::new(s.gather_step),
            reflect_cap: AtomicU32::new(s.reflect_cap.to_bits()),
            reflect_floor: AtomicU32::new(s.reflect_floor.to_bits()),
            air_cap: AtomicU32::new(s.air_cap.to_bits()),
            ambient_most: AtomicU32::new(s.ambient_most.to_bits()),
            ambient_floor: AtomicU32::new(s.ambient_floor.to_bits()),
            ceiling_floor: AtomicU32::new(s.ceiling_floor.to_bits()),
            generation: AtomicU64::new(1),
        }
    }

    fn store(&self, s: Settings) {
        self.preset.store(s.preset as u32, Ordering::Relaxed);
        self.ambient_enabled.store(s.ambient_enabled, Ordering::Relaxed);
        self.ambient_level.store(s.ambient_level.to_bits(), Ordering::Relaxed);
        self.ambient_constant.store(s.ambient_constant.to_bits(), Ordering::Relaxed);
        self.sky_enabled.store(s.sky_enabled, Ordering::Relaxed);
        self.sky_gain.store(s.sky_gain.to_bits(), Ordering::Relaxed);
        self.sky_least.store(s.sky_least.to_bits(), Ordering::Relaxed);
        self.sky_power.store(s.sky_power.to_bits(), Ordering::Relaxed);
        self.ao_strength.store(s.ao_strength.to_bits(), Ordering::Relaxed);
        self.ao_radius.store(s.ao_radius.to_bits(), Ordering::Relaxed);
        self.ao_taps.store(s.ao_taps, Ordering::Relaxed);
        self.contact_steps.store(s.contact_steps, Ordering::Relaxed);
        self.gather_step.store(s.gather_step, Ordering::Relaxed);
        self.reflect_cap.store(s.reflect_cap.to_bits(), Ordering::Relaxed);
        self.reflect_floor.store(s.reflect_floor.to_bits(), Ordering::Relaxed);
        self.air_cap.store(s.air_cap.to_bits(), Ordering::Relaxed);
        self.ambient_most.store(s.ambient_most.to_bits(), Ordering::Relaxed);
        self.ambient_floor.store(s.ambient_floor.to_bits(), Ordering::Relaxed);
        self.ceiling_floor.store(s.ceiling_floor.to_bits(), Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Release);
    }

    fn load(&self) -> Settings {
        let _ = self.generation.load(Ordering::Acquire);
        Settings {
            preset: Preset::from_u32(self.preset.load(Ordering::Relaxed)),
            ambient_enabled: self.ambient_enabled.load(Ordering::Relaxed),
            ambient_level: f32::from_bits(self.ambient_level.load(Ordering::Relaxed)),
            ambient_constant: f32::from_bits(self.ambient_constant.load(Ordering::Relaxed)),
            sky_enabled: self.sky_enabled.load(Ordering::Relaxed),
            sky_gain: f32::from_bits(self.sky_gain.load(Ordering::Relaxed)),
            sky_least: f32::from_bits(self.sky_least.load(Ordering::Relaxed)),
            sky_power: f32::from_bits(self.sky_power.load(Ordering::Relaxed)),
            ao_strength: f32::from_bits(self.ao_strength.load(Ordering::Relaxed)),
            ao_radius: f32::from_bits(self.ao_radius.load(Ordering::Relaxed)),
            ao_taps: self.ao_taps.load(Ordering::Relaxed),
            contact_steps: self.contact_steps.load(Ordering::Relaxed),
            gather_step: self.gather_step.load(Ordering::Relaxed),
            reflect_cap: f32::from_bits(self.reflect_cap.load(Ordering::Relaxed)),
            reflect_floor: f32::from_bits(self.reflect_floor.load(Ordering::Relaxed)),
            air_cap: f32::from_bits(self.air_cap.load(Ordering::Relaxed)),
            ambient_most: f32::from_bits(self.ambient_most.load(Ordering::Relaxed)),
            ambient_floor: f32::from_bits(self.ambient_floor.load(Ordering::Relaxed)),
            ceiling_floor: f32::from_bits(self.ceiling_floor.load(Ordering::Relaxed)),
        }
    }
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn parse_pos_f32(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0)
}

fn parse_nonneg_f32(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0)
}

/// Resolves initial graphics settings from file overrides + environment variables.
pub fn resolve(file: Option<&FileOverrides>, env: impl Fn(&str) -> Option<String>) -> Settings {
    let env_preset = env("VICTORY_GRAPHICS_PRESET").and_then(|s| Preset::parse(&s));
    let file_preset = file.and_then(|f| f.preset.as_deref()).and_then(Preset::parse);
    let has_explicit_preset = env_preset.is_some() || file_preset.is_some();
    let mut s = file_preset.unwrap_or(Preset::Balanced).settings();

    // Preserve backwards compatibility: if neither VICTORY_AMBIENT nor a graphics
    // preset/lighting toggle was explicitly requested, keep ambient stand-in off.
    let env_ambient = env("VICTORY_AMBIENT");
    if !has_explicit_preset && env_ambient.is_none() && file.and_then(|f| f.lighting).is_none() {
        s.ambient_enabled = false;
    }

    // Layer 2: victory.toml [graphics]
    if let Some(f) = file {
        if let Some(on) = f.lighting {
            s.ambient_enabled = on;
        }
        if let Some(false) = f.shading {
            s.ao_taps = 0;
            s.ao_strength = 0.0;
        }
        if let Some(false) = f.shadows {
            s.contact_steps = 0;
        }
        if let Some(taps) = f.ao_taps {
            s.ao_taps = taps.min(20);
            if s.ao_taps == 0 {
                s.ao_strength = 0.0;
            }
        }
        if let Some(steps) = f.contact_steps {
            s.contact_steps = steps.min(8);
        }
        if let Some(step) = f.gather_step {
            s.gather_step = step.min(8);
        }
        if let Some(level) = f.ambient_level
            && level.is_finite()
            && level > 0.0
        {
            s.ambient_level = level;
            s.ambient_enabled = true;
        }
        if let Some(cf) = f.ceiling_floor
            && cf.is_finite()
            && cf >= 0.0
        {
            s.ceiling_floor = cf.clamp(0.0, 1.0);
        }
        if let Some(air) = f.air_cap
            && air.is_finite()
            && air >= 0.0
        {
            s.air_cap = air;
        }
        if let Some(rc) = f.reflect_cap
            && rc.is_finite()
            && rc >= 0.0
        {
            s.reflect_cap = rc;
        }
    }

    // Layer 3: Environment variables (strictly override victory.toml)
    if let Some(p) = env_preset {
        s = p.settings();
    }
    if let Some(v) = env_ambient {
        let trimmed = v.trim();
        if trimmed.eq_ignore_ascii_case("off") || trimmed == "0" || trimmed.is_empty() {
            s.ambient_enabled = false;
            s.ambient_constant = 0.0;
        } else if let Some(rest) = trimmed.strip_prefix("constant:") {
            if let Some(c) = parse_pos_f32(rest) {
                s.ambient_enabled = true;
                s.ambient_constant = c;
                s.ambient_level = 0.0;
            } else {
                s.ambient_enabled = false;
            }
        } else if let Some(level) = parse_pos_f32(trimmed) {
            s.ambient_enabled = true;
            s.ambient_level = level;
            s.ambient_constant = 0.0;
        } else {
            s.ambient_enabled = false;
        }
    }
    if let Some(v) = env("VICTORY_GRAPHICS_LIGHTING").and_then(|s| parse_bool(&s)) {
        s.ambient_enabled = v;
    }
    if let Some(v) = env("VICTORY_GRAPHICS_SHADING").and_then(|s| parse_bool(&s))
        && !v
    {
        s.ao_taps = 0;
        s.ao_strength = 0.0;
    }
    if let Some(v) = env("VICTORY_GRAPHICS_SHADOWS").and_then(|s| parse_bool(&s))
        && !v
    {
        s.contact_steps = 0;
    }
    if let Some(v) = env("VICTORY_GRAPHICS_AO_TAPS").and_then(|s| s.trim().parse::<u32>().ok()) {
        s.ao_taps = v.min(20);
        if s.ao_taps == 0 {
            s.ao_strength = 0.0;
        }
    }
    if let Some(v) = env("VICTORY_GRAPHICS_CONTACT_STEPS").and_then(|s| s.trim().parse::<u32>().ok()) {
        s.contact_steps = v.min(8);
    }
    if let Some(v) = env("VICTORY_GRAPHICS_GATHER_STEP").and_then(|s| s.trim().parse::<u32>().ok()) {
        s.gather_step = v.min(8);
    }
    if let Some(v) = env("VICTORY_GRAPHICS_CEILING_FLOOR").and_then(|s| parse_nonneg_f32(&s)) {
        s.ceiling_floor = v.clamp(0.0, 1.0);
    }
    if let Some(v) = env("VICTORY_AMBIENT_SHADE") {
        let mut parts = v.split(',');
        if let Some(strength) = parts.next().and_then(parse_nonneg_f32) {
            s.ao_strength = strength;
            if strength == 0.0 {
                s.ao_taps = 0;
            }
        }
        if let Some(radius) = parts.next().and_then(parse_nonneg_f32) {
            s.ao_radius = radius;
        }
    }
    if let Some(v) = env("VICTORY_AMBIENT_SKY") {
        let trimmed = v.trim();
        if trimmed.eq_ignore_ascii_case("off") || trimmed == "0" {
            s.sky_enabled = false;
        } else {
            s.sky_enabled = true;
            let mut sky = [s.sky_gain, s.sky_least, s.sky_power];
            for (to, part) in sky.iter_mut().zip(trimmed.split(',')) {
                if let Some(val) = parse_nonneg_f32(part) {
                    *to = val;
                }
            }
            (s.sky_gain, s.sky_least, s.sky_power) = (sky[0], sky[1], sky[2]);
        }
    }
    if let Some(v) = env("VICTORY_AMBIENT_AIR").and_then(|s| parse_nonneg_f32(&s)) {
        s.air_cap = v;
    }
    if let Some(v) = env("VICTORY_AMBIENT_REFLECT").and_then(|s| parse_nonneg_f32(&s)) {
        s.reflect_cap = v;
    }
    if let Some(v) = env("VICTORY_AMBIENT_REFLECT_FLOOR").and_then(|s| parse_nonneg_f32(&s)) {
        s.reflect_floor = v;
    }
    if let Some(v) = env("VICTORY_AMBIENT_MOST").and_then(|s| parse_nonneg_f32(&s)) {
        s.ambient_most = v;
    }
    if let Some(v) = env("VICTORY_AMBIENT_FLOOR").and_then(|s| parse_nonneg_f32(&s)) {
        s.ambient_floor = v;
    }
    s
}

static STATE: OnceLock<AtomicSettings> = OnceLock::new();

fn state() -> &'static AtomicSettings {
    STATE.get_or_init(|| AtomicSettings::from_settings(resolve(None, |k| std::env::var(k).ok())))
}

/// Applies `[graphics]` overrides from `victory.toml` at startup (before title execution).
/// Environment variables still take precedence over file values.
pub fn apply_file_overrides(overrides: &FileOverrides) {
    let resolved = resolve(Some(overrides), |k| std::env::var(k).ok());
    state().store(resolved);
    tracing::info!(
        preset = resolved.preset.name(),
        ambient = resolved.ambient_enabled,
        ao_taps = resolved.ao_taps,
        contact_steps = resolved.contact_steps,
        gather_step = resolved.gather_step,
        ceiling_floor = resolved.ceiling_floor,
        "graphics module configured"
    );
}

/// Current active graphics settings (lock-free atomic snapshot).
pub fn current() -> Settings {
    state().load()
}

/// Applies a preset immediately at runtime (e.g. from `F4` hotkey).
pub fn set_preset(preset: Preset) -> Settings {
    let s = preset.settings();
    state().store(s);
    tracing::info!(
        preset = s.preset.name(),
        ao_taps = s.ao_taps,
        contact_steps = s.contact_steps,
        gather_step = s.gather_step,
        ceiling_floor = s.ceiling_floor,
        "graphics preset changed"
    );
    s
}

/// `F4`: Cycles `Cinema -> Balanced -> Performance -> Minimal -> Cinema`.
pub fn cycle_preset() -> Settings {
    let cur = current();
    let next = if !cur.ambient_enabled {
        cur.preset
    } else {
        cur.preset.next()
    };
    set_preset(next)
}

/// `F5`: Cycles shading/AO tap quality (`20 -> 8 -> 4 -> 0 -> 20`).
pub fn cycle_shading() -> Settings {
    let mut s = current();
    s.ambient_enabled = true;
    (s.ao_taps, s.ao_strength) = match s.ao_taps {
        t if t >= 20 => (8, 1.15),
        t if t >= 8 => (4, 0.65),
        t if t >= 4 => (0, 0.0),
        _ => (20, 1.35),
    };
    if s.ao_taps > 0 && s.ao_radius <= 0.0 {
        s.ao_radius = 0.72;
    }
    state().store(s);
    tracing::info!(ao_taps = s.ao_taps, ao_strength = s.ao_strength, "shading AO quality cycled (F5)");
    s
}

/// `F6`: Cycles contact shadow steps (`8 -> 4 -> 0 -> 8`).
pub fn cycle_shadows() -> Settings {
    let mut s = current();
    s.ambient_enabled = true;
    s.contact_steps = match s.contact_steps {
        c if c >= 8 => 4,
        c if c >= 4 => 0,
        _ => 8,
    };
    if s.contact_steps > 0 && s.ao_strength <= 0.0 {
        s.ao_strength = 1.0;
        s.ao_radius = 0.72;
    }
    state().store(s);
    tracing::info!(contact_steps = s.contact_steps, "contact shadow steps cycled (F6)");
    s
}

/// Compact HUD summary line (fits inside `panel.rs` 40-column ASCII glyph set).
pub fn hud_line() -> String {
    let s = current();
    if !s.ambient_enabled {
        "GFX OFF  F4 PRESET F5 AO F6 SHAD".to_owned()
    } else {
        format!(
            "GFX {} AO{} CS{} G{}  F4/F5/F6",
            s.preset.name(),
            s.ao_taps,
            s.contact_steps,
            s.gather_step
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_progressively_reduce_shading_and_raise_ceiling_floor() {
        let cinema = Preset::Cinema.settings();
        let balanced = Preset::Balanced.settings();
        let perf = Preset::Performance.settings();
        let minimal = Preset::Minimal.settings();

        assert!(cinema.ao_taps > balanced.ao_taps);
        assert!(balanced.ao_taps > perf.ao_taps);
        assert_eq!(minimal.ao_taps, 0);

        assert!(cinema.contact_steps > balanced.contact_steps);
        assert_eq!(perf.contact_steps, 0);
        assert_eq!(minimal.contact_steps, 0);

        assert_eq!(cinema.gather_step, 1);
        assert_eq!(balanced.gather_step, 2);
        assert_eq!(perf.gather_step, 4);
        assert_eq!(minimal.gather_step, 0);

        assert!(cinema.ceiling_floor >= 0.45 && minimal.ceiling_floor >= cinema.ceiling_floor);
    }

    #[test]
    fn resolution_precedence_env_beats_file_beats_preset() {
        let file = FileOverrides {
            preset: Some("cinema".to_owned()),
            ao_taps: Some(8),
            contact_steps: Some(4),
            ceiling_floor: Some(0.55),
            ..Default::default()
        };
        let s_file = resolve(Some(&file), |_| None);
        assert_eq!(s_file.preset, Preset::Cinema);
        assert!(s_file.ambient_enabled);
        assert_eq!(s_file.ao_taps, 8);
        assert_eq!(s_file.contact_steps, 4);
        assert!((s_file.ceiling_floor - 0.55).abs() < 1e-5);

        let s_env = resolve(Some(&file), |k| match k {
            "VICTORY_GRAPHICS_PRESET" => Some("performance".to_owned()),
            "VICTORY_GRAPHICS_AO_TAPS" => Some("4".to_owned()),
            "VICTORY_GRAPHICS_CEILING_FLOOR" => Some("0.62".to_owned()),
            _ => None,
        });
        assert_eq!(s_env.preset, Preset::Performance);
        assert_eq!(s_env.ao_taps, 4);
        assert_eq!(s_env.contact_steps, 0);
        assert!((s_env.ceiling_floor - 0.62).abs() < 1e-5);
    }

    #[test]
    fn legacy_victory_ambient_env_still_works() {
        let off = resolve(None, |_| None);
        assert!(!off.ambient_enabled);

        let est = resolve(None, |k| (k == "VICTORY_AMBIENT").then(|| "0.9".to_owned()));
        assert!(est.ambient_enabled);
        assert!((est.ambient_level - 0.9).abs() < 1e-5);

        let cst = resolve(None, |k| (k == "VICTORY_AMBIENT").then(|| "constant:0.35".to_owned()));
        assert!(cst.ambient_enabled);
        assert!((cst.ambient_constant - 0.35).abs() < 1e-5);
    }

    #[test]
    fn runtime_cycles_update_atomics_and_hud_line() {
        set_preset(Preset::Cinema);
        assert_eq!(current().preset, Preset::Cinema);
        assert_eq!(cycle_preset().preset, Preset::Balanced);
        assert_eq!(cycle_preset().preset, Preset::Performance);
        assert_eq!(cycle_preset().preset, Preset::Minimal);
        assert_eq!(cycle_preset().preset, Preset::Cinema);

        let s1 = cycle_shading();
        assert_eq!(s1.ao_taps, 8);
        let s2 = cycle_shadows();
        assert_eq!(s2.contact_steps, 4);
        let hud = hud_line();
        assert!(hud.contains("CINEMA") && hud.contains("AO8") && hud.contains("CS4"));
        assert!(hud.len() <= 40, "HUD line too wide: {hud}");
    }
}
