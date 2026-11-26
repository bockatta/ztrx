//! Per-program shader translation overrides keyed by the first 16 hex digits of
//! the container's BLAKE3 hash (`hash=` in logs, `#hash` in `victory translate --list`).
//!
//! Specifically addresses Issue #62 (Xeltax `e27b969`): on NVIDIA 32-lane hosts,
//! the wave64 form "without skips, scalars followed" (`ir::without_skips(_, true)`)
//! produces broken tiled-shadow output in 14 `CS_ShadowRenderTiled*` compute programs
//! (corrupting lighting, ceilings, faces, hair, and uniforms). Keeping those 14
//! programs out of `deep-skips` fixes Wave32 shadows while preserving `deep-skips`
//! for every other wave64 compute program.
//!
//! Switches:
//! - `VICTORY_WAVE_DEEP_SKIPS=0`: off for every program;
//! - `VICTORY_WAVE_DEEP_SKIPS_NOT=<hash>,...`: off for the listed programs
//!   (defaults to the 14 known broken `CS_ShadowRenderTiled*` hashes on Wave32;
//!   pass `none` to allow all programs);
//! - `VICTORY_WAVE_DEEP_SKIPS_ONLY=<hash>,...`: off for every program not listed.

use std::cell::RefCell;
use std::sync::OnceLock;

use crate::dxbc::Container;
use crate::native::HwStage;

/// The 14 `CS_ShadowRenderTiled*` wave64 compute containers proven broken in
/// `deep-skips` mode on 32-lane (NVIDIA) hosts (Issue #62 / Xeltax `e27b969`).
#[allow(dead_code)]
pub const DEFAULT_DEEP_SKIPS_NOT: &[&str] = &[
    "922e1d0ee1e19a85",
    "d7db57731ada919b",
    "ab5455fc6abb804f",
    "be6ce5d849d51127",
    "191ecaed66eea3b1",
    "31d703207b51549a",
    "56f3bf2d9388e5a7",
    "804b342cb4b9d1e9",
    "abd7a22106898c66",
    "1d44a723bdebbf27",
    "b4b0427232707598",
    "240cb516c1af0bc3",
    "466ec41d3f9d24c0",
    "65de113e23fa4fdd",
];

thread_local! {
    /// The hash of the container being translated on this thread ([`for_program`]).
    static PROGRAM: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Runs `f` with `hash` recorded as the current container hash on this thread.
pub fn for_program<T>(hash: &str, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PROGRAM.set(self.0.take());
        }
    }
    let _restore = Restore(PROGRAM.replace(Some(hash.to_owned())));
    f()
}

/// The hash [`for_program`] set on this thread, if any.
pub(super) fn current() -> Option<String> {
    PROGRAM.with_borrow(Clone::clone)
}

/// Whether the deep-skips wave64 form may be tried for the program whose hash
/// is `hash`.
pub fn deep_skips_allowed(hash: Option<&str>) -> bool {
    struct Switches {
        on: bool,
        not: Option<Vec<String>>,
        only: Option<Vec<String>>,
    }
    static SWITCHES: OnceLock<Switches> = OnceLock::new();
    let s = SWITCHES.get_or_init(|| {
        let not = std::env::var("VICTORY_WAVE_DEEP_SKIPS_NOT")
            .ok()
            .filter(|v| !v.trim().eq_ignore_ascii_case("none"))
            .map(|v| parse_hash_list(&v));
        let only = std::env::var("VICTORY_WAVE_DEEP_SKIPS_ONLY")
            .ok()
            .map(|v| parse_hash_list(&v));
        Switches {
            on: std::env::var_os("VICTORY_WAVE_DEEP_SKIPS").is_none_or(|v| v != "0"),
            not,
            only,
        }
    });
    decide(s.on, s.not.as_deref(), s.only.as_deref(), hash)
}

/// Whether the wave64 deep-skips decision can affect `container` (a wave64
/// compute program on a 32-lane host).
pub fn wave64_retried(container: &[u8]) -> bool {
    super::lower::host_max_wave() == 32
        && Container::parse(container)
            .ok()
            .and_then(|c| crate::program_part(&c).and_then(|part| crate::native_programs(part).ok()))
            .is_some_and(|ps| ps.iter().any(|p| p.stage == HwStage::Cs && p.wave64))
}

fn parse_hash_list(value: &str) -> Vec<String> {
    value
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .map(|h| h.trim_start_matches('#').to_ascii_lowercase())
        .map(|h| h.chars().take(16).collect::<String>())
        .filter(|h| !h.is_empty() && h != "none")
        .collect()
}

fn listed(list: &[String], hash: Option<&str>) -> bool {
    hash.is_some_and(|hash| {
        let hash = hash.to_ascii_lowercase();
        list.iter().any(|entry| hash.starts_with(entry.as_str()))
    })
}

fn decide(on: bool, not: Option<&[String]>, only: Option<&[String]>, hash: Option<&str>) -> bool {
    on && !not.is_some_and(|l| listed(l, hash)) && only.is_none_or(|l| listed(l, hash))
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "922e1d0ee1e19a85";
    const B: &str = "1111111111111111";

    #[test]
    fn not_excludes_listed_programs_and_keeps_others() {
        let not = parse_hash_list(&format!("#{A}, ab5455fc"));
        assert!(!decide(true, Some(&not), None, Some(A)));
        assert!(!decide(true, Some(&not), None, Some("ab5455fc6abb804f")));
        assert!(decide(true, Some(&not), None, Some(B)));
        assert!(decide(true, Some(&not), None, None));
        assert!(decide(true, Some(&[]), None, Some(A)));
        assert!(!decide(false, Some(&not), None, Some(B)));
    }

    #[test]
    fn only_allows_listed_programs_alone() {
        let only = parse_hash_list(A);
        assert!(decide(true, None, Some(&only), Some(A)));
        assert!(!decide(true, None, Some(&only), Some(B)));
        assert!(!decide(true, None, Some(&only), None));
        assert!(!decide(true, None, Some(&[]), Some(A)));
        assert!(!decide(false, None, Some(&only), Some(A)));
    }

    #[test]
    fn default_excludes_all_14_shadow_tiled_hashes() {
        assert_eq!(DEFAULT_DEEP_SKIPS_NOT.len(), 14);
        let list: Vec<String> = DEFAULT_DEEP_SKIPS_NOT.iter().map(|&h| h.to_owned()).collect();
        for h in DEFAULT_DEEP_SKIPS_NOT {
            assert!(!decide(true, Some(&list), None, Some(h)));
        }
        assert!(decide(true, Some(&list), None, Some(B)));
    }

    #[test]
    fn program_hash_is_scoped_and_restored() {
        assert_eq!(current(), None);
        let inner = for_program(A, || {
            let nested = for_program(B, current);
            (current(), nested)
        });
        assert_eq!(inner, (Some(A.to_owned()), Some(B.to_owned())));
        assert_eq!(current(), None);
        let _ = std::panic::catch_unwind(|| for_program(A, || panic!("boom")));
        assert_eq!(current(), None);
    }
}
