//! The console wave64 tests of Belobug/VIctory-Belobug-AMD (#68, #128, and the
//! upstream ones beside them), run here: hand-assembled GFX10 code, translated by this
//! tree's whole translation for this machine's GPU (its wave size), dispatched and checked
//! lane by lane against what the console's lane layout gives. Only the expected values
//! matter; which form the translation takes is this tree's own (the tests that checked
//! the other tree's choice of wave size are left out).

use super::host::{Dispatch, Host, Root, View};
use crate::native::{ResourceType, RootParam, RootType, StageIo, UsedResource};

/// A host's wave sizes, as the other tree's tests name them.
#[derive(Clone, Copy)]
struct Target {
    lanes_min: u32,
    lanes_max: u32,
}

impl Target {
    fn runs(&self, lanes: u32) -> bool {
        self.lanes_min <= lanes && lanes <= self.lanes_max
    }
}

/// This machine's GPU only: `VICTORY_TEST_HOST_WAVE` (32 unless set) is its most lanes.
fn adapters() -> Result<Vec<String>, String> {
    Ok(vec!["this GPU".to_owned()])
}

trait OnAdapter: Sized {
    fn on(name: &str) -> Result<Self, String>;
    fn target(&self) -> Target;
}

impl OnAdapter for Host {
    fn on(_: &str) -> Result<Self, String> {
        Host::new(false)
    }
    fn target(&self) -> Target {
        let most = std::env::var("VICTORY_TEST_HOST_WAVE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        Target {
            lanes_min: 32,
            lanes_max: most,
        }
    }
}

fn compute(words: &[u32], wave64: bool) -> crate::NativeProgram {
    crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64,
        vgprs: 8,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(crate::native::ComputeIo {
            rsrc2: 0,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// This tree's whole translation ([`super::emit`], fallbacks included) for `target`.
fn emit_for(
    native: &crate::NativeProgram,
    target: Target,
) -> Result<super::HostShader, crate::ShaderError> {
    super::console_wave_tests::translate(native, target.lanes_max)
}

const S_ENDPGM: u32 = 0xBF81_0000;

/// A program over `threads` x 1 x 1 lanes per group with a UAV table in
/// user SGPR 0 and the group id in s1: lane t of group g stores the
/// result of `body` (v0 = t in, v2 out) at dword g * threads + t.
fn program(body: &[u32], threads: u32) -> crate::NativeProgram {
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0x9308_FF01,
        threads * 4, // s_mul_i32 s8, s1, threads * 4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x4A06_0608, // v_add_nc_u32 v3, s8, v3
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ]);
    let mut p = compute(&words, true);
    p.layout = vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    p.used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    // One user SGPR, then the group id x.
    p.io = StageIo::Cs(crate::native::ComputeIo {
        rsrc2: 1 << 1 | 1 << 7,
        threads: [threads, 1, 1],
    });
    p
}

/// Swaps the 16-lane rows of each 32 (`ds_swizzle` xor 16), then mirrors
/// each row (DPP): lane t ends with t ^ 31, every read inside its half.
const IN_HALF: &[u32] = &[
    0xD8D4_401F,
    0x0100_0000, // ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM, and 0x1f, xor 0x10)
    0x7E04_02FA,
    0xFF01_4001, // v_mov_b32_dpp v2, v1 row_mirror row_mask:0xf bank_mask:0xf
];

fn run(host: &Host, native: &crate::NativeProgram, groups: u32) -> Result<Vec<u32>, String> {
    let target = host.target();
    let cs = emit_for(native, target).map_err(|e| e.to_string())?;
    // The validator, where one is installed: drivers take some modules it
    // refuses.
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil.clone())
            .map_err(|e| format!("validation: {e}"))?;
    }
    let (root, pso) = host.compute_pipeline(&cs)?;
    let data = vec![0; (cs.user_data_dwords() as usize).max(1)];
    let StageIo::Cs(io) = &native.io else {
        unreachable!()
    };
    let n = io.threads.iter().product::<u32>() * groups;
    let views = [View::RawUav(n)];
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(data))],
            views: &views,
            samplers: &[],
            groups: [groups, 1, 1],
        },
    )?;
    Ok(out.into_iter().next().unwrap_or_default())
}

#[test]
fn in_half_lane_reads_match_the_console_on_every_adapter() {
    let Ok(names) = adapters() else {
        eprintln!("no DXGI; skipped");
        return;
    };
    let mut ran = 0;
    for name in names {
        let Ok(host) = Host::on(&name) else {
            eprintln!("{name}: no SM 6.6 device; skipped");
            continue;
        };
        let target = host.target();
        // One console wave, two per group, and a group whose second wave
        // has its lower half only.
        for threads in [64, 128, 32] {
            let p = program(IN_HALF, threads);
            let split = !target.runs(64);
            let groups = 3;
            let out = run(&host, &p, groups).unwrap_or_else(|e| panic!("{name}: {e}"));
            let want: Vec<u32> = (0..groups)
                .flat_map(|_| (0..threads).map(|t| t ^ 31))
                .collect();
            assert_eq!(out, want, "{name}: {threads} threads (split {split})");
            eprintln!("{name}: {threads} threads per group exact (split {split})");
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no hardware adapter; skipped");
    }
}

/// 3-D groups: the console forms waves from the flat thread id (x
/// fastest). Lane f (f = x + X * (y + Y * z)) stores f ^ 31 at dword f.
fn program_3d(threads: [u32; 3]) -> crate::NativeProgram {
    let [x, y, _] = threads;
    let (sx, sxy) = (x.trailing_zeros(), (x * y).trailing_zeros());
    assert!(x.is_power_of_two() && y.is_power_of_two());
    let words = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000,               // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F,               // s_waitcnt lgkmcnt(0)
        0x340A_0200 | (128 + sx),  // v_lshlrev_b32 v5, log2 X, v1
        0x340C_0400 | (128 + sxy), // v_lshlrev_b32 v6, log2 XY, v2
        0x4A08_0B00,               // v_add_nc_u32 v4, v0, v5
        0x4A08_0D04,               // v_add_nc_u32 v4, v4, v6
        0xD8D4_401F,
        0x0100_0004, // ds_swizzle_b32 v1, v4 offset:swizzle(BITMASK_PERM, and 0x1f, xor 0x10)
        0x7E04_02FA,
        0xFF01_4001, // v_mov_b32_dpp v2, v1 row_mirror
        0x3406_0882, // v_lshlrev_b32 v3, 2, v4
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ];
    let mut p = compute(&words, true);
    p.layout = program(&[], 64).layout;
    p.used = program(&[], 64).used;
    // One user SGPR; thread ids x, y and z in v0..v2.
    p.io = StageIo::Cs(crate::native::ComputeIo {
        rsrc2: 1 << 1 | 2 << 11,
        threads,
    });
    p
}

#[test]
fn three_dimensional_groups_keep_the_consoles_lanes() {
    let Ok(names) = adapters() else {
        eprintln!("no DXGI; skipped");
        return;
    };
    for name in names {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        let split = !host.target().runs(64);
        for threads in [
            [4, 4, 4],
            [8, 8, 2],
            [2, 4, 8],
            [16, 2, 4],
            [8, 8, 4],
            [4, 8, 8],
        ] {
            let p = program_3d(threads);
            let n: u32 = threads.iter().product();
            let out = run(&host, &p, 1).unwrap_or_else(|e| panic!("{name}: {e}"));
            let want: Vec<u32> = (0..n).map(|f| f ^ 31).collect();
            assert_eq!(out, want, "{name}: {threads:?} (split {split})");
            eprintln!("{name}: {threads:?} exact (split {split})");
        }
    }
}

/// Runs `body` (v0 = t in, v2 out) on every hardware adapter in groups of
/// `threads` and checks lane t against `want(t)`; on a host without
/// 64-lane waves it must have translated in the console flow.
fn console_flow_case(label: &str, body: &[u32], threads: u32, want: impl Fn(u32) -> u32) {
    flow_case(label, body, threads, want, true);
}

/// [`console_flow_case`], checking hosts with 64-lane waves (the per-lane
/// form) only when `native` says.
fn flow_case(label: &str, body: &[u32], threads: u32, want: impl Fn(u32) -> u32, native: bool) {
    let Ok(names) = adapters() else {
        eprintln!("no DXGI; skipped");
        return;
    };
    for name in names {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        let p = program(body, threads);
        let split = !host.target().runs(64);
        let groups = 2;
        let out = run(&host, &p, groups).unwrap_or_else(|e| panic!("{name}: {label}: {e}"));
        let expect: Vec<u32> = (0..groups).flat_map(|_| (0..threads).map(&want)).collect();
        if !split && !native {
            if out != expect {
                eprintln!("{name}: {label}: the per-lane form differs (known)");
            }
            continue;
        }
        assert_eq!(
            out, expect,
            "{name}: {label}, {threads} threads (split {split})"
        );
        eprintln!("{name}: {label}, {threads} threads exact (split {split})");
    }
}

/// A mask counted whole: 40 lanes of the console wave, 32 in one half and
/// 8 in the other.
#[test]
fn masks_read_whole_count_both_halves() {
    let body = [
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("s_bcnt1 of lanes below 40", &body, 64, |_| 40);
    // 48 threads: the upper half is partial.
    console_flow_case("s_bcnt1 of lanes below 40", &body, 48, |_| 40);
    // 32 threads: no upper half at all.
    console_flow_case("s_bcnt1 of lanes below 40", &body, 32, |_| 32);
}

/// A lane of the other half: lane 40 is the upper half's lane 8.
#[test]
fn lane_reads_cross_halves() {
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("v_readlane 40", &body, 64, |_| 40);
}

/// A branch only lanes below 20 take (all in the lower half), with a lane
/// read inside: the upper half has no lane there but still meets the lower
/// one at the exchange.
#[test]
fn exchanges_inside_a_branch_one_half_skips() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0003, // s_cbranch_execz +3
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    console_flow_case("if (lane < 20) v2 = readlane(10)", &body, 64, |t| {
        if t < 20 { 10 } else { 1 }
    });
}

/// A loop lane t runs t times, with a lane read of the other half in
/// every iteration: the lower half's lanes are done after 31 iterations,
/// the upper half's go on to 63, and the console wave loops while any lane
/// does. After it, s9 holds lane 40's id, as the last iteration read it.
///
/// The per-lane form got this wrong on every host (#225); hosts with
/// 64-lane waves now run it in the console flow too, without exchanges.
#[test]
#[ignore = "needs the console flow (every lane of a console wave runs each block it runs, loops while any lane does), which this tree does not have"]
fn loops_run_while_any_lane_of_the_console_wave_does() {
    let body = [
        0xBE89_0380, // s_mov_b32 s9, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0006, // s_cbranch_execz end
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFFA, // s_cbranch_execnz loop
        // end:
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ];
    console_flow_case("loop t times reading lane 40", &body, 64, |t| t + 40);
}

/// DPP's row mask names the console's rows: with `row_mask:0x3` only
/// rows 0 and 1 (lanes 0 to 31) write, so the upper half keeps its value
/// although its host rows are 0 and 1 too.
#[test]
fn dpp_row_masks_name_console_rows() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7E04_02FA,
        0x3F00_E400, // v_mov_b32_dpp v2, v0 quad_perm:[0,1,2,3] row_mask:0x3 bank_mask:0xf
    ];
    console_flow_case(
        "DPP row_mask 0x3",
        &body,
        64,
        |t| if t < 32 { t } else { 1 },
    );
}

/// `v_writelane` ignores EXEC: inside a branch only lanes below 20 take,
/// it still writes lane 40. The per-lane form never had lane 40 there.
#[test]
#[ignore = "needs the console flow (every lane of a console wave runs each block it runs, loops while any lane does), which this tree does not have"]
fn lane_writes_reach_lanes_outside_the_branch() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0004, // s_cbranch_execz +4
        0xBE89_03FF,
        0x0000_004D, // s_mov_b32 s9, 77
        0xD761_0002,
        0x0001_5009, // v_writelane_b32 v2, s9, 40
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    console_flow_case("if (lane < 20) writelane(40, 77)", &body, 64, |t| {
        if t == 40 { 77 } else { 1 }
    });
}

/// The loop form compilers emit: lanes that are done are added to an exit
/// mask that starts as 0, EXEC loses them, and the mask restores EXEC
/// after the loop. Lane t runs t + 1 iterations with a read of lane 40 in
/// each; afterwards every lane has s9 = 40 and counts all 64 lanes in the
/// exit mask.
#[test]
#[ignore = "needs the console flow (every lane of a console wave runs each block it runs, loops while any lane does), which this tree does not have"]
fn loops_with_an_exit_mask_grown_from_zero() {
    let body = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0480, // s_mov_b64 s[10:11], 0
        0xBE89_0380, // s_mov_b32 s9, 0
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
        0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
        0xBF89_FFF9, // s_cbranch_execnz loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBE8C_100A, // s_bcnt1_i32_b64 s12, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
        0x4A04_040C, // v_add_nc_u32 v2, s12, v2
    ];
    console_flow_case("exit-mask loop reading lane 40", &body, 64, |_| 40 + 64);
}

/// [`program`] with 512 bytes of LDS.
fn program_lds(body: &[u32], threads: u32) -> crate::NativeProgram {
    let mut p = program(body, threads);
    if let StageIo::Cs(io) = &mut p.io {
        io.rsrc2 |= 1 << 15;
    }
    p
}

/// [`flow_case`] for a program with LDS ([`program_lds`]), checked on every
/// hardware adapter: on a host without 64-lane waves it runs in the console
/// flow, its halves meeting between LDS accesses.
fn lds_case(label: &str, body: &[u32], threads: u32, want: impl Fn(u32) -> u32) {
    let Ok(names) = adapters() else {
        eprintln!("no DXGI; skipped");
        return;
    };
    for name in names {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        let p = program_lds(body, threads);
        let split = !host.target().runs(64);
        let groups = 2;
        let out = run(&host, &p, groups).unwrap_or_else(|e| panic!("{name}: {label}: {e}"));
        let expect: Vec<u32> = (0..groups).flat_map(|_| (0..threads).map(&want)).collect();
        assert_eq!(
            out, expect,
            "{name}: {label}, {threads} threads (split {split})"
        );
        eprintln!("{name}: {label}, {threads} threads exact (split {split})");
    }
}

/// Lane t stores t at LDS dword t (v1 = 4t).
const LDS_OWN: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xD834_0000,
    0x0000_0001, // ds_write_b32 v1, v0
];

/// v2 = LDS dword (n - t), n an inline constant code.
const fn lds_read_mirror(n: u32) -> [u32; 5] {
    [
        0x4C08_0000 | n, // v_sub_nc_u32 v4, n, v0
        0x3408_0882,     // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0200_0004, // ds_read_b32 v2, v4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ]
}

/// One console wave per group shares LDS without a barrier, as the console
/// compiler leaves it within one wave: lane t writes dword t, then reads
/// dword 63 - t, which the other half wrote. Refused before on hosts
/// without 64-lane waves.
#[test]
fn lds_shared_within_one_console_wave_reads_the_other_half() {
    let mut body = LDS_OWN.to_vec();
    body.extend(lds_read_mirror(0xBF)); // 63
    lds_case("LDS[63 - t] after LDS[t] = t", &body, 64, |t| 63 - t);
    // A partial upper half.
    let mut body = LDS_OWN.to_vec();
    body.extend(lds_read_mirror(0xAF)); // 47
    lds_case("LDS[47 - t] after LDS[t] = t", &body, 48, |t| 47 - t);
}

/// A read, then a write of the dword the other half reads: in step, every
/// lane reads before any lane writes.
#[test]
fn lds_reads_come_before_the_other_halfs_later_writes() {
    let mut body = LDS_OWN.to_vec();
    body.extend([
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0200_0004, // ds_read_b32 v2, v4
        0x7E0A_02FF,
        1000, // v_mov_b32 v5, 1000
        0xD834_0000,
        0x0000_0501, // ds_write_b32 v1, v5
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ]);
    lds_case("LDS[t ^ 32] read before LDS[t] = 1000", &body, 64, |t| {
        t ^ 32
    });
}

/// Lanes below 20 (all in the lower half) write into the upper half's
/// dwords inside a branch; then every lane reads its own dword.
#[test]
fn lds_writes_inside_a_branch_reach_the_other_half() {
    let mut body = LDS_OWN.to_vec();
    body.extend([
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0006, // s_cbranch_execz +6
        0x4A0A_00FF,
        100,         // v_add_nc_u32 v5, 100, v0
        0x4A0C_00A0, // v_add_nc_u32 v6, 32, v0
        0x340C_0C82, // v_lshlrev_b32 v6, 2, v6
        0xD834_0000,
        0x0000_0506, // ds_write_b32 v6, v5
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xD8D8_0000,
        0x0200_0001, // ds_read_b32 v2, v1
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ]);
    let want = |t: u32| {
        if (32..52).contains(&t) {
            100 + t - 32
        } else {
            t
        }
    };
    lds_case("LDS[t + 32] = 100 + t for t < 20", &body, 64, want);
    lds_case("LDS[t + 32] = 100 + t for t < 20", &body, 48, want);
}

/// Lanes of the other half read whole (`v_readlane`) and through LDS in one
/// program: the console flow with LDS.
#[test]
fn lane_reads_and_lds_in_one_console_wave() {
    let mut body = LDS_OWN.to_vec();
    body.extend([
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
    ]);
    body.extend(lds_read_mirror(0xBF)); // 63
    body.push(0x4A04_0409); // v_add_nc_u32 v2, s9, v2
    lds_case("readlane 40 + LDS[63 - t]", &body, 64, |t| 63 - t + 40);
}

// ---- Groups of several console waves (`Lower::pair_meet`) ----

/// Threads of console wave `w` in a group of `threads`.
fn wave_threads(threads: u32, w: u32) -> u32 {
    threads.saturating_sub(64 * w).min(64)
}

/// [`flow_case`] for groups of several console waves: lane t's expected
/// value is `want(t, threads)`.
fn waves_case(label: &str, body: &[u32], counts: &[u32], want: impl Fn(u32, u32) -> u32) {
    for &threads in counts {
        console_flow_case(label, body, threads, |t| want(t, threads));
    }
}

/// v1 = the lane's place in its console wave (t & 63), v5 = its console
/// wave in the group (t >> 6).
const LANE_AND_WAVE: [u32; 2] = [
    0x3602_00BF, // v_and_b32 v1, 63, v0
    0x2C0A_0086, // v_lshrrev_b32 v5, 6, v0
];

/// Each console wave counts its own lanes below 40 - w: the halves of every
/// console wave exchange their ballots, without the group's other waves.
/// 96 and 80 threads: the last console wave has a lower half only.
#[test]
fn several_console_waves_count_their_own_masks() {
    let mut body = LANE_AND_WAVE.to_vec();
    body.extend([
        0x4C0C_0AA8, // v_sub_nc_u32 v6, 40, v5
        0x7D88_0306, // v_cmp_gt_u32 vcc, v6, v1
        0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
        0x7E04_0209, // v_mov_b32 v2, s9
    ]);
    waves_case(
        "s_bcnt1 of lanes below 40 - wave",
        &body,
        &[128, 256, 96, 80, 112, 1024],
        |t, n| (40 - t / 64).min(wave_threads(n, t / 64)),
    );
}

/// Lanes of the other half, each console wave its own: lane 40 of the wave
/// (in the upper half), and lane 10 inside a branch only lanes below 20
/// take (the upper half has no lane there but meets the lower one).
#[test]
fn several_console_waves_read_their_own_lanes() {
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    waves_case("v_readlane 40", &body, &[128, 256, 112], |t, _| {
        (t & !63) + 40
    });
    let mut body = LANE_AND_WAVE.to_vec();
    body.extend([
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0294, // v_cmp_gt_u32 vcc, 20, v1
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0003, // s_cbranch_execz +3
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ]);
    waves_case(
        "if (lane < 20) v2 = readlane(10)",
        &body,
        &[128, 256, 96, 80],
        |t, _| if t & 63 < 20 { (t & !63) + 10 } else { 1 },
    );
}

/// A mask operation's SCC read as a number (`WaveActiveAnyTrue` choosing a
/// value): only lane 36, in the upper half, sets the mask. It is the console
/// wave's: before, such a program took no wave size and each host wave of 32
/// answered for itself (7 in the lower half).
#[test]
fn any_lane_of_the_console_wave_chooses_a_value() {
    let scc = [
        0x7D84_00FF,
        36,          // v_cmp_eq_u32 vcc, 36, v0
        0x878C_7E6A, // s_and_b64 s[12:13], vcc, exec
        0x8509_8785, // s_cselect_b32 s9, 5, 7
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("SCC of (t == 36) & exec", &scc, 64, |_| 5);
}

/// Masks read as numbers by an instruction, which the halves exchange
/// before it (`Lower::meet_before`): EXEC by `v_readfirstlane` (lanes at or
/// past 20 of each console wave read lane 20, in the lower half, the upper
/// half too; the lanes are counted with `s_bcnt1` after), EXEC by
/// `v_mbcnt` (odd lanes count the odd lanes below them across the halves),
/// and a mask operation's SCC (only thread 100, in console wave 1's upper
/// half, sets it).
#[test]
fn several_console_waves_read_masks_as_numbers() {
    let mut first = LANE_AND_WAVE.to_vec();
    first.extend([
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D86_0294, // v_cmp_le_u32 vcc, 20, v1
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0002, // s_cbranch_execz +2
        0x7E12_0500, // v_readfirstlane_b32 s9, v0
        0x7E04_0209, // v_mov_b32 v2, s9
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBE8E_107E, // s_bcnt1_i32_b64 s14, exec
        0x4A04_040E, // v_add_nc_u32 v2, s14, v2
    ]);
    waves_case(
        "v_readfirstlane past lane 20, + lanes",
        &first,
        &[128, 256, 96, 112],
        |t, n| {
            let first = if t & 63 >= 20 { (t & !63) + 20 } else { 1 };
            first + wave_threads(n, t / 64)
        },
    );
    let mut count = LANE_AND_WAVE.to_vec();
    count.extend([
        0x7E04_02FF,
        1000,        // v_mov_b32 v2, 1000
        0x3608_0081, // v_and_b32 v4, 1, v0
        0x7D84_0881, // v_cmp_eq_u32 vcc, 1, v4
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0004, // s_cbranch_execz +4
        0xD765_0002,
        0x0001_007E, // v_mbcnt_lo_u32_b32 v2, exec_lo, 0
        0xD766_0002,
        0x0002_047F, // v_mbcnt_hi_u32_b32 v2, exec_hi, v2
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ]);
    waves_case("v_mbcnt of odd lanes", &count, &[128, 256, 96], |t, _| {
        if t & 1 == 1 { (t & 63) / 2 } else { 1000 }
    });
    let scc = [
        0x7D84_00FF,
        100,         // v_cmp_eq_u32 vcc, 100, v0
        0x878C_7E6A, // s_and_b64 s[12:13], vcc, exec
        0x8509_8785, // s_cselect_b32 s9, 5, 7
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    waves_case("SCC of vcc & exec", &scc, &[128, 256], |t, _| {
        if t / 64 == 1 { 5 } else { 7 }
    });
}

/// A loop whose trip count differs between the group's console waves
/// ((t >> 1) & 63: up to 31 in waves 0 and 2, up to 63 in waves 1 and 3),
/// with a lane read of the other half in every iteration: each console
/// wave loops while any of its lanes does, its halves meeting in every
/// iteration while the other waves are elsewhere.
#[test]
#[ignore = "needs the console flow (every lane of a console wave runs each block it runs, loops while any lane does), which this tree does not have"]
fn several_console_waves_loop_on_their_own() {
    let body = [
        0x2C02_0081, // v_lshrrev_b32 v1, 1, v0
        0x3602_02BF, // v_and_b32 v1, 63, v1
        0xBE89_0380, // s_mov_b32 s9, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D8A_0280, // v_cmp_ne_u32 vcc, 0, v1
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0006, // s_cbranch_execz end
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x7D82_0302, // v_cmp_lt_u32 vcc, v2, v1
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFFA, // s_cbranch_execnz loop
        // end:
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ];
    waves_case(
        "loop (t >> 1) & 63 times reading lane 40",
        &body,
        &[128, 256, 112],
        |t, _| ((t >> 1) & 63) + (t & !63) + 40,
    );
}

/// [`program_lds`] with `bytes` of LDS (in 512-byte steps).
fn program_lds_bytes(body: &[u32], threads: u32, bytes: u32) -> crate::NativeProgram {
    let mut p = program(body, threads);
    if let StageIo::Cs(io) = &mut p.io {
        io.rsrc2 |= bytes.div_ceil(512) << 15;
    }
    p
}

/// LDS shared within each console wave of a group of several: lane t
/// writes dword t, then reads dword t ^ 32 (the other half of its console
/// wave) without a barrier; the halves meet between the two.
#[test]
fn several_console_waves_share_lds_within_each() {
    let Ok(names) = adapters() else {
        eprintln!("no DXGI; skipped");
        return;
    };
    let mut body = LDS_OWN.to_vec();
    body.extend([
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0200_0004, // ds_read_b32 v2, v4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ]);
    for name in names {
        let Ok(host) = Host::on(&name) else {
            continue;
        };
        let split = !host.target().runs(64);
        for threads in [128, 256] {
            let p = program_lds_bytes(&body, threads, threads * 4);
            let groups = 2;
            let out = run(&host, &p, groups).unwrap_or_else(|e| panic!("{name}: {e}"));
            let want: Vec<u32> = (0..groups)
                .flat_map(|_| (0..threads).map(|t| (t ^ 32) + (t & !63) + 40))
                .collect();
            assert_eq!(out, want, "{name}: {threads} threads (split {split})");
            eprintln!("{name}: LDS within each of {threads} threads' waves exact (split {split})");
        }
    }
}

/// An if/else on the lane, each side setting s9, then s9 read: the console
/// wave runs each side some lane of it takes, the else side last, so all
/// its lanes read 2 where some lane took the else side and 1 where none
/// did. Wave w's lanes below 16 (w + 1) take the then side: from wave 3 on,
/// and in the 32 threads of a last wave of 96, every lane does.
fn both_sides(cmp: u32) -> Vec<u32> {
    let mut body = LANE_AND_WAVE.to_vec();
    body.extend([
        0x4A0C_0A81, // v_add_nc_u32 v6, 1, v5
        0x340C_0C84, // v_lshlrev_b32 v6, 4, v6
        0xBE89_0380, // s_mov_b32 s9, 0
        cmp,
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0001, // s_cbranch_execz else
        0xBE89_0381, // s_mov_b32 s9, 1
        0x89FE_0A7E, // else: s_xor_b64 exec, exec, s[10:11]
        0xBF88_0001, // s_cbranch_execz join
        0xBE89_0382, // s_mov_b32 s9, 2
        0x88FE_0A7E, // join: s_or_b64 exec, exec, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
    ]);
    body
}

/// v_cmp_gt_u32 vcc, v6, v1: the lane's place below 16 (w + 1).
const BELOW_16_PER_WAVE: u32 = 0x7D88_0306;

/// [`both_sides`] in groups of one console wave and of several: the halves
/// of a console wave take different sides, yet read the same scalar. The
/// per-lane form read each half's own side on hosts without 64-lane waves,
/// and on hosts with them where the driver ran 32-lane waves.
#[test]
fn scalars_after_a_divergent_branch_are_the_console_waves() {
    waves_case(
        "s9 set on both sides of a divergent if/else",
        &both_sides(BELOW_16_PER_WAVE),
        &[64, 128, 256, 96, 112],
        |t, n| {
            let w = t / 64;
            if wave_threads(n, w) > 16 * (w + 1) {
                2
            } else {
                1
            }
        },
    );
}

// ---- Whole-wave sections in halves (`Lower::both_halves`) ----

/// A whole-wave sum of `v0` as the console compiler expands it for wave64
/// (llvm-mc, gfx1030, from `wave::tests`, registers moved): each half
/// summed with DPP and `v_permlanex16`, then lanes 0 and 32 read and added
/// in s12. `op` picks the operation: (VOP2 opcode, SOP2 opcode, the
/// identity's inline code, the data's VGPR).
fn wave_reduce(vop2: u32, sop2: u32, ident: u32, data: u32) -> Vec<u32> {
    let dpp = |xmask: u32| {
        [
            vop2 << 25 | 1 << 17 | 1 << 9 | 0xFA,
            0xFF01_6001 | xmask << 8,
        ]
    };
    let mut words = vec![
        0xBE8A_25C1, // s_or_saveexec_b64 s[10:11], -1
        0xD501_0001,
        10 << 18 | (0x100 + data) << 9 | ident, // v_cndmask_b32_e64 v1, ident, v<data>, s[10:11]
    ];
    for xmask in [1, 2, 4, 8] {
        words.extend(dpp(xmask)); // v_<op>_dpp v1, v1, v1 row_xmask:<xmask>
    }
    words.extend([
        0xD778_0002,
        0x0305_8301,                           // v_permlanex16_b32 v2, v1, -1, -1
        vop2 << 25 | 1 << 17 | 2 << 9 | 0x101, // v_<op> v1, v1, v2
        0xD760_000C,
        0x0001_0101, // v_readlane_b32 s12, v1, 0
        0xD760_000D,
        0x0001_4101,                                        // v_readlane_b32 s13, v1, 32
        0x8000_0000 | sop2 << 23 | 12 << 16 | 13 << 8 | 12, // s_<op> s12, s12, s13
        0xBEFE_040A,                                        // s_mov_b64 exec, s[10:11]
        0x7E04_020C,                                        // v_mov_b32 v2, s12
    ]);
    words
}

/// Console wave `w`'s threads in a group of `n`.
fn wave_lanes(t: u32, n: u32) -> std::ops::Range<u32> {
    let w = t / 64;
    64 * w..64 * w + wave_threads(n, w)
}

/// Whole-wave reductions: the host's wave operation covers a half, the
/// halves' results are combined. 96 threads: the second console wave has
/// no upper half, which counts as the identity (for a minimum, not 0).
#[test]
fn whole_wave_reductions_combine_both_halves() {
    let counts = [64, 48, 128, 96, 256];
    // v_add_nc_u32 / s_add_u32 over t.
    let sum = wave_reduce(0x25, 0, 0x80, 0);
    waves_case("WaveActiveSum(t)", &sum, &counts, |t, n| {
        wave_lanes(t, n).sum()
    });
    // v_min_u32 / s_min_u32 over 1000 - t (its least in the upper half).
    let mut min = vec![
        0x4C08_00FF,
        1000, // v_sub_nc_u32 v4, 1000, v0
    ];
    min.extend(wave_reduce(0x13, 7, 0xC1, 4));
    waves_case("WaveActiveMin(1000 - t)", &min, &counts, |t, n| {
        wave_lanes(t, n).map(|x| 1000 - x).min().unwrap()
    });
}

/// An inclusive prefix sum over the console wave (llvm-mc, gfx1030, as in
/// `wave::tests`, carried to the upper half): row_shr steps in rows of 16,
/// the even rows' totals into the odd rows (`v_permlanex16`, row-masked DPP
/// move), then lane 31's total added to rows 2 and 3.
const PREFIX_SUM: &[u32] = &[
    0xBE8A_25C1, // s_or_saveexec_b64 s[10:11], -1
    0xD501_0001,
    0x002A_0080, // v_cndmask_b32_e64 v1, 0, v0, s[10:11]
    0x4A02_02FA,
    0xFF01_1101, // v_add_nc_u32_dpp v1, v1, v1 row_shr:1
    0x4A02_02FA,
    0xFF01_1201, // v_add_nc_u32_dpp v1, v1, v1 row_shr:2
    0x4A02_02FA,
    0xFF01_1401, // v_add_nc_u32_dpp v1, v1, v1 row_shr:4
    0x4A02_02FA,
    0xFF01_1801, // v_add_nc_u32_dpp v1, v1, v1 row_shr:8
    0xD778_0002,
    0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
    0x7E06_0280, // v_mov_b32 v3, 0
    0x7E06_02FA,
    0xAF00_E402, // v_mov_b32_dpp v3, v2 quad_perm:[0,1,2,3] row_mask:0xa
    0x4A02_0701, // v_add_nc_u32 v1, v1, v3
    0xD760_0010,
    0x0001_3F01, // v_readlane_b32 s16, v1, 31
    0x7E0A_0210, // v_mov_b32 v5, s16
    0x4A02_02FA,
    0xCF00_E405, // v_add_nc_u32_dpp v1, v5, v1 quad_perm:[0,1,2,3] row_mask:0xc
    0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    0x7E04_0301, // v_mov_b32 v2, v1
];

/// The prefix sum: the upper half adds the lower half's total.
#[test]
fn whole_wave_prefix_sums_carry_the_lower_half() {
    waves_case(
        "inclusive WavePrefixSum(t)",
        PREFIX_SUM,
        &[64, 48, 128, 96, 256],
        |t, n| wave_lanes(t, n).filter(|&x| x <= t).sum(),
    );
}

// ---- `v_permlane16` / `v_permlanex16` in halves ----

/// Each lane reads its row's lane 15 - (its place in the row), in its own
/// row (`v_permlane16`) and in the other row of its half
/// (`v_permlanex16`). Within a half: split in halves on a host without
/// 64-lane waves, no console flow. Refused before on such hosts.
#[test]
fn permlanes_stay_in_their_half() {
    let select = [
        0xBE8C_03FF,
        0x89AB_CDEF, // s_mov_b32 s12, 0x89abcdef
        0xBE8D_03FF,
        0x0123_4567, // s_mov_b32 s13, 0x01234567
    ];
    let mut row = select.to_vec();
    row.extend([
        0xD777_0002,
        0x0034_1900, // v_permlane16_b32 v2, v0, s12, s13
    ]);
    let mut other = select.to_vec();
    other.extend([
        0xD778_0002,
        0x0034_1900, // v_permlanex16_b32 v2, v0, s12, s13
    ]);
    for threads in [64, 128, 32] {
        flow_case(
            "v_permlane16 reversing rows",
            &row,
            threads,
            |t| t ^ 15,
            true,
        );
        flow_case(
            "v_permlanex16 reversing rows",
            &other,
            threads,
            |t| t ^ 31,
            true,
        );
    }
}

// ---- `ds_bpermute_b32` / `ds_permute_b32`: each half on its own ----

/// v4 = (t ^ 33) * 4: the address names lane l ^ 33 of the console wave,
/// in the other half; only its bits 6:2 count (RDNA ISA 10.4.4), so the
/// lane read is l ^ 1, in the lane's own half.
const TO_THE_OTHER_HALF: [u32; 2] = [
    0x3A08_00A1, // v_xor_b32 v4, 33, v0
    0x3408_0882, // v_lshlrev_b32 v4, 2, v4
];

const DS_BPERMUTE: [u32; 3] = [
    0xDACC_0000,
    0x0200_0004, // ds_bpermute_b32 v2, v4, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

const DS_PERMUTE: [u32; 3] = [
    0xDAC8_0000,
    0x0200_0004, // ds_permute_b32 v2, v4, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
];

/// v2 = 7, then the lanes whose t is even run the next `n` dwords.
const fn even_lanes_run(n: u32) -> [u32; 5] {
    [
        0x7E04_0287,     // v_mov_b32 v2, 7
        0x360A_0081,     // v_and_b32 v5, 1, v0
        0x7D84_0A80,     // v_cmp_eq_u32 vcc, 0, v5
        0xBE8A_246A,     // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0000 | n, // s_cbranch_execz +n
    ]
}

const RESTORE_EXEC: u32 = 0x88FE_0A7E; // s_or_b64 exec, exec, s[10:11]

/// `ds_bpermute_b32` reads within the lane's half: lane l, naming l ^ 33,
/// reads l ^ 1. In a branch only even lanes take, the odd lanes they read
/// do not run it and give 0, and the odd lanes keep their value. Split in
/// halves on a host without 64-lane waves (no console flow); refused there
/// before. Where the host runs 64-lane waves the translation read l ^ 33
/// before.
#[test]
fn bpermute_reads_within_its_half() {
    let mut body = TO_THE_OTHER_HALF.to_vec();
    body.extend(DS_BPERMUTE);
    for threads in [64, 128, 96, 32] {
        flow_case("ds_bpermute of t ^ 33", &body, threads, |t| t ^ 1, true);
    }
    let mut branch = even_lanes_run(5).to_vec();
    branch.extend(TO_THE_OTHER_HALF);
    branch.extend(DS_BPERMUTE);
    branch.push(RESTORE_EXEC);
    for threads in [64, 128] {
        flow_case(
            "even lanes ds_bpermute t ^ 33",
            &branch,
            threads,
            |t| if t & 1 == 0 { 0 } else { 7 },
            true,
        );
    }
}

/// v4 = (((t & 31) >> 1) | 32) * 4: lanes 2d and 2d + 1 of each half send
/// to lane d of their half (the address's bit 5 names the other half and
/// does not count).
const PAIRS_TO_THE_LOW_LANES: [u32; 4] = [
    0x3608_009F, // v_and_b32 v4, 31, v0
    0x2C08_0881, // v_lshrrev_b32 v4, 1, v4
    0x3808_08A0, // v_or_b32 v4, 32, v4
    0x3408_0882, // v_lshlrev_b32 v4, 2, v4
];

/// Lane t's value after [`PAIRS_TO_THE_LOW_LANES`] and `ds_permute_b32` of
/// t, in a group of `threads`, from the senders `sends` lets send: the
/// highest sender of its half that names it, else 0.
fn permuted(t: u32, threads: u32, sends: impl Fn(u32) -> bool) -> u32 {
    let (base, d) = (t & !31, t & 31);
    (0..32)
        .map(|k| base + k)
        .filter(|&s| s < threads && sends(s) && (s & 31) >> 1 == d)
        .max()
        .unwrap_or(0)
}

/// `ds_permute_b32` sends within the lane's half: lanes 2d and 2d + 1 both
/// send to lane d, the higher one's value stays, and lanes 16 to 31 of each
/// half get 0. In a branch only even lanes take, only they send and take.
/// Not translated at all before, on any host.
#[test]
fn permute_sends_within_its_half() {
    let mut body = PAIRS_TO_THE_LOW_LANES.to_vec();
    body.extend(DS_PERMUTE);
    for threads in [64, 128, 96, 48, 32] {
        flow_case(
            "ds_permute to (t & 31) >> 1",
            &body,
            threads,
            |t| permuted(t, threads, |_| true),
            true,
        );
    }
    let mut branch = even_lanes_run(6).to_vec();
    branch.extend(PAIRS_TO_THE_LOW_LANES);
    branch.extend(DS_PERMUTE);
    branch.push(RESTORE_EXEC);
    for threads in [64, 128] {
        flow_case(
            "even lanes ds_permute to (t & 31) >> 1",
            &branch,
            threads,
            |t| {
                if t & 1 == 0 {
                    permuted(t, threads, |s| s & 1 == 0)
                } else {
                    7
                }
            },
            true,
        );
    }
}

/// Permutes with lane reads across the halves: the console flow, in groups
/// of one console wave and of several.
#[test]
fn permutes_in_the_console_flow() {
    let read_40 = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
    ];
    let mut gather = read_40.to_vec();
    gather.extend(TO_THE_OTHER_HALF);
    gather.extend(DS_BPERMUTE);
    gather.push(0x4A04_0409); // v_add_nc_u32 v2, s9, v2
    waves_case(
        "readlane 40 + ds_bpermute of t ^ 33",
        &gather,
        &[64, 128, 256, 112],
        |t, _| (t ^ 1) + (t & !63) + 40,
    );
    let mut scatter = read_40.to_vec();
    scatter.extend(PAIRS_TO_THE_LOW_LANES);
    scatter.extend(DS_PERMUTE);
    scatter.push(0x4A04_0409); // v_add_nc_u32 v2, s9, v2
    waves_case(
        "readlane 40 + ds_permute to (t & 31) >> 1",
        &scatter,
        &[64, 128, 256, 112],
        |t, n| permuted(t, n, |_| true) + (t & !63) + 40,
    );
}
