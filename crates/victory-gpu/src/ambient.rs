//! A stand-in for the indirect light the title traces rays for.
//!
//! With the ray-tracing structure builder off (`crate::pipeline::builder_runs`)
//! the title's rays find nothing. Its program that resolves the diffuse light
//! of its screen probes (`CS_ScreenProbes_TemporalResolve_*`) then leaves the
//! sky's light outdoors, nothing indoors (walls are black where no lamp or
//! sun reaches) and, in blocks of the picture's tiles, what the probes'
//! memory held. Right after that program the light is raised to an estimate
//! and bounded by a multiple of it.
//!
//! That a stand-in can be put there, and in which of the program's images,
//! is beloooo's find (VIctory-Belobug-AMD, 06031237, 4f45ab88, cd9f78ce,
//! GPL-3.0-or-later), which puts a constant there, shaded in creases from the
//! scene's depth. The estimate here is worked out from the frame instead, so
//! it follows the hour, the room and its lamps by itself: the light the
//! visible surfaces send out (the title keeps it for its own screen traces:
//! `CS_CacheScreenSpaceRadiance`, the frame before) is averaged in 64x36
//! cells, and every point gets the mean of the cells weighted by the share of
//! its surroundings they cover, as a directional value (first-order harmonics
//! at 160x90 points, taken between them by depth).
//!
//! What it assumes: what is not on screen sends out what is (with a slow
//! mean of the last seconds, so that turning into a dark corner is not black
//! at once). No occlusion but the creases'.
//!
//! `VICTORY_AMBIENT`:
//! - unset or `0`/`off`: the title's own output.
//! - a number: that share of the estimate (1: all of it).
//! - `constant:<level>`: the constant, as beloooo's fork has it.
//!
//! `VICTORY_AMBIENT_SHADE=<strength>,<radius in metres>` sets the creases.
//!
//! The air of a room is lit by the same probes: without rays they hold the
//! sky's light indoors too, a brown veil at sunset. `VICTORY_AMBIENT_AIR`
//! bounds what the air scatters by a multiple of the light around.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// The pixel program that renders the sky dome as the title reflects it:
/// `128x128 R16G16B16A16_FLOAT` in the title's own linear HDR light units
/// (Belobug e48eb83e / 4f62c2a5). Used to match indirect ambient exposure
/// across noon interiors (cafe), sunset (prison), and night automatically.
pub(crate) const SKY: &[u8] = b"ps_sky_paraboloid_reflection";

static SKY_IMAGE: Mutex<Option<Arc<crate::resource::HostTexture>>> = Mutex::new(None);

/// Records the host copy of the title's sky dome texture when drawn.
pub(crate) fn sky_drawn(image: usize) {
    let Some(copy) = crate::resource::host_copy(image as This) else {
        return;
    };
    // SAFETY: a live host texture.
    let d = unsafe { copy.texture.GetDesc() };
    if d.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        return;
    }
    let mut held = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner());
    if !held.as_ref().is_some_and(|h| Arc::ptr_eq(h, &copy)) {
        *held = Some(copy);
    }
}

/// Sky-dome exposure curve from the live graphics module (or `None` if disabled).
fn sky_follow(g: &crate::graphics::Settings) -> Option<[f32; 3]> {
    g.sky_enabled.then_some([g.sky_gain, g.sky_least, g.sky_power])
}

/// Programs of the title the stand-in follows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    /// Resolves the diffuse light.
    Resolve,
    /// Keeps the light visible surfaces send out.
    Radiance,
    /// Adds the lamps' light to what the air scatters (the last program
    /// that writes that volume before it is summed up along the view).
    Air,
    /// Writes the title's reflections (`CS_UpdateHistory`, Belobug c3bd5612):
    /// without rays, glossy surfaces indoors get an unoccluded sky glare.
    Reflect,
}

/// The program's part in this, by its name in the container.
pub(crate) fn role(container: &[u8]) -> Option<Role> {
    let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
    if has(b"CS_ScreenProbes_TemporalResolve") {
        Some(Role::Resolve)
    } else if has(b"CS_CacheScreenSpaceRadiance") {
        Some(Role::Radiance)
    } else if has(b"CS_AccumulateLocalLightInscatteringVolume") {
        Some(Role::Air)
    } else if has(b"CS_UpdateHistory") {
        Some(Role::Reflect)
    } else {
        None
    }
}

/// Where the resolve writes the diffuse light (R9G9B9E5 as 32-bit integers)
/// and reads the scene's depth, and where the other writes the light of
/// visible surfaces: root parameter and place in its table, in dwords.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);
pub(crate) const RADIANCE: (u8, u32) = (1, 16);
/// Where the light the air scatters is.
pub(crate) const AIR: (u8, u32) = (1, 32);
/// Where `CS_UpdateHistory` writes reflections (R9G9B9E5 as R32_UINT).
pub(crate) const REFLECTIONS: (u8, u32) = (1, 40);

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Debug)]
enum Set {
    Off,
    Estimate(f32),
    Constant(f32),
}

#[cfg(test)]
fn parse(set: &str) -> Set {
    let set = set.trim();
    let number = |t: &str| t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0);
    match set.strip_prefix("constant:") {
        Some(level) => number(level).map_or(Set::Off, Set::Constant),
        None => number(set).map_or(Set::Off, Set::Estimate),
    }
}

/// Opt-in hybrid mode (`VICTORY_AMBIENT_WITH_RT=1`): allow subtle ambient
/// bounce + reflection soft-knee to fill unmapped indoor voids alongside the
/// ray-tracing structure builder (`VICTORY_BVH_BUILDER=1`).
fn with_rt() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_AMBIENT_WITH_RT").is_some_and(|v| v == "1"))
}

/// Whether the stand-in is on: enabled in [`crate::graphics`], and either the
/// RT structure builder is off or hybrid `VICTORY_AMBIENT_WITH_RT=1` is active.
pub(crate) fn on() -> bool {
    crate::graphics::current().ambient_enabled && (!crate::pipeline::builder_runs() || with_rt())
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=16, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1), UAV(u2)"
#define CX 64
#define CY 36
#define CELLS (CX * CY)
#define PX 160
#define PY 90
cbuffer Set : register(b0) {
    float level; float constant; float strength; float radius;
    float near; float tan_y; uint width; uint height;
    float follow; float most; float air; float reflect_floor;
    uint ao_taps; uint contact_steps; uint gather_step; float ceiling_floor;
};
Texture2D<float> depth : register(t0);
// Cells:
//   [0 .. CELLS)                 = radiance RGB (a = 1)
//   [CELLS]                      = slow temporal mean radiance RGB
//   [CELLS + 1 .. 2*CELLS + 1)   = view-space position XYZ, w = z^2 area weight
//   [2*CELLS + 1 .. 3*CELLS + 1) = view-space emitter surface normal XYZ
//   [3*CELLS + 1]                = sky-dome exposure-matched ambient RGB (a = 1 when active)
RWStructuredBuffer<float4> cells : register(u1);
// Three a point: red, green and blue, each a mean (w) and a direction (xyz).
RWStructuredBuffer<float4> points : register(u2);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = min(near / max(d, 1e-7), 1000.0);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}

// Reconstruct view-space surface normal from depth with silhouette-preserving diffs.
float3 surface_normal(int2 p, float2 size, float3 c) {
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
    return n;
}

#ifdef KEEP
RWTexture2D<float3> radiance : register(u0);
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void keep(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    radiance.GetDimensions(w, h);
    float3 sum = 0;
    for (uint y = 0; y < 5; y++) for (uint x = 0; x < 5; x++) {
        uint2 p = uint2((float2(at.xy) + (float2(x, y) + 0.5) / 5.0) * float2(w, h) / float2(CX, CY));
        float3 c = max(radiance[p], 0.0);
        c = any(isnan(c)) || any(isinf(c)) ? 0.0 : c;
        // Lamps and glints count, but not for everything.
        sum += c / max(1.0, dot(c, LUMA) / 20.0);
    }
    cells[at.y * CX + at.x] = float4(sum / 25.0, 1.0);
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void mean() {
    float3 sum = 0;
    for (uint i = 0; i < CELLS; i++) sum += cells[i].rgb;
    float4 was = cells[CELLS];
    float3 now = sum / CELLS;
    cells[CELLS] = float4(was.a > 0.0 ? lerp(was.rgb, now, follow) : now, 1.0);
}
#endif

#ifdef PLACE
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void place(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    int2 q = int2((float2(at.xy) + 0.5) * size / float2(CX, CY));
    float3 p = seen(q, size);
    float3 n = surface_normal(q, size, p);
    uint idx = at.y * CX + at.x;
    cells[CELLS + 1 + idx] = float4(p, p.z * p.z);
    cells[2 * CELLS + 1 + idx] = float4(n, 1.0);
}
#endif

#ifdef GATHER
[RootSignature(RS)]
[numthreads(8, 2, 1)]
void gather(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(PX, PY)), size);
    float r2 = 0.10 + 0.0014 * p.z * p.z;
    float whole = 0;
    float3 toward = 0, light = 0, red = 0, green = 0, blue = 0;
    uint step = max(gather_step, 1u);
    for (uint i = 0; i < CELLS; i += step) {
        float4 c = cells[CELLS + 1 + i];
        float3 nc = cells[2 * CELLS + 1 + i].xyz;
        float3 L = cells[i].rgb;
        float3 d = c.xyz - p;
        float dd = dot(d, d);
        float3 dir = d / sqrt(dd + 0.25 * r2);
        // Physical emitter facing: surfaces bounce light forward into the room they face.
        float emit_facing = saturate(dot(-dir, nc) * 0.85 + 0.15);
        float w = c.w * emit_facing / (dd + r2);
        whole += w;
        toward += w * dir;
        light += w * L;
        red += w * L.r * dir;
        green += w * L.g * dir;
        blue += w * L.b * dir;
    }
    whole = max(whole, 1e-6);
    light /= whole;
    toward /= whole;
    uint at0 = (at.y * PX + at.x) * 3;
    points[at0] = float4(red / whole - light.r * toward, light.r);
    points[at0 + 1] = float4(green / whole - light.g * toward, light.g);
    points[at0 + 2] = float4(blue / whole - light.b * toward, light.b);
}
#endif

#ifdef AIR
RWTexture3D<float3> scattered : register(u0);
// No more than a multiple of the light around: of the room as the last
// seconds showed it, and of the surface seen in this direction (towards a
// window the air outside may be as bright as the sky is).
[RootSignature(RS)]
[numthreads(8, 8, 2)]
void clear_air(uint3 at : SV_DispatchThreadID) {
    uint w, h, d;
    scattered.GetDimensions(w, h, d);
    if (at.x >= w || at.y >= h || at.z >= d) return;
    uint2 q = min(uint2((float2(at.xy) + 0.5) / float2(w, h) * float2(PX, PY)), uint2(PX - 1, PY - 1));
    uint at0 = (q.y * PX + q.x) * 3;
    float3 seen_there = float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
    float far = float(at.z) / float(d);
    float cap = air * lerp(dot(cells[CELLS].rgb, LUMA), dot(seen_there, LUMA), far);
    float3 c = scattered[at];
    float has = dot(c, LUMA);
    if (has > cap) scattered[at] = c * (cap / has);
}
#endif

#ifdef RAISE
RWTexture2D<uint> light : register(u0);
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void raise(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    if (at.x >= width || at.y >= height) return;
    int2 p = int2(at.xy);
    float3 c = seen(p, size);
    float3 n = surface_normal(p, size, c);
    // Configurable dual-scale GTAO / contact + architectural corner AO + bent normal.
    float open = 1.0;
    float3 bent_n = n;
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    if (strength > 0.0 && ao_taps > 0u && c.z < 350.0) {
        float px_base = radius * size.y / (2.0 * tan_y * c.z);
        float3 bent = float3(0.0, 0.0, 0.0);
        // Scale 1: Tight contact AO (4 or 8 taps depending on graphics preset).
        float r_tight = max(radius * 0.35, 0.14);
        float px_tight = min(px_base * 0.35, 36.0);
        float sum_tight = 0.0;
        int nt = int(min(ao_taps, 8u));
        if (px_tight >= 1.25 && nt > 0) {
            for (int i = 0; i < nt; i++) {
                float a = i * 2.399963 + noise * 6.2831853;
                float2 o = float2(cos(a), sin(a)) * (sqrt((float(i) + 0.5) / float(nt)) * px_tight);
                float3 v = seen(p + int2(round(o)), size) - c;
                float vv = dot(v, v);
                float inv_d = rsqrt(vv + 1e-6);
                float over = dot(v, n) - 0.003 * c.z - 0.005;
                float falloff = saturate(1.0 - vv / (r_tight * r_tight));
                float occ = max(over, 0.0) * inv_d * falloff;
                sum_tight += occ;
                bent += (v * inv_d) * (1.0 - occ);
            }
            sum_tight /= float(nt);
        }
        // Scale 2: Medium architectural corner AO (only when ao_taps > 8, e.g. Cinema preset).
        float r_wide = radius * 1.15;
        float px_wide = min(px_base * 1.15, 110.0);
        float sum_wide = 0.0;
        int nw = ao_taps > 8u ? int(min(ao_taps - 8u, 12u)) : 0;
        if (px_wide >= 2.0 && nw > 0) {
            for (int i = 0; i < nw; i++) {
                float a = (i + 8) * 2.399963 + noise * 6.2831853;
                float2 o = float2(cos(a), sin(a)) * (sqrt((float(i) + 0.5) / float(nw)) * px_wide);
                float3 v = seen(p + int2(round(o)), size) - c;
                float vv = dot(v, v);
                float inv_d = rsqrt(vv + 1e-6);
                float over = dot(v, n) - 0.005 * c.z - 0.010;
                float falloff = saturate(1.0 - vv / (r_wide * r_wide));
                float occ = max(over, 0.0) * inv_d * falloff;
                sum_wide += occ;
                bent += (v * inv_d) * (1.0 - occ);
            }
            sum_wide /= float(nw);
        }
        float ao_tight = saturate(1.0 - strength * 1.75 * sum_tight);
        float ao_wide  = nw > 0 ? saturate(1.0 - strength * 1.65 * sum_wide) : ao_tight;
        float dist_fade = saturate(1.0 - (c.z - 220.0) / 130.0);
        // Protect downward-facing indoor ceilings from harsh AO corner crush.
        float ao_floor = n.y < -0.55 ? 0.34 : 0.11;
        float raw_ao = saturate(ao_tight * 0.55 + ao_wide * 0.45);
        float ao_shaped = pow(raw_ao, 1.15);
        open = lerp(1.0, lerp(ao_floor, 1.0, ao_shaped), dist_fade);
        bent_n = normalize(n + bent * (0.045 * dist_fade));
    }
    float3 found;
    if (constant > 0.0) {
        found = float3(1.0, 0.98, 0.94) * constant * open;
        light[p] = pack(found);
        return;
    }
    float3 all;
    float3 dir_sh;
    float3 sh_vec = float3(0.0, 0.0, 0.0);
    if (gather_step == 0u) {
        // Minimal preset: fast sky/room-mean fill without SH1 point lookups.
        all = max(cells[CELLS].rgb, float3(0.02, 0.02, 0.02));
        dir_sh = all;
    } else {
        // Between the four points around, those at its own depth first; which
        // four is dithered a little.
        float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5 + (noise - 0.5) * 0.5;
        int2 g0 = int2(floor(g));
        float2 f = g - float2(g0);
        float4 sr = 0, sg = 0, sb = 0;
        float sum = 0;
        for (int k = 0; k < 4; k++) {
            int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
            float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
            float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
            float w = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
            uint at0 = (q.y * PX + q.x) * 3;
            sr += w * points[at0];
            sg += w * points[at0 + 1];
            sb += w * points[at0 + 2];
            sum += w;
        }
        all = float3(sr.w, sg.w, sb.w) / sum;
        // Directional SH1 bounce evaluated along bent normal for grounded colored bounce.
        dir_sh = all + 1.55 * float3(dot(sr.xyz, bent_n), dot(sg.xyz, bent_n), dot(sb.xyz, bent_n)) / sum;
        dir_sh = clamp(dir_sh, 0.18 * all, 2.8 * all);
        sh_vec = (sr.xyz * LUMA.r + sg.xyz * LUMA.g + sb.xyz * LUMA.b) / sum;
    }
    float3 bounce = lerp(dir_sh, cells[CELLS].rgb, 0.12) * level;
    // Screen-space directional contact shadows (configurable steps; disabled on
    // downward-facing ceilings n.y <= -0.55 so ceilings never self-shadow black).
    float dir_shadow = 1.0;
    if (strength > 0.0 && contact_steps > 0u && n.y > -0.55 && c.z < 120.0) {
        float sh_len = length(sh_vec);
        float3 up_bias = float3(0.0, 0.78, -0.32);
        float3 l_dir = sh_len > 1e-4 ? (sh_vec / sh_len) : up_bias;
        float3 ray_dir = normalize(lerp(up_bias, l_dir, saturate(sh_len / max(dot(all, LUMA) * 0.45, 1e-4))));
        float aspect = size.x / size.y;
        float occ_max = 0.0;
        float max_reach = min(radius * 1.35, 1.15);
        float thick = max(0.04, 0.018 * c.z);
        int ns = int(min(contact_steps, 8u));
        for (int s = 0; s < ns; s++) {
            float frac_s = (float(s) + 0.35 + 0.65 * noise) / float(ns);
            float dist_m = max_reach * (frac_s * frac_s);
            float3 sp = c + ray_dir * dist_m;
            if (sp.z <= near) break;
            float2 sndc = float2(sp.x / (sp.z * tan_y * aspect), -sp.y / (sp.z * tan_y));
            int2 sq = int2((sndc * 0.5 + 0.5) * size);
            if (sq.x < 0 || sq.y < 0 || sq.x >= int(width) || sq.y >= int(height)) break;
            float scene_z = seen(sq, size).z;
            float dz = sp.z - scene_z;
            if (dz > 0.015 && dz < thick + 0.35 * dist_m) {
                float weight = saturate(1.0 - frac_s * 0.72) * saturate(1.0 - dz / (thick + 0.45 * dist_m));
                occ_max = max(occ_max, weight);
            }
        }
        float s_fade = saturate(1.0 - (c.z - 60.0) / 60.0);
        dir_shadow = lerp(1.0, lerp(1.0, 0.32, occ_max * saturate(strength * 0.75)), s_fade);
    }
    float4 sky_entry = cells[3 * CELLS + 1];
    float hemi_floor = clamp(ceiling_floor, 0.36, 0.85);
    if (sky_entry.a > 0.0) {
        // Calibrated sky-dome indirect floor with lifted ceiling hemisphere so
        // indoor ceilings stay illuminated while sun shadows stay crisp outdoors.
        float all_luma = max(dot(all, LUMA), 1e-4);
        float3 dir_ratio = clamp(dir_sh / all_luma, 0.45, 1.90);
        float3 room_avg = max(cells[CELLS].rgb, float3(0.008, 0.0078, 0.0075));
        float room_luma = max(dot(room_avg, LUMA), 0.008);
        float3 room_tint = lerp(float3(1.0, 1.0, 1.0), clamp(room_avg / room_luma, 0.55, 1.55), 0.42);
        float hemi = lerp(hemi_floor, 1.0, saturate(bent_n.y * 0.5 + 0.5));
        float indoor_fill = lerp(1.0, 0.48, saturate((c.z - 12.0) / 18.0));
        float3 sky_fill = sky_entry.rgb * room_tint * lerp(float3(1.0, 1.0, 1.0), dir_ratio, 0.60)
            * hemi * indoor_fill * max(level, 0.65);
        bounce = max(bounce, sky_fill);
    } else if (reflect_floor > 0.0) {
        float3 room_avg = max(cells[CELLS].rgb, float3(0.012, 0.0118, 0.0112));
        float room_luma = max(dot(room_avg, LUMA), 0.012);
        float3 room_tint = lerp(float3(1.0, 0.98, 0.94), clamp(room_avg / room_luma, 0.4, 1.8), 0.55);
        float hemi = lerp(hemi_floor, 1.0, saturate(bent_n.y * 0.5 + 0.5));
        bounce = max(bounce, room_tint * reflect_floor * hemi);
    }
    found = bounce * open * dir_shadow;
    if (most <= 0.0) {
        light[p] = pack(found);
        return;
    }
    float3 was = unpack(light[p]);
    // Keep the title's own screen-space/RT traced bounce & sky light where valid,
    // modulated gently by contact shadow contrast.
    float top = max(dot(found, LUMA) * most, 1e-6);
    float has = dot(was, LUMA);
    was *= min(1.0, top / max(has, 1e-6)) * lerp(1.0, open * dir_shadow, 0.55);
    light[p] = pack(max(was, found));
}
#endif

#ifdef SKY
#define RS_SKY "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0)), UAV(u1)"
cbuffer SkySet : register(b0) {
    float sky_gain; float sky_least; float sky_power; float sky_follow;
};
Texture2DArray<float4> sky_tex : register(t0);
[RootSignature(RS_SKY)]
[numthreads(1, 1, 1)]
void sky_mean() {
    uint sw, sh, slices;
    sky_tex.GetDimensions(sw, sh, slices);
    if (sw == 0 || sh == 0) {
        cells[3 * CELLS + 1] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    float3 sum = 0.0;
    const int N = 8;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float3 there = sky_tex.Load(int4((x * 2 + 1) * sw / (N * 2), (y * 2 + 1) * sh / (N * 2), 0, 0)).rgb;
            there = any(isnan(there)) || any(isinf(there)) ? 0.0 : there;
            sum += clamp(there, 0.0, 8.0);
        }
    }
    float3 mean = sum / (N * N);
    float m = dot(mean, LUMA);
    const float3 NIGHT = float3(0.6, 0.75, 1.0);
    float3 hue = lerp(NIGHT, lerp(float3(1.0, 1.0, 1.0), mean / max(m, 1e-6), 0.45), saturate(m / 0.05));
    float3 now = hue * (sky_gain * pow(max(m, 0.0), sky_power) + sky_least);
    float4 was = cells[3 * CELLS + 1];
    cells[3 * CELLS + 1] = float4(was.a > 0.0 ? lerp(was.rgb, now, sky_follow) : now, 1.0);
}
#endif

#ifdef REFLECT
RWTexture2D<uint> reflected : register(u0);
float3 unpack_r(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack_r(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void clamp_reflect(uint3 at : SV_DispatchThreadID) {
    if (at.x >= width || at.y >= height) return;
    int2 p = int2(at.xy);
    float3 c = unpack_r(reflected[p]);
    float4 sky_entry = cells[3 * CELLS + 1];
    float3 rad_base = cells[CELLS].rgb * level;
    float3 base = constant > 0.0
        ? float3(1.0, 0.98, 0.94) * constant
        : (sky_entry.a > 0.0 ? max(rad_base, sky_entry.rgb * level) : rad_base);
    float base_luma = max(dot(base, LUMA), 0.02);
    float has = dot(c, LUMA);
    if (air > 0.0) {
        // Soft-knee shoulder: keeps glossy SSR reflections crisp up to `knee`
        // while suppressing unoccluded indoor white sky-glaze on eyes/floors/cloth.
        float knee = air * base_luma;
        if (has > knee) {
            float excess = has - knee;
            float compressed = knee + excess / (1.0 + excess / max(knee * 0.50, 0.04));
            c *= compressed / max(has, 1e-6);
        }
    }
    if (reflect_floor > 0.0) {
        c = max(c, base * reflect_floor);
    }
    reflected[p] = pack_r(c);
}
#endif
"#;

/// Bytes of the cells' buffer (radiance + mean + positions + emitter normals + sky exposure) and of the points'.
const CELLS: u64 = 64 * 36;
const CELL_BYTES: u64 = (3 * CELLS + 2) * 16;
const POINT_BYTES: u64 = 160 * 90 * 48;
const SKY_SLOTS: u32 = 32;

struct Pass {
    root: ID3D12RootSignature,
    sky_root: ID3D12RootSignature,
    keep: ID3D12PipelineState,
    mean: ID3D12PipelineState,
    place: ID3D12PipelineState,
    gather: ID3D12PipelineState,
    raise: ID3D12PipelineState,
    air: ID3D12PipelineState,
    reflect: ID3D12PipelineState,
    sky_mean: ID3D12PipelineState,
    sky_heap: ID3D12DescriptorHeap,
    sky_cpu: usize,
    sky_gpu: u64,
    sky_step: u32,
    sky_next: AtomicU32,
    cells: ID3D12Resource,
    points: ID3D12Resource,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device, buffer: impl Fn(u64) -> Option<ID3D12Resource>) -> windows::core::Result<Pass> {
    let mut root: Option<ID3D12RootSignature> = None;
    let mut pipeline = |part: &str, entry| -> windows::core::Result<ID3D12PipelineState> {
        let code = crate::host::compile(&format!("#define {part}\n{SOURCE}"), entry, s!("cs_5_1"))?;
        // SAFETY: the root signature is inside the shader; plain creations.
        unsafe {
            if root.is_none() {
                root = Some(device.CreateRootSignature(0, &code)?);
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            device.CreateComputePipelineState(&desc)
        }
    };
    let keep = pipeline("KEEP", s!("keep"))?;
    let mean = pipeline("KEEP", s!("mean"))?;
    let place = pipeline("PLACE", s!("place"))?;
    let gather = pipeline("GATHER", s!("gather"))?;
    let raise = pipeline("RAISE", s!("raise"))?;
    let air = pipeline("AIR", s!("clear_air"))?;
    let reflect = pipeline("REFLECT", s!("clamp_reflect"))?;
    let sky_code = crate::host::compile(&format!("#define SKY\n{SOURCE}"), s!("sky_mean"), s!("cs_5_1"))?;
    // SAFETY: self-contained root signature inside compiled bytecode; plain D3D12 creations.
    let (sky_root, sky_mean, sky_heap, sky_cpu, sky_gpu, sky_step) = unsafe {
        let sky_root: ID3D12RootSignature = device.CreateRootSignature(0, &sky_code)?;
        let sky_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&sky_root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: sky_code.as_ptr().cast(),
                BytecodeLength: sky_code.len(),
            },
            ..Default::default()
        };
        let sky_mean = device.CreateComputePipelineState(&sky_desc)?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let sky_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SKY_SLOTS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        let sky_cpu = sky_heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let sky_gpu = sky_heap.GetGPUDescriptorHandleForHeapStart().ptr;
        let sky_step = device.GetDescriptorHandleIncrementSize(kind);
        (sky_root, sky_mean, sky_heap, sky_cpu, sky_gpu, sky_step)
    };
    let made = |bytes| buffer(bytes).ok_or_else(windows::core::Error::empty);
    Ok(Pass {
        root: root.ok_or_else(windows::core::Error::empty)?,
        sky_root,
        keep,
        mean,
        place,
        gather,
        raise,
        air,
        reflect,
        sky_mean,
        sky_heap,
        sky_cpu,
        sky_gpu,
        sky_step,
        sky_next: AtomicU32::new(0),
        cells: made(CELL_BYTES)?,
        points: made(POINT_BYTES)?,
    })
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        make(&host.device, |bytes| host.uav_buffer(bytes))
            .inspect_err(|e| tracing::warn!(error = %e, "no stand-in for the indirect light"))
            .ok()
    })
    .as_ref()
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Whether the light of a frame has been kept: the estimate has something
/// to begin with.
static KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What the passes read and write, as views in the heap the list has set.
pub(crate) struct Views {
    /// The image the title's program just wrote.
    pub(crate) image: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The scene's depth, for [`Role::Resolve`].
    pub(crate) depth: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The size of the image, and how many slices it has.
    pub(crate) size: [u32; 2],
    pub(crate) slices: u32,
    /// The title's currently bound descriptor heaps (restored after sky-dome sampling).
    pub(crate) view_heap: ID3D12DescriptorHeap,
    pub(crate) sampler_heap: Option<ID3D12DescriptorHeap>,
}

/// Samples the title's sky dome (`ps_sky_paraboloid_reflection`) into
/// `cells[3 * CELLS + 1]` and restores the list's descriptor heaps.
fn update_sky(list: &HostList, pass: &Pass, views: &Views, g: &crate::graphics::Settings) {
    let Some([gain, least, power]) = sky_follow(g) else {
        return;
    };
    let Some(sky_copy) = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
        return;
    };
    let Some(host) = host() else {
        return;
    };
    let read_state =
        (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0) as u32;
    list.keep(&sky_copy.texture);
    list.settle(&sky_copy, read_state);
    // SAFETY: live sky texture; ring slot in our own shader-visible heap; valid open command list.
    unsafe {
        let d = sky_copy.texture.GetDesc();
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                width = d.Width,
                height = d.Height,
                slices = d.DepthOrArraySize,
                gain,
                least,
                power,
                "ambient light follows the sky dome for time-of-day exposure"
            );
        });
        let slot = pass.sky_next.fetch_add(1, Ordering::Relaxed) % SKY_SLOTS;
        let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: pass.sky_cpu + (slot * pass.sky_step) as usize,
        };
        let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: pass.sky_gpu + u64::from(slot * pass.sky_step),
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: d.Format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    FirstArraySlice: 0,
                    ArraySize: u32::from(d.DepthOrArraySize).max(1),
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        host.device.CreateShaderResourceView(&sky_copy.texture, Some(&srv), cpu);
        let l = &list.list;
        let sky_consts = [gain.to_bits(), least.to_bits(), power.to_bits(), 0.08f32.to_bits()];
        l.SetDescriptorHeaps(&[Some(pass.sky_heap.clone())]);
        l.SetComputeRootSignature(&pass.sky_root);
        l.SetComputeRoot32BitConstants(0, 4, sky_consts.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, pass.cells.GetGPUVirtualAddress());
        l.SetPipelineState(&pass.sky_mean);
        l.Dispatch(1, 1, 1);
        uav_barrier(l);
        // Restore the title's heaps before running the main resolve/reflect passes.
        if let Some(sampler) = &views.sampler_heap {
            l.SetDescriptorHeaps(&[Some(views.view_heap.clone()), Some(sampler.clone())]);
        } else {
            l.SetDescriptorHeaps(&[Some(views.view_heap.clone())]);
        }
    }
}

/// After the title's program of that role, in `list` with the program's view
/// heap set. Replaces the list's compute root signature and pipeline; false
/// if it recorded nothing.
pub(crate) fn after(list: &HostList, role: Role, views: &Views) -> bool {
    let Some(pass) = pass() else {
        return false;
    };
    let g = crate::graphics::current();
    if !g.ambient_enabled {
        return false;
    }
    let (level, constant) = if g.ambient_constant > 0.0 {
        (0.0f32, g.ambient_constant)
    } else if g.ambient_level > 0.0 {
        (g.ambient_level, 0.0f32)
    } else {
        return false;
    };
    if role != Role::Radiance && constant == 0.0 && !KEPT.load(Ordering::Relaxed) {
        return false;
    }
    if role == Role::Air && (constant > 0.0 || g.air_cap <= 0.0 || g.gather_step == 0) {
        return false;
    }
    let (reflect_cap, reflect_floor) = (g.reflect_cap, g.reflect_floor);
    if role == Role::Reflect && reflect_cap <= 0.0 && reflect_floor <= 0.0 {
        return false;
    }
    static SAID: Mutex<u32> = Mutex::new(0);
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if *said < 4 {
            *said += 1;
            tracing::info!(
                ?role,
                preset = g.preset.name(),
                size = ?views.size,
                level,
                constant,
                ao_taps = g.ao_taps,
                contact_steps = g.contact_steps,
                gather_step = g.gather_step,
                "a stand-in for the indirect light"
            );
        }
    }
    list.keep(&pass.cells);
    list.keep(&pass.points);
    if role == Role::Resolve && constant == 0.0 {
        update_sky(list, pass, views, &g);
    }
    let floats = [level, constant, g.ao_strength, g.ao_radius, 0.1, 0.45];
    let mut words = [0u32; 16];
    for (to, v) in words.iter_mut().zip(floats) {
        *to = v.to_bits();
    }
    words[6] = views.size[0];
    words[7] = views.size[1];
    words[8] = 0.04f32.to_bits();
    words[9] = g.ambient_most.to_bits();
    words[10] = if role == Role::Reflect { reflect_cap } else { g.air_cap }.to_bits();
    words[11] = if role == Role::Resolve { g.ambient_floor } else { reflect_floor }.to_bits();
    words[12] = g.ao_taps;
    words[13] = g.contact_steps;
    words[14] = g.gather_step;
    words[15] = g.ceiling_floor.to_bits();
    let l = &list.list;
    // SAFETY: an open list with the heap of the views set; the title's
    // program has just used both views, so their textures are in the states
    // these views need; buffers the list keeps.
    unsafe {
        uav_barrier(l);
        l.SetComputeRootSignature(&pass.root);
        l.SetComputeRoot32BitConstants(0, 16, words.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, views.depth);
        l.SetComputeRootDescriptorTable(2, views.image);
        l.SetComputeRootUnorderedAccessView(3, pass.cells.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, pass.points.GetGPUVirtualAddress());
        let run = |pso: &ID3D12PipelineState, x, y, z| {
            l.SetPipelineState(pso);
            l.Dispatch(x, y, z);
            uav_barrier(l);
        };
        match role {
            Role::Radiance => {
                run(&pass.keep, 8, 9, 1);
                run(&pass.mean, 1, 1, 1);
                KEPT.store(true, Ordering::Relaxed);
            }
            Role::Air => run(&pass.air, views.size[0].div_ceil(8), views.size[1].div_ceil(8), views.slices.div_ceil(2)),
            Role::Reflect => run(&pass.reflect, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1),
            Role::Resolve => {
                if constant == 0.0 && g.gather_step > 0 {
                    run(&pass.place, 8, 9, 1);
                    run(&pass.gather, 20, 45, 1);
                }
                run(&pass.raise, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings() {
        assert_eq!(parse(""), Set::Off);
        assert_eq!(parse("off"), Set::Off);
        assert_eq!(parse("0"), Set::Off);
        assert_eq!(parse("0.8"), Set::Estimate(0.8));
        assert_eq!(parse("constant:0.3"), Set::Constant(0.3));
        assert_eq!(parse("constant:nan"), Set::Off);
    }

    #[test]
    fn roles_by_name() {
        assert_eq!(role(b"..CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped.."), Some(Role::Resolve));
        assert_eq!(role(b"..CS_CacheScreenSpaceRadiance_eed2fc35.."), Some(Role::Radiance));
        assert_eq!(role(b"CS_AccumulateLocalLightInscatteringVolume_7e2f5eef"), Some(Role::Air));
        assert_eq!(role(b"CS_TemporalResolve"), None);
    }

    /// The passes compile with the system's compiler and the host takes them.
    #[test]
    fn the_passes_build() {
        let Some(host) = host() else {
            return;
        };
        make(&host.device, |bytes| host.uav_buffer(bytes)).expect("the passes");
    }
}
