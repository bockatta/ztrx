//! Runs the indexed texture pass with synthetic, fully mapped RGBA8 data.

use super::{HEAP_IIDS, HeapDesc, Placement, SHADER_VISIBLE, VIEWS_KIND, create_heap, create_placed_srv, create_srv, heap};
use crate::draw::{HEAP_TEXTURE_BYTES, State};
use crate::object::{self, S_OK, This};
use crate::resource::{self, Memory, PAGEABLE_IIDS};
use std::ptr::null;
use std::sync::atomic::Ordering::Relaxed;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory4};

const MIB: u64 = 1 << 20;

// 2D, linear, single-mip RGBA8. Include the D3D12 description's padding.
#[repr(align(8))]
struct TextureDesc([u32; 14]);

impl TextureDesc {
    fn rgba8(height: u32) -> Self {
        Self([3, 0, 0, 0, 1024, 0, height, 1 | (1 << 16), 28, 1, 0, 1, 0, 0])
    }

    fn resource(&self, memory: &Memory) -> This {
        let mut resource = null();
        assert_eq!(
            resource::create_at_address(
                null(),
                memory.address(),
                self.0.as_ptr().cast(),
                0,
                null(),
                &PAGEABLE_IIDS[0],
                &mut resource,
            ),
            S_OK
        );
        resource
    }
}

fn release(resource: This) {
    let release = object::slot::<extern "system" fn(This) -> u32>(resource, 0x10);
    assert_eq!(release(resource), 0);
}

fn live_bytes() -> u64 {
    resource::HOST_TEXTURES[1].load(Relaxed)
}

#[test]
#[ignore = "requires fresh diagnostics process with D3D12 WARP, a 10 MiB texture budget and rechecks disabled"]
fn indexed_texture_budget() {
    for (var, expected) in [
        ("VICTORY_D3D12_WARP", "1"),
        ("VICTORY_TEXTURE_BUDGET_MB", "10"),
        ("VICTORY_HEAP_RECHECK", "0"),
        ("VICTORY_PLACED_FILLER", "0"),
    ] {
        assert_eq!(std::env::var(var).as_deref(), Ok(expected));
    }
    let device = crate::host::host().expect("D3D12 WARP");
    // SAFETY: queries on the factory, adapter and live device.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
        let warp: IDXGIAdapter1 = factory.EnumWarpAdapter().unwrap();
        let desc = warp.GetDesc1().unwrap();
        let luid = device.device.GetAdapterLuid();
        assert_eq!((luid.LowPart, luid.HighPart), (desc.AdapterLuid.LowPart, desc.AdapterLuid.HighPart));
        desc
    };
    println!("WARP adapter {:04x}:{:04x}", adapter.VendorId, adapter.DeviceId);

    #[repr(C)]
    struct Extended(HeapDesc, Placement);
    let desc = Extended(
        HeapDesc { kind: VIEWS_KIND, count: 2, flags: SHADER_VISIBLE, nodes: 0 },
        Placement { address: 0, size: 0 },
    );
    let mut heap_object = null();
    assert_eq!(create_heap(null(), &desc.0, &HEAP_IIDS[0], &mut heap_object), S_OK);
    let heap = heap(heap_object);
    let host = heap.host().unwrap();
    let state = State::default();
    let pass = || state.heap_textures(&device.new_list().unwrap(), heap, host);

    let backing = Memory::zeroed(8 * MIB, 64).unwrap();
    let baseline = TextureDesc::rgba8(2048).resource(&backing);
    assert!(resource::host_resource(baseline).is_some());
    assert_eq!(live_bytes(), 8 * MIB);

    let desc = TextureDesc::rgba8(512);
    let memory_a = Memory::zeroed(2 * MIB, 64).unwrap();
    let memory_b = Memory::zeroed(2 * MIB, 64).unwrap();
    let slot_a = heap.handle(0).unwrap();
    let slot_b = heap.handle(1).unwrap();

    // Observe a real upload recorded for a copy that already exists at the limit.
    let existing = desc.resource(&memory_a);
    assert!(resource::host_resource(existing).is_some());
    assert_eq!(live_bytes(), 10 * MIB);
    assert!(!resource::room());
    create_srv(null(), existing, null(), slot_a);
    let staged_before = HEAP_TEXTURE_BYTES.load(Relaxed);
    pass();
    let control_staged = HEAP_TEXTURE_BYTES.load(Relaxed) - staged_before;
    let control_live = live_bytes();
    assert!(host.texture_slot_gone(0));
    release(existing);
    assert_eq!(live_bytes(), 8 * MIB);

    // The production queue serves the newest descriptor first. Write B before A.
    let address_b = memory_b.address();
    let mut view_b = [0; 16];
    view_b[..6].copy_from_slice(&[28, 4, 5768, 0, 0, 1]); // RGBA8 Texture2D SRV
    view_b[10] = address_b as u32;
    view_b[11] = (address_b >> 32) as u32;
    create_placed_srv(null(), desc.0.as_ptr().cast(), &view_b, slot_b);
    let a = desc.resource(&memory_a);
    create_srv(null(), a, null(), slot_a);
    let frame = crate::frame::current();
    let (due, _) = host.texture_slots_due(frame, 2);
    assert_eq!(due.iter().map(|&(slot, _)| slot).collect::<Vec<_>>(), [0, 1]);

    pass();
    let first_live = live_bytes();
    let (pending, _) = host.texture_slots_due(frame + 1, 2);
    assert!(host.texture_slot_gone(0));
    release(a);
    let released_live = live_bytes();
    let mut token = 0;
    assert_eq!(crate::frame::wait(null(), 0, 1000, null(), 0, &mut token), S_OK);
    pass();
    let retry_live = live_bytes();
    let (remaining, _) = host.texture_slots_due(crate::frame::current(), 2);

    println!(
        "first live={} MiB, due={pending:?}, B generation={}; after release={} MiB, retry={} MiB, due={remaining:?}; existing staged={} MiB, live={} MiB",
        first_live / MIB, due[1].1, released_live / MIB, retry_live / MIB, control_staged / MIB, control_live / MIB
    );

    // A new descriptor for B changes only the two padding words. The
    // existing placed copy must still be found when the budget is full.
    assert!(!resource::room());
    let placed_count = resource::placed_textures();
    let mut alias = desc.0;
    (alias[1], alias[13]) = (0x1234, 0x5678);
    create_placed_srv(null(), &alias, &view_b, slot_b);
    let frame = crate::frame::current();
    let (alias_due, _) = host.texture_slots_due(frame, 2);
    assert_eq!(alias_due.len(), 1);
    assert_eq!(alias_due[0].0, 1);
    let staged_before = HEAP_TEXTURE_BYTES.load(Relaxed);
    pass();
    let alias_staged = HEAP_TEXTURE_BYTES.load(Relaxed) - staged_before;
    let alias_live = live_bytes();
    let alias_resources = resource::placed_textures() - placed_count;
    let (alias_pending, _) = host.texture_slots_due(frame, 2);
    println!(
        "placed alias staged={} MiB, live={} MiB, new placed resources={alias_resources}, due={alias_pending:?}",
        alias_staged / MIB, alias_live / MIB,
    );

    assert_eq!(first_live, 10 * MIB);
    assert_eq!(pending, [due[1]]);
    assert_eq!(released_live, 8 * MIB);
    assert_eq!(retry_live, 10 * MIB);
    assert!(remaining.is_empty());
    assert_eq!(control_staged, 2 * MIB);
    assert_eq!(control_live, 10 * MIB);
    assert_eq!(alias_staged, 2 * MIB);
    assert_eq!(alias_live, 10 * MIB);
    assert_eq!(alias_resources, 0);
    assert!(alias_pending.is_empty());

    release(heap_object);
    release(baseline);
}
