//! Budget logic is tested against scripted pressure, never against whatever the
//! machine happens to be doing.

use minions_core::memory::{MemoryProbe, MemorySnapshot};
use minions_core::scheduler::*;
use std::sync::Mutex;

const GIB: u64 = 1024 * 1024 * 1024;

struct Scripted {
    available: Mutex<u64>,
    total: Mutex<u64>,
}

impl Scripted {
    fn new(gib: f64) -> Self {
        Self { available: Mutex::new((gib * GIB as f64) as u64), total: Mutex::new(36 * GIB) }
    }
    fn with_total(gib: f64, total_gib: f64) -> Self {
        Self {
            available: Mutex::new((gib * GIB as f64) as u64),
            total: Mutex::new((total_gib * GIB as f64) as u64),
        }
    }
    fn set(&self, gib: f64) {
        *self.available.lock().unwrap() = (gib * GIB as f64) as u64;
    }
}

impl MemoryProbe for Scripted {
    fn snapshot(&self) -> anyhow::Result<MemorySnapshot> {
        // One page per byte keeps the arithmetic exact and the intent obvious.
        let free = *self.available.lock().unwrap();
        let total = *self.total.lock().unwrap();
        Ok(MemorySnapshot {
            page_size: 1,
            free,
            active: total.saturating_sub(free),
            inactive: 0,
            speculative: 0,
            wired: 0,
            purgeable: 0,
            compressed: 0,
        })
    }
}

fn gib(n: f64) -> u64 {
    (n * GIB as f64) as u64
}

#[test]
fn measured_costs_match_what_the_machine_reported() {
    // 7.98 GiB + 274 KiB per token; at 8192 that is 10.12 GiB.
    let c = measured_default("qwen2.5-coder:14b").unwrap();
    let at_8k = c.at(8192) as f64 / GIB as f64;
    assert!((at_8k - 10.12).abs() < 0.05, "14B at 8192 should be 10.12 GiB, got {at_8k:.2}");
    let at_16k = c.at(16384) as f64 / GIB as f64;
    assert!((at_16k - 12.26).abs() < 0.05, "14B at 16384 should be 12.26 GiB, got {at_16k:.2}");

    let chat = measured_default("qwen2.5:7b").unwrap();
    let chat_8k = chat.at(8192) as f64 / GIB as f64;
    assert!((chat_8k - 4.97).abs() < 0.05, "7b at 8192 should be 4.97 GiB, got {chat_8k:.2}");
}

#[test]
fn a_model_that_fits_starts_without_disturbing_the_chat() {
    let probe = Scripted::new(18.0);
    let s = Scheduler::new(&probe, gib(2.0));
    assert_eq!(s.admit("qwen2.5:14b", 8192).unwrap(), Admission::Start { evict_chat: false });
}

#[test]
fn the_chat_is_evicted_before_a_pipeline_node_waits() {
    // 17.4 available, 2.0 reserved -> 15.4 headroom. Chat at 8192 takes 4.97,
    // leaving 10.4 free, which a 14B at 16384 (12.26) does not fit into.
    let probe = Scripted::new(17.4);
    let mut s = Scheduler::new(&probe, gib(2.0));
    assert!(s.place_chat("qwen2.5:7b", 8192).unwrap());
    probe.set(17.4 - 4.97);

    match s.admit("qwen2.5-coder:14b", 16384).unwrap() {
        Admission::Start { evict_chat } => assert!(evict_chat, "the companion should go first"),
        other => panic!("expected a start after evicting chat, got {other:?}"),
    }
}

#[test]
fn a_model_too_large_for_the_machine_halts_rather_than_swaps() {
    // Larger than the machine's total, not merely larger than what is free.
    let probe = Scripted::with_total(6.0, 8.0);
    let s = Scheduler::new(&probe, gib(2.0));
    match s.admit("qwen2.5:14b", 32768).unwrap() {
        Admission::Impossible { needed, ceiling } => {
            assert!(needed > ceiling, "needed {needed} should exceed the ceiling {ceiling}");
        }
        other => panic!("expected Impossible, got {other:?}"),
    }
}

#[test]
fn the_same_model_at_the_same_window_is_not_reloaded() {
    let probe = Scripted::new(18.0);
    let mut s = Scheduler::new(&probe, gib(2.0));
    s.place("qwen2.5:14b", 8192, false).unwrap();
    probe.set(18.0 - 10.12); // it is resident now

    assert_eq!(
        s.admit("qwen2.5:14b", 8192).unwrap(),
        Admission::Start { evict_chat: false },
        "a resident model must be reused, not queued behind itself"
    );
}

#[test]
fn switching_models_may_reuse_what_the_slot_already_holds() {
    // Only one pipeline model is resident, so the outgoing one's memory counts
    // as available to the incoming one.
    let probe = Scripted::new(18.0);
    let mut s = Scheduler::new(&probe, gib(2.0));
    s.place("qwen2.5:14b", 16384, false).unwrap();
    probe.set(18.0 - 12.26);

    assert_eq!(
        s.admit("qwen2.5-coder:14b", 16384).unwrap(),
        Admission::Start { evict_chat: false },
        "the slot it is replacing should be counted as free"
    );
}

#[test]
fn the_reserve_is_never_lent_to_a_model() {
    // 12.5 available, 2.0 reserved -> 10.5 headroom; a 14B at 16384 needs 12.26.
    let probe = Scripted::new(12.5);
    let s = Scheduler::new(&probe, gib(2.0));
    assert!(
        !matches!(s.admit("qwen2.5-coder:14b", 16384).unwrap(), Admission::Start { .. }),
        "the model was allowed to eat the system's reserve"
    );
}

#[test]
fn chat_is_refused_rather_than_squeezing_a_running_pipeline() {
    let probe = Scripted::new(3.0);
    let mut s = Scheduler::new(&probe, gib(2.0));
    assert!(!s.place_chat("qwen2.5:7b", 8192).unwrap(), "chat took memory it did not have");
    assert!(s.chat().is_none());
}

#[test]
fn an_unmeasured_model_is_refused_with_a_useful_message() {
    let probe = Scripted::new(30.0);
    let s = Scheduler::new(&probe, gib(2.0));
    let err = s.admit("llama-9000:70b", 8192).unwrap_err().to_string();
    assert!(err.contains("bind it to a slot"), "unhelpful: {err}");
}

#[test]
fn a_learned_cost_overrides_the_measured_default() {
    let probe = Scripted::new(30.0);
    let mut s = Scheduler::new(&probe, gib(2.0));
    let before = s.cost_of("qwen2.5:14b", 8192).unwrap();
    s.learn("qwen2.5:14b", ModelCost { weights: GIB, bytes_per_token: 0 });
    let after = s.cost_of("qwen2.5:14b", 8192).unwrap();
    assert_ne!(before, after);
    assert_eq!(after, GIB, "measurement at binding must win over the default");
}

// ---- gaps found by mutation testing, 2026-08-16 ----

#[test]
fn a_model_that_could_fit_later_waits_instead_of_halting() {
    // 4 GiB free of 36 total: a 14B at 8192 needs 10.12, which the machine can
    // hold once other applications release. That is waiting, not impossible —
    // and the two are only distinguishable by the total.
    let probe = Scripted::with_total(4.0, 36.0);
    let s = Scheduler::new(&probe, gib(2.0));
    match s.admit("qwen2.5:14b", 8192).unwrap() {
        Admission::WaitForMemory { needed, available } => {
            assert!(needed > available, "needed {needed} should exceed available {available}");
        }
        other => panic!("expected WaitForMemory, got {other:?}"),
    }
}

#[test]
fn chat_at_exactly_the_headroom_is_admitted() {
    // The boundary is inclusive: a model that fits precisely must be allowed.
    let chat = measured_default("qwen2.5:7b").unwrap().at(8192);
    let reserve = gib(2.0);
    let probe = Scripted::new((chat + reserve) as f64 / GIB as f64);
    let mut s = Scheduler::new(&probe, reserve);
    assert!(s.place_chat("qwen2.5:7b", 8192).unwrap(), "an exact fit must be admitted");
}

#[test]
fn chat_one_byte_over_the_headroom_is_refused() {
    let chat = measured_default("qwen2.5:7b").unwrap().at(8192);
    let reserve = gib(2.0);
    let probe = Scripted::new((chat + reserve - 1) as f64 / GIB as f64);
    let mut s = Scheduler::new(&probe, reserve);
    assert!(!s.place_chat("qwen2.5:7b", 8192).unwrap(), "an over-fit must be refused");
}

#[test]
fn releasing_a_slot_frees_it_for_the_next_model() {
    let probe = Scripted::new(13.0);
    let mut s = Scheduler::new(&probe, gib(2.0));
    s.place("qwen2.5:14b", 8192, false).unwrap();
    assert!(s.pipeline().is_some());
    s.release_pipeline();
    assert!(s.pipeline().is_none(), "release_pipeline did nothing");

    assert!(s.place_chat("qwen2.5:7b", 8192).unwrap());
    assert!(s.chat().is_some());
    s.release_chat();
    assert!(s.chat().is_none(), "release_chat did nothing");
}

#[test]
fn admission_at_exactly_the_available_memory_starts() {
    let need = measured_default("qwen2.5:14b").unwrap().at(8192);
    let reserve = gib(2.0);
    let probe = Scripted::new((need + reserve) as f64 / GIB as f64);
    let s = Scheduler::new(&probe, reserve);
    assert_eq!(s.admit("qwen2.5:14b", 8192).unwrap(), Admission::Start { evict_chat: false });
}

#[test]
fn the_embedder_costs_its_weights_and_no_cache() {
    let c = measured_default("nomic-embed-text").expect("the embedder must be costed");
    assert_eq!(c.bytes_per_token, 0, "an embedder holds no conversation, so no cache");
    let mb = c.at(16384) as f64 / (1024.0 * 1024.0);
    assert!((mb - 274.0).abs() < 1.0, "expected about 274 MiB, got {mb:.1}");
    assert!(measured_default("something-unheard-of").is_none());
}

#[test]
fn a_model_needing_exactly_the_machines_total_waits_rather_than_being_refused() {
    // The boundary between "not now" and "not ever" is inclusive: a model that
    // fits the machine exactly is possible, just not yet.
    let need = measured_default("qwen2.5:14b").unwrap().at(8192);
    let reserve = gib(2.0);
    let probe = Scripted::with_total(1.0, (need + reserve) as f64 / GIB as f64);
    let s = Scheduler::new(&probe, reserve);
    match s.admit("qwen2.5:14b", 8192).unwrap() {
        Admission::WaitForMemory { .. } => {}
        other => panic!("an exact fit against the total must wait, not be refused: {other:?}"),
    }
}
