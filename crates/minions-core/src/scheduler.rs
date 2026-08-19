//! Two slots, not a pool.
//!
//! The pipeline is sequential — each node consumes its predecessor's document —
//! so at most one pipeline model is resident, plus the chat companion when the
//! user has it open. There is no allocation problem to solve, and this module is
//! small because of it rather than in spite of it.
//!
//! Cost is measured, never derived: `weights + bytes_per_token × window`,
//! obtained by loading a model at two windows when it is bound to a slot.
//! Deriving it from layer counts was tried and was wrong by a third.

use crate::memory::MemoryProbe;
use anyhow::Result;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCost {
    pub weights: u64,
    pub bytes_per_token: u64,
}

impl ModelCost {
    pub fn at(&self, window: u32) -> u64 {
        self.weights + self.bytes_per_token * window as u64
    }
}

/// Measured on this machine 2026-08-16; see docs/MEASUREMENTS.md.
pub fn measured_default(model: &str) -> Option<ModelCost> {
    let gib = 1024u64 * 1024 * 1024;
    let kib = 1024u64;
    match model {
        m if m.contains("14b") => Some(ModelCost { weights: 7_98 * gib / 100, bytes_per_token: 274 * kib }),
        m if m.contains("7b") => Some(ModelCost { weights: 4_12 * gib / 100, bytes_per_token: 113 * kib }),
        m if m.contains("embed") => Some(ModelCost { weights: 274 * 1024 * 1024, bytes_per_token: 0 }),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// Start now. `evict_chat` when the companion has to go first.
    Start { evict_chat: bool },
    /// Not now, but the machine could hold it once something is released.
    WaitForMemory { needed: u64, available: u64 },
    /// Not ever on this machine at this window — the run halts rather than swaps.
    Impossible { needed: u64, ceiling: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resident {
    pub model: String,
    pub window: u32,
    pub cost: u64,
}

pub struct Scheduler<'a> {
    probe: &'a dyn MemoryProbe,
    /// Held back for the system and for this application. Never lent to models.
    reserve: u64,
    costs: BTreeMap<String, ModelCost>,
    pipeline: Option<Resident>,
    chat: Option<Resident>,
}

impl<'a> Scheduler<'a> {
    pub fn new(probe: &'a dyn MemoryProbe, reserve: u64) -> Self {
        Self { probe, reserve, costs: BTreeMap::new(), pipeline: None, chat: None }
    }

    pub fn learn(&mut self, model: &str, cost: ModelCost) {
        self.costs.insert(model.to_string(), cost);
    }

    pub fn cost_of(&self, model: &str, window: u32) -> Option<u64> {
        self.costs
            .get(model)
            .copied()
            .or_else(|| measured_default(model))
            .map(|c| c.at(window))
    }

    pub fn pipeline(&self) -> Option<&Resident> {
        self.pipeline.as_ref()
    }

    pub fn chat(&self) -> Option<&Resident> {
        self.chat.as_ref()
    }

    /// Memory a model may actually take, after the system's share is held back.
    fn headroom(&self) -> Result<u64> {
        let snap = self.probe.snapshot()?;
        Ok(snap.available_bytes().saturating_sub(self.reserve))
    }

    pub fn admit(&self, model: &str, window: u32) -> Result<Admission> {
        let needed = self
            .cost_of(model, window)
            .ok_or_else(|| anyhow::anyhow!("no measured cost for `{model}`; bind it to a slot first"))?;

        let snap = self.probe.snapshot()?;

        // Whatever the pipeline slot already holds is ours to reuse or release,
        // so it counts as available to the model replacing it. That accounting
        // also makes a separate "already resident" case unnecessary: a model
        // asking for the slot it occupies always fits.
        let mine = self.pipeline.as_ref().map(|r| r.cost).unwrap_or(0);
        let chat_cost = self.chat.as_ref().map(|r| r.cost).unwrap_or(0);
        let free = snap.available_bytes().saturating_sub(self.reserve) + mine;

        if needed <= free {
            return Ok(Admission::Start { evict_chat: false });
        }
        if needed <= free + chat_cost {
            // The companion goes first. It never blocks a pipeline, and costs
            // 0.7 s to bring back.
            return Ok(Admission::Start { evict_chat: true });
        }

        // "Never here" and "not right now" are different questions, and only
        // the machine's total answers the first. Judging both by current
        // availability would make the waiting state unreachable.
        let ceiling = snap.total_bytes().saturating_sub(self.reserve);
        if needed > ceiling {
            return Ok(Admission::Impossible { needed, ceiling })
        }
        Ok(Admission::WaitForMemory { needed, available: free })
    }

    /// Records what admission decided. Eviction of the previous pipeline model
    /// is implicit: the slot holds one.
    pub fn place(&mut self, model: &str, window: u32, evict_chat: bool) -> Result<()> {
        if evict_chat {
            self.chat = None;
        }
        let cost = self
            .cost_of(model, window)
            .ok_or_else(|| anyhow::anyhow!("no measured cost for `{model}`"))?;
        self.pipeline = Some(Resident { model: model.to_string(), window, cost });
        Ok(())
    }

    pub fn place_chat(&mut self, model: &str, window: u32) -> Result<bool> {
        let cost = self
            .cost_of(model, window)
            .ok_or_else(|| anyhow::anyhow!("no measured cost for `{model}`"))?;
        if cost > self.headroom()? {
            return Ok(false);
        }
        self.chat = Some(Resident { model: model.to_string(), window, cost });
        Ok(true)
    }

    pub fn release_pipeline(&mut self) {
        self.pipeline = None;
    }

    pub fn release_chat(&mut self) {
        self.chat = None;
    }
}
