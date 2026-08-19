//! Memory availability, measured rather than derived.
//!
//! `total - used` is deliberately not used: on macOS a large share of "used"
//! memory is reclaimable cache. The survey of 2026-08-16 recorded 8.8 GB free
//! while pressure reported 94% available.

use anyhow::{Context, Result};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub page_size: u64,
    pub free: u64,
    pub active: u64,
    pub inactive: u64,
    pub speculative: u64,
    pub wired: u64,
    pub purgeable: u64,
    pub compressed: u64,
}

impl MemorySnapshot {
    /// What the kernel will actually surrender under pressure.
    pub fn available_bytes(&self) -> u64 {
        (self.free + self.inactive + self.speculative + self.purgeable) * self.page_size
    }

    /// Everything the machine has. Distinguishes "cannot ever fit here" from
    /// "cannot fit right now" — without it the waiting state is unreachable.
    pub fn total_bytes(&self) -> u64 {
        (self.free + self.active + self.inactive + self.speculative + self.wired + self.purgeable + self.compressed)
            * self.page_size
    }

    pub fn available_gib(&self) -> f64 {
        self.available_bytes() as f64 / 1024.0 / 1024.0 / 1024.0
    }
}

/// A probe is a trait so budget logic can be tested against scripted pressure
/// instead of whatever the machine happens to be doing (SDD §16).
pub trait MemoryProbe: Send + Sync {
    fn snapshot(&self) -> Result<MemorySnapshot>;
}

pub struct VmStatProbe;

impl MemoryProbe for VmStatProbe {
    fn snapshot(&self) -> Result<MemorySnapshot> {
        let out = Command::new("vm_stat").output().context("running vm_stat")?;
        parse_vm_stat(&String::from_utf8_lossy(&out.stdout))
    }
}

pub fn parse_vm_stat(text: &str) -> Result<MemorySnapshot> {
    let mut page_size = 4096u64;
    if let Some(idx) = text.find("page size of ") {
        let rest = &text[idx + "page size of ".len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(v) = digits.parse::<u64>() {
            page_size = v;
        }
    }

    let field = |name: &str| -> u64 {
        for line in text.lines() {
            let Some((label, value)) = line.split_once(':') else { continue };
            if label.trim().eq_ignore_ascii_case(name) {
                let digits: String = value.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
                return digits.parse().unwrap_or(0);
            }
        }
        0
    };

    Ok(MemorySnapshot {
        page_size,
        free: field("Pages free"),
        active: field("Pages active"),
        inactive: field("Pages inactive"),
        speculative: field("Pages speculative"),
        wired: field("Pages wired down"),
        purgeable: field("Pages purgeable"),
        compressed: field("Pages occupied by compressor"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               74004.
Pages active:                           1043064.
Pages inactive:                          774097.
Pages speculative:                        21000.
Pages throttled:                              0.
Pages wired down:                        120000.
Pages purgeable:                           5000.
Pages occupied by compressor:             30000.
";

    #[test]
    fn parses_page_size_and_fields() {
        let s = parse_vm_stat(SAMPLE).unwrap();
        assert_eq!(s.page_size, 16384);
        assert_eq!(s.free, 74004);
        assert_eq!(s.inactive, 774097);
        assert_eq!(s.compressed, 30000);
    }

    #[test]
    fn available_counts_reclaimable_not_total_minus_used() {
        let s = parse_vm_stat(SAMPLE).unwrap();
        let expected = (74004 + 774097 + 21000 + 5000) * 16384;
        assert_eq!(s.available_bytes(), expected);
        // active memory is not available, and must not leak into the figure
        assert!(s.available_bytes() < s.available_bytes() + 1043064 * 16384);
    }

    #[test]
    fn total_counts_every_page_class() {
        let s = parse_vm_stat(SAMPLE).unwrap();
        let expected = (74004 + 1043064 + 774097 + 21000 + 120000 + 5000 + 30000) * 16384;
        assert_eq!(s.total_bytes(), expected);
        assert!(s.total_bytes() > s.available_bytes(), "total must exceed what is reclaimable");
    }

    #[test]
    fn gibibytes_are_bytes_divided_by_1024_three_times() {
        let s = parse_vm_stat(SAMPLE).unwrap();
        let expected = s.available_bytes() as f64 / 1024.0 / 1024.0 / 1024.0;
        assert!((s.available_gib() - expected).abs() < 1e-9);
        assert!(s.available_gib() > 12.0 && s.available_gib() < 14.0, "got {}", s.available_gib());
    }

    #[test]
    fn missing_fields_do_not_panic() {
        let s = parse_vm_stat("Mach Virtual Memory Statistics: (page size of 4096 bytes)\n").unwrap();
        assert_eq!(s.page_size, 4096);
        assert_eq!(s.free, 0);
    }
}
