use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

use crate::safety::Severity;

/// One frame of simulation state — everything needed for offline replay/analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimFrame {
    pub tick: u64,
    pub timestamp: f64,
    /// Drone A state: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
    pub state_a: [f64; 13],
    /// Drone B state
    pub state_b: [f64; 13],
    /// Motor commands A: [f1,f2,f3,f4]
    pub motors_a: [f64; 4],
    /// Motor commands B
    pub motors_b: [f64; 4],
    /// Wind force on A
    pub wind_a: [f64; 3],
    /// Wind force on B
    pub wind_b: [f64; 3],
    /// Safety severity for A (0=Ok, 1=Warning, 2=Critical)
    pub safety_a: u8,
    /// Safety severity for B
    pub safety_b: u8,
    /// Lock-on progress A→B
    pub lock_progress_a: f64,
    /// Lock-on progress B→A
    pub lock_progress_b: f64,
    /// A sees B
    pub a_sees_b: bool,
    /// B sees A
    pub b_sees_a: bool,
}

impl SimFrame {
    pub fn safety_severity_a(&self) -> Severity {
        match self.safety_a {
            0 => Severity::Ok,
            1 => Severity::Warning,
            _ => Severity::Critical,
        }
    }

    pub fn safety_severity_b(&self) -> Severity {
        match self.safety_b {
            0 => Severity::Ok,
            1 => Severity::Warning,
            _ => Severity::Critical,
        }
    }
}

/// Records simulation frames for offline analysis and replay.
#[derive(Debug, Clone, Default)]
pub struct SimRecorder {
    frames: Vec<SimFrame>,
    enabled: bool,
}

impl SimRecorder {
    pub fn new(enabled: bool) -> Self {
        Self {
            frames: Vec::new(),
            enabled,
        }
    }

    /// Record a frame. No-op if disabled.
    pub fn record(&mut self, frame: SimFrame) {
        if self.enabled {
            self.frames.push(frame);
        }
    }

    /// Number of recorded frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Get a frame by index.
    pub fn frame(&self, index: usize) -> Option<&SimFrame> {
        self.frames.get(index)
    }

    /// Iterator over all frames.
    pub fn frames(&self) -> &[SimFrame] {
        &self.frames
    }

    /// Clear all recorded frames.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Save to file using bincode.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let bytes = bincode::serialize(&self.frames).map_err(io::Error::other)?;
        fs::write(path, bytes)
    }

    /// Load from bincode file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let frames: Vec<SimFrame> = bincode::deserialize(&bytes).map_err(io::Error::other)?;
        Ok(Self {
            enabled: true,
            frames,
        })
    }
}

/// Helper: convert Severity to u8 for serialization.
pub fn severity_to_u8(s: Severity) -> u8 {
    match s {
        Severity::Ok => 0,
        Severity::Warning => 1,
        Severity::Critical => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame(tick: u64) -> SimFrame {
        SimFrame {
            tick,
            timestamp: tick as f64 * 0.01,
            state_a: [
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            state_b: [
                5.0, 5.0, 1.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            motors_a: [0.066, 0.066, 0.066, 0.066],
            motors_b: [0.066, 0.066, 0.066, 0.066],
            wind_a: [0.0, 0.0, 0.0],
            wind_b: [0.0, 0.0, 0.0],
            safety_a: 0,
            safety_b: 0,
            lock_progress_a: 0.0,
            lock_progress_b: 0.0,
            a_sees_b: true,
            b_sees_a: false,
        }
    }

    #[test]
    fn test_record_and_retrieve() {
        let mut rec = SimRecorder::new(true);
        rec.record(sample_frame(0));
        rec.record(sample_frame(1));
        rec.record(sample_frame(2));

        assert_eq!(rec.len(), 3);
        assert_eq!(rec.frame(1).unwrap().tick, 1);
    }

    #[test]
    fn test_disabled_recorder_skips() {
        let mut rec = SimRecorder::new(false);
        rec.record(sample_frame(0));
        assert_eq!(rec.len(), 0);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut rec = SimRecorder::new(true);
        for i in 0..100 {
            rec.record(sample_frame(i));
        }

        let dir = std::env::temp_dir();
        let path = dir.join("aces_test_recording.bin");
        rec.save(&path).unwrap();

        let loaded = SimRecorder::load(&path).unwrap();
        assert_eq!(loaded.len(), 100);
        assert_eq!(loaded.frame(42).unwrap().tick, 42);
        assert!(loaded.frame(42).unwrap().a_sees_b);

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_severity_roundtrip() {
        assert_eq!(severity_to_u8(Severity::Ok), 0);
        assert_eq!(severity_to_u8(Severity::Warning), 1);
        assert_eq!(severity_to_u8(Severity::Critical), 2);

        let frame = SimFrame {
            safety_a: severity_to_u8(Severity::Warning),
            safety_b: severity_to_u8(Severity::Critical),
            ..sample_frame(0)
        };
        assert_eq!(frame.safety_severity_a(), Severity::Warning);
        assert_eq!(frame.safety_severity_b(), Severity::Critical);
    }
}
