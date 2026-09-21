//! Self-play shards: a small header, then fixed-size little-endian records, one per position.
//! `research/az-train/src/az_train/gonnect_records.py` reads them with numpy.
//!
//! Record (`32 + 4 + 4 + 4 + 4 * actions` bytes): black, white, ko and legal cell masks (`u64`
//! each), the mover's outcome (`f32`, `+1` win, `-1` loss), game index (`u32`), ply (`u8`), flags
//! (`u8`, see `encode::FLAG_*`), two padding bytes, then the dense improved-policy target
//! (`f32` per action).

use std::io;
use std::path::Path;

use super::encode::{num_actions, Fields};

const MAGIC: &[u8; 8] = b"GNCSHRD1";
const HEADER_BYTES: usize = 8 + 3 * 4;

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub fields: Fields,
    pub value: f32,
    pub game: u32,
    pub ply: u8,
    pub policy: Vec<f32>,
}

pub fn record_bytes(size: usize) -> usize {
    32 + 4 + 4 + 4 + 4 * num_actions(size)
}

impl Record {
    fn write_to(&self, out: &mut Vec<u8>) {
        for mask in [
            self.fields.black,
            self.fields.white,
            self.fields.ko,
            self.fields.legal,
        ] {
            out.extend_from_slice(&mask.to_le_bytes());
        }
        out.extend_from_slice(&self.value.to_le_bytes());
        out.extend_from_slice(&self.game.to_le_bytes());
        out.extend_from_slice(&[self.ply, self.fields.flags, 0, 0]);
        for p in &self.policy {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }

    fn read_from(b: &[u8]) -> Record {
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let f32_at = |i: usize| f32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        Record {
            fields: Fields {
                black: u64_at(0),
                white: u64_at(8),
                ko: u64_at(16),
                legal: u64_at(24),
                flags: b[41],
            },
            value: f32_at(32),
            game: u32::from_le_bytes(b[36..40].try_into().unwrap()),
            ply: b[40],
            policy: b[44..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        }
    }
}

/// Write a whole shard atomically (temp file, then rename), so a killed run never leaves a
/// half-written shard behind under the final name.
pub fn write_shard(path: &Path, size: usize, records: &[Record]) -> io::Result<()> {
    let mut out = Vec::with_capacity(HEADER_BYTES + records.len() * record_bytes(size));
    out.extend_from_slice(MAGIC);
    for v in [
        size as u32,
        num_actions(size) as u32,
        record_bytes(size) as u32,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for r in records {
        assert_eq!(r.policy.len(), num_actions(size));
        r.write_to(&mut out);
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, out)?;
    std::fs::rename(tmp, path)
}

pub fn read_shard(path: &Path) -> io::Result<(usize, Vec<Record>)> {
    let bytes = std::fs::read(path)?;
    let bad = |m: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {m}", path.display()),
        )
    };
    if bytes.len() < HEADER_BYTES || &bytes[..8] != MAGIC {
        return Err(bad("not a GNCSHRD1 shard"));
    }
    let word =
        |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
    let (size, actions, per) = (word(0), word(1), word(2));
    if actions != num_actions(size) || per != record_bytes(size) {
        return Err(bad("header disagrees with the record layout"));
    }
    let body = &bytes[HEADER_BYTES..];
    if body.len() % per != 0 {
        return Err(bad("truncated record"));
    }
    Ok((
        size,
        body.chunks_exact(per).map(Record::read_from).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::encode::{FLAG_BLACK_TO_MOVE, FLAG_SWAP_LEGAL};

    fn record(i: u32) -> Record {
        let mut policy = vec![0.0f32; 51];
        policy[(i % 49) as usize] = 0.75;
        policy[50] = 0.25;
        Record {
            fields: Fields {
                black: 0x1_0000_0000 + i as u64,
                white: 7 << 20,
                ko: 1 << 3,
                legal: (1 << 49) - 1,
                flags: FLAG_BLACK_TO_MOVE | FLAG_SWAP_LEGAL,
            },
            value: if i.is_multiple_of(2) { 1.0 } else { -1.0 },
            game: i * 3,
            ply: (i % 60) as u8,
            policy,
        }
    }

    #[test]
    fn shard_round_trips_and_has_the_documented_record_size() {
        assert_eq!(record_bytes(7), 248);
        let records: Vec<Record> = (0..5).map(record).collect();
        let path = std::env::temp_dir().join(format!("gonnect-shard-{}.bin", std::process::id()));
        write_shard(&path, 7, &records).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len() as usize,
            HEADER_BYTES + 5 * 248
        );
        let (size, back) = read_shard(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(size, 7);
        assert_eq!(back, records);
    }
}
