//! Self-play shards: a small header, then fixed-size little-endian records, one per position.
//! `research/az-train/src/az_train/gonnect_records.py` reads them with numpy.
//!
//! Header: the magic, then `size`, `actions`, `record_bytes` and `mask_words` (`u32` each).
//!
//! Record (`32 * mask_words + 4 + 4 + 4 + 4 * actions` bytes, `mask_words = ceil(size^2 / 64)`):
//! black, white, ko and legal cell masks (`mask_words` `u64`s each, cell `c` is bit `c % 64` of
//! word `c / 64`), the mover's outcome (`f32`, `+1` win, `-1` loss), game index (`u32`), ply
//! (`u16`), flags (`u8`, see `encode::FLAG_*`), one padding byte, then the dense improved-policy
//! target (`f32` per action).

use std::io;
use std::path::Path;

use super::encode::{mask_words, num_actions, Fields, Mask};

const MAGIC: &[u8; 8] = b"GNCSHRD2";
const HEADER_BYTES: usize = 8 + 4 * 4;

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub fields: Fields,
    pub value: f32,
    pub game: u32,
    pub ply: u16,
    pub policy: Vec<f32>,
}

pub fn record_bytes(size: usize) -> usize {
    32 * mask_words(size) + 4 + 4 + 4 + 4 * num_actions(size)
}

impl Record {
    fn write_to(&self, out: &mut Vec<u8>, words: usize) {
        for mask in [
            self.fields.black,
            self.fields.white,
            self.fields.ko,
            self.fields.legal,
        ] {
            for word in &mask.0[..words] {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.value.to_le_bytes());
        out.extend_from_slice(&self.game.to_le_bytes());
        out.extend_from_slice(&self.ply.to_le_bytes());
        out.extend_from_slice(&[self.fields.flags, 0]);
        for p in &self.policy {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }

    fn read_from(b: &[u8], words: usize) -> Record {
        let masks_end = 32 * words;
        let mask = |k: usize| {
            let mut m = Mask::default();
            for (w, chunk) in m.0[..words]
                .iter_mut()
                .zip(b[k * 8 * words..(k + 1) * 8 * words].chunks_exact(8))
            {
                *w = u64::from_le_bytes(chunk.try_into().unwrap());
            }
            m
        };
        let tail = &b[masks_end..];
        Record {
            fields: Fields {
                black: mask(0),
                white: mask(1),
                ko: mask(2),
                legal: mask(3),
                flags: tail[10],
            },
            value: f32::from_le_bytes(tail[0..4].try_into().unwrap()),
            game: u32::from_le_bytes(tail[4..8].try_into().unwrap()),
            ply: u16::from_le_bytes(tail[8..10].try_into().unwrap()),
            policy: tail[12..]
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
        mask_words(size) as u32,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for r in records {
        assert_eq!(r.policy.len(), num_actions(size));
        r.write_to(&mut out, mask_words(size));
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
        return Err(bad("not a GNCSHRD2 shard"));
    }
    let word =
        |i: usize| u32::from_le_bytes(bytes[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
    let (size, actions, per, words) = (word(0), word(1), word(2), word(3));
    if actions != num_actions(size) || per != record_bytes(size) || words != mask_words(size) {
        return Err(bad("header disagrees with the record layout"));
    }
    let body = &bytes[HEADER_BYTES..];
    if body.len() % per != 0 {
        return Err(bad("truncated record"));
    }
    Ok((
        size,
        body.chunks_exact(per)
            .map(|b| Record::read_from(b, words))
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::encode::{FLAG_BLACK_TO_MOVE, FLAG_SWAP_LEGAL};

    /// A record whose masks use every bit position of every word the size needs.
    fn record(size: usize, i: u32) -> Record {
        let cells = size * size;
        let mut policy = vec![0.0f32; num_actions(size)];
        policy[i as usize % cells] = 0.75;
        policy[cells + 1] = 0.25;
        let mut black = Mask::default();
        let mut legal = Mask::default();
        for c in 0..cells {
            if (c + i as usize).is_multiple_of(3) {
                black.set(c);
            }
            if c.is_multiple_of(2) {
                legal.set(c);
            }
        }
        let mut ko = Mask::default();
        ko.set(cells - 1);
        Record {
            fields: Fields {
                black,
                white: Mask::default(),
                ko,
                legal,
                flags: FLAG_BLACK_TO_MOVE | FLAG_SWAP_LEGAL,
            },
            value: if i.is_multiple_of(2) { 1.0 } else { -1.0 },
            game: i * 3,
            ply: 250 + i as u16 * 7,
            policy,
        }
    }

    #[test]
    fn shard_round_trips_at_every_size_with_the_documented_record_size() {
        for (size, words, bytes) in [(5, 1, 32 + 12 + 108), (7, 1, 248), (9, 2, 64 + 12 + 332), (19, 6, 192 + 12 + 1452)] {
            assert_eq!(mask_words(size), words);
            assert_eq!(record_bytes(size), bytes, "size {size}");
            let records: Vec<Record> = (0..5).map(|i| record(size, i)).collect();
            let path = std::env::temp_dir()
                .join(format!("gonnect-shard-{size}-{}.bin", std::process::id()));
            write_shard(&path, size, &records).unwrap();
            assert_eq!(
                std::fs::metadata(&path).unwrap().len() as usize,
                HEADER_BYTES + 5 * bytes
            );
            let (read_size, back) = read_shard(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(read_size, size);
            assert_eq!(back, records, "size {size}");
            assert!(back.iter().any(|r| r.ply > 255), "the ply is wider than a byte");
        }
    }

    #[test]
    fn a_truncated_or_foreign_shard_is_rejected() {
        let path = std::env::temp_dir().join(format!("gonnect-shard-bad-{}.bin", std::process::id()));
        std::fs::write(&path, b"GNCSHRD1 old format").unwrap();
        assert!(read_shard(&path).is_err());
        let records = vec![record(9, 0)];
        write_shard(&path, 9, &records).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.pop();
        std::fs::write(&path, bytes).unwrap();
        assert!(read_shard(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
