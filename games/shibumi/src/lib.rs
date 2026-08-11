/// A shibumi
///
/// 11 12 13 14
///               21 22 23
/// 08 09 10 11              26 27
///               18 19 20           28
/// 04 05 06 07              24 25
///               15 16 17
/// 00 01 02 03
///
/// n=16          n=9        n=4     n=1
///
/// col/row/stack indexing also is supported.
pub struct Shibumi(u32);

const WIDTH: usize = 4;
const STACK_LEVELS: usize = 4;

const MASKS: [u32; 3] = [0b110011, 0b11011, 0b1111];

impl Default for Shibumi {
    fn default() -> Self {
        Self::new()
    }
}

impl Shibumi {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn index(col: usize, row: usize, stack: usize) -> usize {
        assert!(stack < STACK_LEVELS);
        let base_size = WIDTH - stack;
        assert!(col < base_size);
        assert!(row < base_size);
        let base_start = (stack * (WIDTH + (WIDTH - 1))) >> 1;
        let row_offset = (WIDTH - 1 - row) * base_size;
        base_start + row_offset + col
    }

    pub fn set_bit(&mut self, x: usize, y: usize, stack: usize) {
        let pos = Self::index(x, y, stack);
        self.0 |= 1 << pos;
    }

    pub fn clear_bit(&mut self, x: usize, y: usize, stack: usize) {
        let pos = Self::index(x, y, stack);
        self.0 &= !(1 << pos);
    }

    // Check if a bit at a given position and stack level is set
    pub fn is_set(&self, x: usize, y: usize, stack: usize) -> bool {
        let pos = Self::index(x, y, stack);
        (self.0 & (1 << pos)) != 0
    }

    pub fn is_valid_position(&self, col: usize, row: usize, stack: usize) -> bool {
        if col >= WIDTH - stack || row >= WIDTH - stack || stack >= STACK_LEVELS {
            return false;
        }
        if self.is_set(col, row, stack) {
            return false;
        }
        if stack == 0 {
            true
        } else {
            let mask = MASKS[stack - 1];
            let pos = Self::index(col, row, stack);
            let support_mask =
                mask << pos | mask << (pos + 1) | mask << (pos + WIDTH) | mask << (pos + WIDTH + 1);
            self.0 & support_mask == support_mask
        }
    }

    pub fn print(&self, stack: usize) {
        let base_width = WIDTH - stack;
        for y in 0..base_width {
            for x in 0..base_width {
                if self.is_set(x, y, stack) {
                    print!(" X");
                } else {
                    print!(" .");
                }
            }
            println!();
        }
    }

    pub fn print_top_down_view(&self) {
        // todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shibumi() {
        let mut bitboard = Shibumi::new();
        bitboard.set_bit(1, 1, 1);
        bitboard.set_bit(2, 2, 1);
        bitboard.set_bit(1, 1, 2);
        bitboard.print(3);
        println!("\n---");
        bitboard.print(2);
        println!("\n---");
        bitboard.print(1);
        println!("\n---");
        bitboard.print(0);
        println!();
        bitboard.print_top_down_view();
    }
}
