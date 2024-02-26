use std::fmt;

pub trait RectangularBoard {
    const NUM_DISPLAY_ROWS: usize;
    const NUM_DISPLAY_COLS: usize;

    fn rank_labels() -> String {
        "ABCDEFGHIJKLMNOPQRSTUVWX".into()
    }

    fn display_char_at(&self, row: usize, col: usize) -> char;
}

// Define a newtype wrapper for implementing Display
pub struct RectangularBoardDisplay<'a, T>(pub &'a T)
where
    T: RectangularBoard;

impl<T> fmt::Display for RectangularBoardDisplay<'_, T>
where
    T: RectangularBoard,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const FILES: &[u8] = b"ABCDEFGH";
        write!(f, " ")?;
        for c in FILES.iter().take(T::NUM_DISPLAY_COLS) {
            write!(f, " {}", *c as char)?;
        }
        writeln!(f)?;
        for row in (0..T::NUM_DISPLAY_ROWS).rev() {
            write!(f, "{}", row + 1)?;
            for col in 0..T::NUM_DISPLAY_COLS {
                write!(f, " {}", self.0.display_char_at(row, col))?;
            }
            write!(f, " {}", row + 1)?;
            writeln!(f)?;
        }
        write!(f, " ")?;
        for c in FILES.iter().take(T::NUM_DISPLAY_COLS) {
            write!(f, " {}", *c as char)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

#[cfg(test)]
mod example {
    use super::*;

    pub struct ExampleBoard {
        pub data: [[char; 7]; 7],
    }

    impl RectangularBoard for ExampleBoard {
        const NUM_DISPLAY_ROWS: usize = 7;
        const NUM_DISPLAY_COLS: usize = 7;

        fn display_char_at(&self, row: usize, col: usize) -> char {
            self.data[row][col]
        }
    }

    #[test]
    fn test_example() {
        let example_board = example::ExampleBoard {
            data: [
                ['.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', 'X', '.', '.'],
                ['.', '.', 'O', 'O', 'X', '.', '.'],
                ['.', '.', 'O', 'X', '.', '.', '.'],
                ['.', '.', 'O', 'X', '.', '.', '.'],
                ['.', '.', 'O', 'X', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.'],
            ],
        };

        println!("{}", RectangularBoardDisplay(&example_board));
    }
}
