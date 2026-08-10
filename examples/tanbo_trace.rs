// Random Tanbo game trace, one move per step.
// Usage: cargo run --release --example tanbo_trace [-- 9|11|13|19]
use mcts::game::Game;
use mcts::games::tanbo::*;
use rand::Rng;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(9);

    match n {
        9 => trace::<9>(),
        11 => trace::<11>(),
        13 => trace::<13>(),
        19 => trace::<19>(),
        other => {
            eprintln!("Unsupported board size {other}, using 9");
            trace::<9>();
        }
    }
}

fn trace<const N: usize>() {
    let mut rng = rand::thread_rng();
    let mut state = State::<N>::default();
    let mut step = 0usize;

    println!("=== Tanbo {N}×{N} (dense 2025 init) ===");
    println!("[step {step}]");
    println!("{state}");
    println!();

    loop {
        if Tanbo::<N>::is_terminal(&state) {
            break;
        }

        let mut actions = Vec::new();
        Tanbo::<N>::generate_actions(&state, &mut actions);
        let idx = rng.gen_range(0..actions.len());
        let action = actions[idx].clone();
        let notation = Tanbo::<N>::notation(&state, &action);

        state = Tanbo::<N>::apply(state, &action);
        step += 1;

        println!("[step {step}] {notation}");
        println!("{state}");
        println!();

        if step >= 200 {
            println!("(cut off at 200 steps)");
            break;
        }
    }

    match Tanbo::<N>::winner(&state) {
        Some(p) => println!("Winner: {p:?}"),
        None => println!("Draw / no winner"),
    }
    println!("Total steps: {step}");
}