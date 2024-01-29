minimax-rs:
- implements MCTS-Solver which has limited applicability
- MCTS-Solver assigns +/-INF scores and so breaks UCT unless multiple values maintained

Changes from minimax-rs
- rename DumbRollout to NaiveRollout
- reduce `Game::M: Copy` constraint to `Clone`
