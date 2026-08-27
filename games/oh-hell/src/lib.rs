//! Oh Hell (a.k.a. Blackout / Nomination Whist): a single dealt hand of a
//! trick-taking card game with a bidding phase, for `P` players holding `K`
//! cards each from a standard 52-card deck.
//!
//! Dealing: `State::new` shuffles the deck, deals `K` cards to each of the
//! `P` seats, then turns up the next card to set the trump suit for the
//! hand -- that card itself is set aside and never dealt to anyone.
//!
//! Bidding: seats bid in order (seat 0 first, seat `P - 1` last) how many
//! tricks they predict they'll win, `0..=K`. The last bidder is
//! additionally forbidden from naming the one value that would make every
//! bid this hand sum to exactly `K` -- the traditional "screw the dealer"
//! rule, which guarantees at least one seat misses their bid.
//!
//! Trick play: the winner of the previous trick leads the next one (seat 0
//! leads the first trick, since bidding and the opening lead both start to
//! the dealer's left); a seat that holds a card of the led suit must play
//! one, otherwise any card (including trump) is legal. The trick is won by
//! the highest trump played, or if none was, the highest card of the led
//! suit.
//!
//! Scoring, once every card has been played: a seat that won exactly as
//! many tricks as it bid scores `10 + tricks_won`; every other seat scores
//! `0`. [`Game::winner`] is whichever seat has the strict-highest score;
//! a tie is a draw. This crate models one dealt hand rather than a full
//! match played across a ramp of hand sizes -- the bidding, suit-following,
//! and hidden-hand structure a hand exercises doesn't depend on match-level
//! score bookkeeping across multiple hands.
//!
//! Hidden information: only a player's own remaining hand is visible to
//! them; [`OhHell::determinize`] resamples every other seat's hand from the
//! pool of cards the mover hasn't personally seen (not on the board... not
//! played, not in the mover's own hand, and not the turned-up trump card).
//! A seat that fails to follow a led suit reveals, from that point in the
//! hand onward, that it holds none of that suit -- [`State::known_void`]
//! tracks this per seat so a redeal never guesses that seat back into a
//! suit it's already been shown not to hold.

use mcts::game::{Game, PlayerIndex};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use std::fmt;

pub const NUM_SUITS: usize = 4;
pub const NUM_RANKS: usize = 13;
pub const DECK_SIZE: usize = NUM_SUITS * NUM_RANKS;

const DEFAULT_SEED: u64 = 0x0BEE_11CA_5D0E_A177;

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize)]
#[repr(u8)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

impl Suit {
    pub const ALL: [Suit; NUM_SUITS] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    fn char(self) -> char {
        match self {
            Suit::Clubs => 'C',
            Suit::Diamonds => 'D',
            Suit::Hearts => 'H',
            Suit::Spades => 'S',
        }
    }
}

/// A playing card: `rank` is `2..=14` (`11`=Jack, `12`=Queen, `13`=King,
/// `14`=Ace).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize)]
pub struct Card {
    pub suit: Suit,
    pub rank: u8,
}

impl Card {
    /// This card's position (`0..DECK_SIZE`) in suit-major, rank-ascending
    /// order -- the same ordering `full_deck`/`from_index` use, so the two
    /// are inverses.
    #[inline]
    pub fn index(self) -> usize {
        self.suit.index() * NUM_RANKS + (self.rank - 2) as usize
    }

    pub fn from_index(index: usize) -> Card {
        Card {
            suit: Suit::ALL[index / NUM_RANKS],
            rank: (index % NUM_RANKS) as u8 + 2,
        }
    }

    fn rank_char(self) -> char {
        match self.rank {
            2..=9 => (b'0' + self.rank) as char,
            10 => 'T',
            11 => 'J',
            12 => 'Q',
            13 => 'K',
            14 => 'A',
            _ => unreachable!("card rank out of range"),
        }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank_char(), self.suit.char())
    }
}

fn full_deck() -> [Card; DECK_SIZE] {
    std::array::from_fn(Card::from_index)
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
pub struct Player(pub u8);

impl PlayerIndex for Player {
    fn to_index(&self) -> usize {
        self.0 as usize
    }
}

impl Player {
    fn from_index(index: usize) -> Self {
        Player(index as u8)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Bidding,
    Playing,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize)]
pub enum Action {
    Bid(u8),
    Play(Card),
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct State<const P: usize, const K: usize> {
    pub hands: [[Option<Card>; K]; P],
    pub trump: Suit,
    pub bids: [Option<u8>; P],
    pub tricks_won: [u8; P],
    /// This trick's card from each seat, indexed by seat -- not play order.
    pub current_trick: [Option<Card>; P],
    /// Set to the leader's suit as soon as the first card of a trick is
    /// played, and cleared once the trick resolves.
    pub led_suit: Option<Suit>,
    /// `known_void[seat][suit]` is set the moment `seat` is observed
    /// failing to follow `suit`, and never cleared -- a seat can't regain a
    /// suit partway through a hand.
    pub known_void: [[bool; NUM_SUITS]; P],
    /// Every card that has left play: dealt to no one, this hand's trump
    /// turn-up, or already played to a trick.
    pub played: [bool; DECK_SIZE],
    pub current_player: usize,
    pub phase: Phase,
}

impl<const P: usize, const K: usize> Default for State<P, K> {
    fn default() -> Self {
        Self::new(DEFAULT_SEED)
    }
}

impl<const P: usize, const K: usize> State<P, K> {
    pub fn new(seed: u64) -> Self {
        const { assert!(P >= 2, "Oh Hell needs at least two players") };
        const { assert!(K >= 1, "Oh Hell needs at least one card per hand") };
        const {
            assert!(
                P * K < DECK_SIZE,
                "P * K must leave at least one card for the trump turn-up"
            )
        };

        let mut rng = SmallRng::seed_from_u64(seed);
        let mut deck = full_deck();
        deck.shuffle(&mut rng);

        let mut hands = [[None; K]; P];
        let mut next = 0;
        for round in 0..K {
            for hand in hands.iter_mut() {
                hand[round] = Some(deck[next]);
                next += 1;
            }
        }
        for hand in hands.iter_mut() {
            sort_hand(hand);
        }
        let trump = deck[next].suit;

        let mut played = [false; DECK_SIZE];
        played[deck[next].index()] = true;

        State {
            hands,
            trump,
            bids: [None; P],
            tricks_won: [0; P],
            current_trick: [None; P],
            led_suit: None,
            known_void: [[false; NUM_SUITS]; P],
            played,
            current_player: 0,
            phase: Phase::Bidding,
        }
    }

    #[inline]
    fn dealer() -> usize {
        P - 1
    }

    fn hand_size(&self, seat: usize) -> usize {
        self.hands[seat].iter().filter(|c| c.is_some()).count()
    }

    fn hands_all_empty(&self) -> bool {
        self.hands.iter().all(|h| h.iter().all(Option::is_none))
    }

    fn generate_bid_actions(&self, actions: &mut Vec<Action>) {
        let sum_so_far: u32 = self.bids.iter().flatten().map(|&b| b as u32).sum();
        let is_last_bidder = self.current_player == Self::dealer();
        for b in 0..=(K as u8) {
            if is_last_bidder && sum_so_far + b as u32 == K as u32 {
                continue;
            }
            actions.push(Action::Bid(b));
        }
    }

    fn generate_play_actions(&self, actions: &mut Vec<Action>) {
        let hand = &self.hands[self.current_player];
        let following: Vec<Card> = match self.led_suit {
            Some(s) => hand
                .iter()
                .flatten()
                .copied()
                .filter(|c| c.suit == s)
                .collect(),
            None => Vec::new(),
        };
        if following.is_empty() {
            actions.extend(hand.iter().flatten().copied().map(Action::Play));
        } else {
            actions.extend(following.into_iter().map(Action::Play));
        }
    }

    fn apply_bid(&mut self, bid: u8) {
        self.bids[self.current_player] = Some(bid);
        self.current_player = (self.current_player + 1) % P;
        if self.current_player == 0 {
            self.phase = Phase::Playing;
        }
    }

    fn apply_play(&mut self, card: Card) {
        let seat = self.current_player;
        for slot in self.hands[seat].iter_mut() {
            if *slot == Some(card) {
                *slot = None;
                break;
            }
        }
        sort_hand(&mut self.hands[seat]);

        match self.led_suit {
            None => self.led_suit = Some(card.suit),
            Some(led) if card.suit != led => {
                // Legal only because `seat` holds no more of `led` -- reveal
                // that void for the rest of the hand.
                self.known_void[seat][led.index()] = true;
            }
            Some(_) => {}
        }

        self.played[card.index()] = true;
        self.current_trick[seat] = Some(card);
        self.current_player = (self.current_player + 1) % P;

        if self.current_trick.iter().all(Option::is_some) {
            let winner = self.resolve_trick();
            self.tricks_won[winner] += 1;
            self.current_trick = [None; P];
            self.led_suit = None;
            self.current_player = winner;
        }
    }

    /// The seat that wins a full trick: ranks a card by (is it trump, is it
    /// the led suit, its rank), so trump beats led-suit beats anything else,
    /// and within a category the higher rank wins -- independent of which
    /// seat happened to lead or the order cards are compared in.
    fn resolve_trick(&self) -> usize {
        let led = self
            .led_suit
            .expect("a full trick always has a led suit set by its first card");
        let trump = self.trump;
        let key = |c: Card| -> (u8, u8) {
            let category = if c.suit == trump {
                2
            } else if c.suit == led {
                1
            } else {
                0
            };
            (category, c.rank)
        };
        self.current_trick
            .iter()
            .enumerate()
            .map(|(seat, slot)| {
                (
                    seat,
                    key(slot.expect("every seat has played by the time a trick resolves")),
                )
            })
            .max_by_key(|&(_, k)| k)
            .expect("P >= 2, so a trick always has a seat")
            .0
    }

    fn score(&self, seat: usize) -> i32 {
        if self.bids[seat] == Some(self.tricks_won[seat]) {
            10 + self.tricks_won[seat] as i32
        } else {
            0
        }
    }

    fn compute_winner(&self) -> Option<Player> {
        let scores: [i32; P] = std::array::from_fn(|seat| self.score(seat));
        let best = *scores.iter().max().unwrap();
        let mut winner = None;
        for (seat, &s) in scores.iter().enumerate() {
            if s == best {
                if winner.is_some() {
                    return None;
                }
                winner = Some(seat);
            }
        }
        winner.map(Player::from_index)
    }

    /// Resamples every seat but `observer`'s hand from the pool of cards
    /// `observer` hasn't personally seen: not `observer`'s own hand, not
    /// played, and not the trump turn-up. A seat already shown void in a
    /// suit is dealt only from the remaining suits, and seats with more
    /// known-void suits are dealt first, since they have the fewest
    /// eligible cards left to draw from as the pool shrinks.
    fn redeal_hidden_hands(&mut self, observer: usize, rng: &mut SmallRng) {
        let observer_hand = self.hands[observer];
        let mut pool: Vec<Card> = (0..DECK_SIZE)
            .map(Card::from_index)
            .filter(|c| !self.played[c.index()] && !observer_hand.contains(&Some(*c)))
            .collect();
        pool.shuffle(rng);

        let mut order: Vec<usize> = (0..P).filter(|&s| s != observer).collect();
        order
            .sort_by_key(|&s| std::cmp::Reverse(self.known_void[s].iter().filter(|&&v| v).count()));

        for seat in order {
            let need = self.hand_size(seat);
            let void = self.known_void[seat];
            let mut eligible: Vec<usize> = pool
                .iter()
                .enumerate()
                .filter(|(_, c)| !void[c.suit.index()])
                .map(|(i, _)| i)
                .collect();
            eligible.shuffle(rng);
            eligible.truncate(need);
            debug_assert_eq!(
                eligible.len(),
                need,
                "not enough non-void-suit cards left in the pool to redeal seat {seat}"
            );
            // Descending so removing by index doesn't invalidate the
            // indices still queued for removal.
            eligible.sort_unstable_by(|a, b| b.cmp(a));

            let mut new_hand = [None; K];
            for (slot, &i) in new_hand.iter_mut().zip(eligible.iter()) {
                *slot = Some(pool.remove(i));
            }
            sort_hand(&mut new_hand);
            self.hands[seat] = new_hand;
        }
    }

    fn compute_hash(&self) -> u64 {
        let mut h: u64 = 0x9E37_79B1_85EB_CA87;
        let mix = |h: u64, v: u64| -> u64 { (h ^ v).wrapping_mul(0x0100_0000_01B3) };
        for (seat, hand) in self.hands.iter().enumerate() {
            for card in hand.iter().flatten() {
                h = mix(h, 0x1000 | (seat as u64) << 8 | card.index() as u64);
            }
        }
        self.mix_public_hash(h, mix)
    }

    /// Hash of everything every seat can see: bids, tricks won, the cards on
    /// the table this trick, revealed voids, trump, whose turn it is, and
    /// the phase -- but not any hand. Two states differing only in who
    /// holds which unplayed cards are the same information set to an
    /// onlooker and hash equal here, unlike `compute_hash`.
    pub fn public_hash(&self) -> u64 {
        let h: u64 = 0xC0FF_EE00_D15E_ABCD;
        let mix = |h: u64, v: u64| -> u64 { (h ^ v).wrapping_mul(0x0100_0000_01B3) };
        self.mix_public_hash(h, mix)
    }

    fn mix_public_hash(&self, mut h: u64, mix: impl Fn(u64, u64) -> u64) -> u64 {
        for (seat, &bid) in self.bids.iter().enumerate() {
            h = mix(
                h,
                0x2000 | (seat as u64) << 8 | bid.map_or(0xFF, |b| b as u64),
            );
        }
        for (seat, &won) in self.tricks_won.iter().enumerate() {
            h = mix(h, 0x3000 | (seat as u64) << 8 | won as u64);
        }
        for (seat, slot) in self.current_trick.iter().enumerate() {
            if let Some(c) = slot {
                h = mix(h, 0x4000 | (seat as u64) << 8 | c.index() as u64);
            }
        }
        for (seat, voids) in self.known_void.iter().enumerate() {
            for (s, &v) in voids.iter().enumerate() {
                h = mix(h, 0x5000 | (seat as u64) << 8 | (s as u64) << 4 | v as u64);
            }
        }
        h = mix(h, 0x6000 | self.trump.index() as u64);
        h = mix(h, 0x7000 | self.current_player as u64);
        h = mix(h, 0x8000 | matches!(self.phase, Phase::Playing) as u64);
        h
    }
}

/// Sorts a hand so identical states reached in different orders (dealt vs.
/// redealt, played in different sequences) compare and hash the same:
/// held cards first by (suit, rank), `None`s last.
fn sort_hand<const K: usize>(hand: &mut [Option<Card>; K]) {
    hand.sort_unstable_by_key(|slot| match slot {
        Some(c) => (0u8, c.suit.index() as u8, c.rank),
        None => (1u8, 0, 0),
    });
}

impl<const P: usize, const K: usize> fmt::Display for State<P, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "trump: {:?}", self.trump)?;
        for seat in 0..P {
            let hand: Vec<String> = self.hands[seat]
                .iter()
                .flatten()
                .map(|c| c.to_string())
                .collect();
            writeln!(
                f,
                "P{seat} hand: [{}] bid: {:?} tricks: {}",
                hand.join(" "),
                self.bids[seat],
                self.tricks_won[seat]
            )?;
        }
        if self.current_trick.iter().any(Option::is_some) {
            let trick: Vec<String> = self
                .current_trick
                .iter()
                .map(|c| c.map(|c| c.to_string()).unwrap_or_else(|| "-".into()))
                .collect();
            writeln!(f, "trick: [{}]", trick.join(" "))?;
        }
        writeln!(
            f,
            "player {} to move, phase {:?}",
            self.current_player, self.phase
        )
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone)]
pub struct OhHell<const P: usize, const K: usize>;

/// Four players, seven cards each -- a common single-hand Oh Hell size.
pub type OhHellStandard = OhHell<4, 7>;

impl<const P: usize, const K: usize> Game for OhHell<P, K> {
    type S = State<P, K>;
    type A = Action;
    type P = Player;

    fn generate_actions(state: &Self::S, actions: &mut Vec<Self::A>) {
        match state.phase {
            Phase::Bidding => state.generate_bid_actions(actions),
            Phase::Playing => state.generate_play_actions(actions),
        }
    }

    fn apply(mut state: Self::S, action: &Self::A) -> Self::S {
        match action {
            Action::Bid(b) => state.apply_bid(*b),
            Action::Play(c) => state.apply_play(*c),
        }
        state
    }

    fn is_terminal(state: &Self::S) -> bool {
        matches!(state.phase, Phase::Playing) && state.hands_all_empty()
    }

    fn winner(state: &Self::S) -> Option<Self::P> {
        if !Self::is_terminal(state) {
            unreachable!("Oh Hell scoring is only meaningful once the hand is fully played");
        }
        state.compute_winner()
    }

    fn player_to_move(state: &Self::S) -> Self::P {
        Player::from_index(state.current_player)
    }

    fn num_players() -> usize {
        P
    }

    fn has_hidden_information() -> bool {
        true
    }

    /// Resamples every hand but the mover's own, so a search rooted at
    /// their turn only ever sees what they actually know: their own cards
    /// for certain, everyone else's as one guess consistent with what's
    /// been played and which suits they've been shown to be void in.
    fn determinize(mut state: Self::S, rng: &mut SmallRng) -> Self::S {
        let observer = state.current_player;
        state.redeal_hidden_hands(observer, rng);
        state
    }

    fn zobrist_hash(state: &Self::S) -> u64 {
        state.compute_hash()
    }

    fn notation(_state: &Self::S, action: &Self::A) -> String {
        match action {
            Action::Bid(b) => format!("Bid({b})"),
            Action::Play(c) => format!("Play({c})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::strategies::{
        mcts::{render, strategy, SearchConfig, TreeSearch},
        Search,
    };
    use mcts::util::random_play;
    use rand::Rng;
    use rand::SeedableRng;

    type OhHell3x3 = OhHell<3, 3>;
    type OhHell4x7 = OhHellStandard;

    #[test]
    fn random_playouts_terminate() {
        random_play::<OhHell3x3>();
        random_play::<OhHell4x7>();
    }

    #[test]
    fn deal_gives_every_seat_k_cards_and_sets_aside_one_trump_card() {
        let state = State::<4, 7>::new(3);
        for seat in 0..4 {
            assert_eq!(state.hand_size(seat), 7);
        }
        let mut seen = std::collections::HashSet::new();
        for hand in &state.hands {
            for card in hand.iter().flatten() {
                assert!(seen.insert(*card), "duplicate card {card:?} across hands");
            }
        }
        assert_eq!(seen.len(), 28);
        assert_eq!(state.played.iter().filter(|&&p| p).count(), 1);
    }

    #[test]
    fn bidding_forbids_the_dealer_from_making_bids_sum_to_the_hand_size() {
        let base = State::<3, 3>::new(1);

        // Seat 1 (not the dealer, seat 2) may bid anything.
        let not_dealer = State::<3, 3> {
            current_player: 1,
            bids: [Some(1), None, None],
            ..base
        };
        let mut actions = Vec::new();
        not_dealer.generate_bid_actions(&mut actions);
        assert_eq!(actions.len(), 4); // 0..=3

        // Seats 0 and 1 bid a combined 2; seat 2 (the dealer) may not bid 1,
        // since 2 + 1 == K (3).
        let dealer_turn = State::<3, 3> {
            current_player: 2,
            bids: [Some(1), Some(1), None],
            ..base
        };
        let mut actions = Vec::new();
        dealer_turn.generate_bid_actions(&mut actions);
        assert_eq!(
            actions,
            vec![Action::Bid(0), Action::Bid(2), Action::Bid(3)]
        );
    }

    #[test]
    fn must_follow_the_led_suit_when_holding_it() {
        let mut state = State::<3, 3>::new(1);
        state.phase = Phase::Playing;
        state.current_player = 1;
        state.led_suit = Some(Suit::Hearts);
        state.hands[1] = [
            Some(Card {
                suit: Suit::Hearts,
                rank: 5,
            }),
            Some(Card {
                suit: Suit::Clubs,
                rank: 9,
            }),
            None,
        ];

        let mut actions = Vec::new();
        OhHell::<3, 3>::generate_actions(&state, &mut actions);
        assert_eq!(
            actions,
            vec![Action::Play(Card {
                suit: Suit::Hearts,
                rank: 5
            })]
        );
    }

    #[test]
    fn may_play_anything_when_void_in_the_led_suit() {
        let mut state = State::<3, 3>::new(1);
        state.phase = Phase::Playing;
        state.current_player = 1;
        state.led_suit = Some(Suit::Hearts);
        state.hands[1] = [
            Some(Card {
                suit: Suit::Clubs,
                rank: 9,
            }),
            Some(Card {
                suit: Suit::Spades,
                rank: 2,
            }),
            None,
        ];

        let mut actions = Vec::new();
        OhHell::<3, 3>::generate_actions(&state, &mut actions);
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn trick_resolution_prefers_trump_over_the_led_suit_and_rank_within_a_category() {
        let mut state = State::<4, 4>::new(1);
        state.trump = Suit::Spades;
        state.led_suit = Some(Suit::Clubs);
        state.current_trick = [
            Some(Card {
                suit: Suit::Clubs,
                rank: 14,
            }), // Ace of the led suit -- would win if no trump were played.
            Some(Card {
                suit: Suit::Spades,
                rank: 2,
            }), // Lowest trump still beats the led-suit ace.
            Some(Card {
                suit: Suit::Diamonds,
                rank: 13,
            }), // Off-suit, non-trump: can never win.
            Some(Card {
                suit: Suit::Spades,
                rank: 9,
            }), // Higher trump than seat 1's.
        ];
        assert_eq!(state.resolve_trick(), 3);
    }

    #[test]
    fn trick_resolution_falls_back_to_the_highest_led_suit_card_with_no_trump_played() {
        let mut state = State::<3, 3>::new(1);
        state.trump = Suit::Spades;
        state.led_suit = Some(Suit::Hearts);
        state.current_trick = [
            Some(Card {
                suit: Suit::Hearts,
                rank: 7,
            }),
            Some(Card {
                suit: Suit::Clubs,
                rank: 14,
            }),
            Some(Card {
                suit: Suit::Hearts,
                rank: 12,
            }),
        ];
        assert_eq!(state.resolve_trick(), 2);
    }

    #[test]
    fn scoring_rewards_an_exact_bid_and_zeroes_a_missed_one() {
        let mut state = State::<3, 3>::new(1);
        state.bids = [Some(2), Some(1), Some(0)];
        state.tricks_won = [2, 0, 0];
        // Seat 0 hits its bid exactly: 10 + 2 = 12. Seat 1 misses (bid 1,
        // won 0): 0. Seat 2 also hits its bid exactly (bid 0, won 0): 10.
        assert_eq!(state.compute_winner(), Some(Player(0)));
    }

    #[test]
    fn scoring_is_a_draw_when_two_seats_tie_for_the_best_score() {
        let mut state = State::<3, 3>::new(1);
        state.bids = [Some(1), Some(1), Some(0)];
        state.tricks_won = [1, 1, 3];
        assert_eq!(state.compute_winner(), None);
    }

    #[test]
    fn determinize_keeps_the_movers_own_hand_and_all_played_cards_fixed() {
        let mut state = State::<4, 5>::new(9);
        // Simulate a bit of play so hand sizes differ from the initial deal.
        let played_by_1 = state.hands[1][0].take().unwrap();
        state.played[played_by_1.index()] = true;
        let played_by_2 = state.hands[2][0].take().unwrap();
        state.played[played_by_2.index()] = true;
        let mover = 0;
        let mover_hand_before = state.hands[mover];

        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..10 {
            let determinized = OhHell::<4, 5>::determinize(state, &mut rng);
            assert_eq!(determinized.hands[mover], mover_hand_before);
            assert_eq!(determinized.played, state.played);
            for seat in 0..4 {
                assert_eq!(determinized.hand_size(seat), state.hand_size(seat));
            }

            let mut seen = std::collections::HashSet::new();
            for hand in &determinized.hands {
                for card in hand.iter().flatten() {
                    assert!(seen.insert(*card), "determinize dealt a duplicate card");
                }
            }
        }
    }

    #[test]
    fn determinize_never_deals_a_known_void_suit_to_that_seat() {
        let mut state = State::<3, 6>::new(2);
        state.phase = Phase::Playing;
        state.current_player = 0;
        state.known_void[1] = [false, true, false, false]; // seat 1 void in Diamonds

        let mut rng = SmallRng::seed_from_u64(4);
        for _ in 0..30 {
            let determinized = OhHell::<3, 6>::determinize(state, &mut rng);
            for card in determinized.hands[1].iter().flatten() {
                assert_ne!(card.suit, Suit::Diamonds);
            }
        }
    }

    #[test]
    fn random_playout_invariants_hold_across_many_seeds() {
        for seed in 0..25u64 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut state = State::<4, 6>::new(rng.gen());
            let mut actions = Vec::new();
            for _ in 0..500 {
                if OhHell4x6::is_terminal(&state) {
                    break;
                }
                actions.clear();
                OhHell4x6::generate_actions(&state, &mut actions);
                assert!(!actions.is_empty());
                let a = actions[rng.gen_range(0..actions.len())];
                state = OhHell4x6::apply(state, &a);
            }
            assert!(OhHell4x6::is_terminal(&state));
            let total_tricks: u32 = state.tricks_won.iter().map(|&t| t as u32).sum();
            assert_eq!(total_tricks, 6);
        }
    }

    type OhHell4x6 = OhHell<4, 6>;

    impl<const P: usize, const K: usize> render::NodeRender for State<P, K> {}

    #[test]
    fn ismcts_self_play_stays_legal_and_tracks_availability() {
        let mut search: TreeSearch<OhHell3x3, strategy::Ucb1> = TreeSearch::new().config(
            SearchConfig::new()
                .use_ismcts(true)
                .max_iterations(40)
                .seed(11),
        );

        let mut state = State::<3, 3>::new(13);
        for _ in 0..6 {
            if OhHell3x3::is_terminal(&state) {
                break;
            }
            let action = search.choose_action(&state);

            let mut legal = Vec::new();
            OhHell3x3::generate_actions(&state, &mut legal);
            assert!(
                legal.contains(&action),
                "ISMCTS chose an action illegal against the real state"
            );

            let root = search.index.get(search.root_id);
            let children = root.children();
            assert!(children.is_growable());
            let root_idx = (0..children.len())
                .find(|&i| children.action(i) == action)
                .unwrap();
            assert!(children.availability(root_idx) > 0);

            state = OhHell3x3::apply(state, &action);
        }
    }
}
