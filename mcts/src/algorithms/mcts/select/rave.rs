use super::super::config::BackpropFlags;
use super::super::config::GRAVE;
use super::super::index::Id;
use super::super::node::real_action;
use super::super::node::ChildArray;
use super::super::select::SelectContext;
use super::super::select::SelectPolicy;
use super::variance::ucb1_tuned;
use crate::game::Game;

// Ameneyro, F.V., Galvan, E., Morales, A.F.K., 2020. Playing Carcassonne with
// Monte Carlo Tree Search.
//
// Cazenave, T., 2015. Generalized Rapid Action Value Estimation, in:
// Proceedings of the Twenty-Fourth International Joint Conference on Artificial
// Intelligence. Presented at the International Joint Conference on Artificial
// Intelligence, Buenos Aires, Argentina.
//
// Gelly, S., Silver, D., 2011. Monte-Carlo tree search and rapid action value
// estimation in computer Go. Artificial Intelligence 175, 1856–1875. https://
// doi.org/10.1016/j.artint.2011.03.007
//
// Rimmel, A., Teytaud, F., Teytaud, O., 2011. Biasing Monte-Carlo
// Simulations through RAVE Values, in: Van Den Herik, H.J., Iida, H.,
// Plaat, A. (Eds.), Computers and Games, Lecture Notes in Computer Science.
// Springer Berlin Heidelberg, Berlin, Heidelberg, pp. 59–68. https://
// doi.org/10.1007/978-3-642-17928-0_6
//
// Sironi, C.F., Winands, M.H.M., 2016. Comparison of rapid action value
// estimation variants for general game playing, in: 2016 IEEE Conference
// on Computational Intelligence and Games (CIG). Presented at the 2016 IEEE
// Conference on Computational Intelligence and Games (CIG), IEEE, Santorini,
// Greece, pp. 1–8. https://doi.org/10.1109/CIG.2016.7860429
//
// Sironi, C.F., Winands, M.H.M., 2018. On-Line Parameter Tuning for Monte-Carlo
// Tree Search in General Game Playing, in: Cazenave, T., Winands, M.H.M.,
// Saffidine, A. (Eds.), Computer Games, Communications in Computer and
// Information Science. Springer International Publishing, Cham, pp. 75–95.
// https://doi.org/10.1007/978-3-319-75931-9_6

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RaveSchedule {
    // HandSelected comes from CadiaPlayer
    // MinMSE and CadiaPlayare are both described in Gelly, Silver 2011
    // k=1000 for go
    HandSelected { k: u32 },
    // TODO: default bias
    MinMSE { bias: f64 },
    // Traditional Rave. I have seen recommendations to start tuning with rave = 700
    Threshold { rave: u32 },
}

impl Default for RaveSchedule {
    fn default() -> Self {
        RaveSchedule::HandSelected { k: 1000 }
    }
}

impl RaveSchedule {
    pub(crate) fn beta(&self, n: u32, amaf_n: u32) -> f64 {
        let n = n as f64;
        let amaf_n = amaf_n as f64;
        match self {
            RaveSchedule::HandSelected { k } => {
                let k = *k as f64;
                (k / (3. * n + k)).sqrt()
            }
            RaveSchedule::MinMSE { bias } => amaf_n / (n + amaf_n + 4. * n * amaf_n * bias * bias),

            RaveSchedule::Threshold { rave } => 0f64.max(*rave as f64 - n) / *rave as f64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RaveUcb {
    None,
    Ucb1 { exploration_constant: f64 },
    Ucb1Tuned { exploration_constant: f64 },
}
impl Default for RaveUcb {
    fn default() -> Self {
        Self::Ucb1 {
            exploration_constant: 2f64.sqrt(),
        }
    }
}

impl RaveUcb {
    fn score(&self, parent_log: f64, n: u32, sum_squared_score: f64, exploit: f64) -> f64 {
        match self {
            RaveUcb::None => 0.,
            RaveUcb::Ucb1 {
                exploration_constant,
            } => exploration_constant * (parent_log / n as f64).sqrt(),
            RaveUcb::Ucb1Tuned {
                exploration_constant,
            } => {
                let sample_variance = 0f64.max(sum_squared_score / n as f64 - exploit * exploit);
                let visits_fraction = parent_log / n as f64;
                ucb1_tuned(
                    *exploration_constant,
                    0., // RAVE provides the exploitation term.
                    sample_variance,
                    visits_fraction,
                )
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Rave {
    pub threshold: u32, // 0 == RAVE, inf = HRAVE, else GRAVE
    pub schedule: RaveSchedule,
    pub ucb: RaveUcb,
}

impl Default for Rave {
    fn default() -> Self {
        Self {
            threshold: 700,
            schedule: RaveSchedule::default(),
            ucb: RaveUcb::default(),
        }
    }
}

impl Rave {
    pub fn new(threshold: u32, schedule: RaveSchedule, ucb: RaveUcb) -> Self {
        Self {
            threshold,
            schedule,
            ucb,
        }
    }

    pub fn threshold(mut self, threshold: u32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn schedule(mut self, schedule: RaveSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn ucb(mut self, ucb: RaveUcb) -> Self {
        self.ucb = ucb;
        self
    }
}

impl Rave {
    fn get_ref<G: Game>(&self, ctx: &SelectContext<'_, G>, node_id: Id, idx: usize) -> Id {
        let mut stack = ctx.stack.clone();
        stack.push(node_id, idx);
        let rev_pairs = stack.reverse_pairs();

        if ctx.index.get(node_id).is_root() {
            return node_id;
        }

        // TODO: we can push this down during select descent rather than walking back up.
        for ((parent_id, _), (child_id, child_idx)) in rev_pairs {
            if stack
                .get_stats(
                    ctx.index,
                    ctx.root_stats,
                    ctx.graph_stats,
                    *parent_id,
                    *child_id,
                    *child_idx,
                )
                .total_visits()
                >= self.threshold
            {
                return *child_id;
            }
        }
        stack.root()
    }

    #[inline(always)]
    fn amaf_score(n: u32, q: f64) -> f64 {
        if n == 0 {
            0.
        } else {
            q / n as f64
        }
    }
}

impl<G: Game> SelectPolicy<G> for Rave {
    fn label(&self) -> String {
        "rave".into()
    }

    type Score = f64;
    type Aux = f64;

    #[inline(always)]
    fn setup(&mut self, ctx: &SelectContext<'_, G>) -> f64 {
        ((ctx.current_stats().num_visits() as f64).max(1.)).ln()
    }

    #[inline(always)]
    fn score_child(
        &self,
        ctx: &SelectContext<'_, G>,
        child_id: Id,
        children: &ChildArray<G::A>,
        idx: usize,
        parent_log: f64,
    ) -> f64 {
        let ref_id = self.get_ref(ctx, child_id, idx);
        let hash = ctx.index.get(ref_id).hash;
        let action = real_action::<G>(children, idx, ctx.incoming_sym);
        let grave_stats = ctx
            .grave
            .get(&hash)
            .and_then(|player| player[ctx.player].get(&action).cloned())
            .unwrap_or_default();

        let amaf_n = grave_stats.num_visits;
        let amaf_q = grave_stats.score;

        let snap = ctx.child_snapshot(child_id, children, idx);
        let n = snap.total_visits();
        let exploit = snap.exploitation_score();
        let explore = self
            .ucb
            .score(parent_log, n, snap.sum_squared_score, exploit);

        let b = self.schedule.beta(n, amaf_n);
        let mean_score = snap.expected_score();
        let amaf = Self::amaf_score(amaf_n, amaf_q);

        (1. - b) * mean_score + b * amaf + explore
    }

    #[inline(always)]
    fn unvisited_value(&self, ctx: &SelectContext<'_, G>, _: f64) -> f64 {
        ctx.current_stats()
            .value_estimate_unvisited(ctx.player, ctx.q_init)
    }

    fn backprop_flags(&self) -> BackpropFlags {
        BackpropFlags(GRAVE)
    }
}
