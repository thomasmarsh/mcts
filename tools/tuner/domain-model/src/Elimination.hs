module Elimination where

-- | An active elimination action: prune a candidate or audit-continue it.
data EliminationAction = Prune | AuditContinue
  deriving (Eq, Show)

-- | Paired-bootstrap elimination margin: the confidence shortfall at the cut.
data PairedProbabilityMargin = PairedProbabilityMargin
  { ppmEliminationProbabilityThreshold :: Double
  , ppmFavorableProbability            :: Double
  , ppmThresholdMinusProbability       :: Double
  }
  deriving (Eq, Show)

-- | Rank-cut elimination margin: how far below the survivor cutoff a candidate
-- fell. ``shrmRanksBelowCutoff`` is a positive one-based distance; ``shrmSparedCount``
-- is the number of near-tie candidates carried past the cut on the same look.
data SuccessiveHalvingRankMargin = SuccessiveHalvingRankMargin
  { shrmRank                :: Int
  , shrmTargetSurvivorCount :: Int
  , shrmRanksBelowCutoff    :: Int
  , shrmSparedCount         :: Int
  }
  deriving (Eq, Show)

-- | The typed elimination decision margin carried on an enforced action.
data EliminationDecisionMargin
  = PairedProbabilityMarginElim PairedProbabilityMargin
  | SuccessiveHalvingRankMarginElim SuccessiveHalvingRankMargin
  deriving (Eq, Show)

-- | A single candidate elimination action with its decision margin.
data CandidateEliminationAction = CandidateEliminationAction
  { ceaCandidateId :: String
  , ceaAction      :: EliminationAction
  , ceaMargin      :: EliminationDecisionMargin
  }
  deriving (Eq, Show)

-- | An apply-elimination allocation: actions for one cohort at one prefix.
data ApplyElimination = ApplyElimination
  { aeCohortIndex :: Int
  , aePrefixId    :: String
  , aeActions     :: [CandidateEliminationAction]
  }
  deriving (Eq, Show)

-- | A suspension of active elimination after a boundary reversal.
data SuspendActiveElimination = SuspendActiveElimination
  { saeAfterCohortIndex      :: Int
  , saeTriggeringCandidateIds :: [String]
  , saeTriggeringPrefixIds    :: [String]
  , saeSafetyRuleVersion      :: String
  }
  deriving (Eq, Show)

-- | An audited boundary reversal: a pruned candidate that reached the
-- boundary after audit continuation.
data AuditedBoundaryReversal = AuditedBoundaryReversal
  { abrCohortIndex                     :: Int
  , abrCandidateId                     :: String
  , abrPrefixId                        :: String
  , abrBoundaryCandidateId             :: String
  , abrMaximumPrefixPairedMeanDifference :: Double
  }
  deriving (Eq, Show)
