module Shadow where

import Evaluation (TaskPrefix)

-- | The frozen set of shadow-race method versions.
data ShadowMethodVersion
  = StratifiedPairedBootstrapV1
  | StratifiedPairedBootstrapAllStrataV2
  | SuccessiveHalvingCommonPrefixEta2V1
  | SuccessiveHalvingSpareNearTieV1
  deriving (Eq, Show)

-- | Disposition of a candidate in a shadow race decision.
data ShadowDisposition = Continue | Eliminate | Protected
  deriving (Eq, Show)

-- | Evidence for a paired bootstrap shadow decision.
data PairedBootstrapEvidence = PairedBootstrapEvidence
  { pbeFavorableResamples :: Int
  , pbeTotalResamples     :: Int
  }
  deriving (Eq, Show)

-- | Evidence for a successive halving shadow decision.
data SuccessiveHalvingEvidence = SuccessiveHalvingEvidence
  { sheRank              :: Maybe Int
  , shePriorSurvivorCount :: Int
  , sheTargetSurvivorCount :: Int
  , sheNewlyEliminated    :: Bool
  }
  deriving (Eq, Show)

-- | Union of shadow decision evidence.
data ShadowDecisionEvidence
  = PairedBootstrapEv PairedBootstrapEvidence
  | SuccessiveHalvingEv SuccessiveHalvingEvidence
  deriving (Eq, Show)

-- | A single candidate's shadow race decision.
data ShadowCandidateDecision = ShadowCandidateDecision
  { scdCandidateId :: String
  , scdDisposition :: ShadowDisposition
  , scdEvidence    :: ShadowDecisionEvidence
  }
  deriving (Eq, Show)

-- | The kind of shadow policy.
data ShadowPolicyKind = PairedBootstrap | SuccessiveHalving
  deriving (Eq, Show)

-- | A complete shadow race decision: evidence-only elimination decisions for one
-- cohort at one tuning prefix.
data ShadowRaceDecision = ShadowRaceDecision
  { srdCohortIndex         :: Int
  , srdPrefixId            :: String
  , srdObservationIds      :: [String]
  , srdBoundaryCandidateId :: String
  , srdDecisions           :: [ShadowCandidateDecision]
  , srdPolicyKind          :: ShadowPolicyKind
  , srdPolicyVersion       :: ShadowMethodVersion
  }
  deriving (Eq, Show)

-- | Whether a tuning prefix is eligible for shadow decisions.
shadowPrefixEligible :: [TaskPrefix] -> TaskPrefix -> TaskPrefix -> Int -> Bool
shadowPrefixEligible = undefined

-- | Run a paired bootstrap shadow race.
decidePairedBootstrapShadowRace :: ShadowRaceDecision
decidePairedBootstrapShadowRace = undefined

-- | Run a successive halving shadow race.
decideSuccessiveHalvingShadowRace :: ShadowRaceDecision
decideSuccessiveHalvingShadowRace = undefined

-- | Stratum-level difference evidence for paired bootstrap.
data StratumDifferences = StratumDifferences
  { sdStratumId :: String
  , sdTaskIds   :: [String]
  , sdValues    :: [Double]
  }
  deriving (Eq, Show)

-- | Per-stratum favorable resample counts.
favorableResamplesByStratum :: [StratumDifferences] -> [(String, Int)]
favorableResamplesByStratum = undefined