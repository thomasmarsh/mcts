module Bakeoff where

import Effort (SearchEffort)

-- | Shared frozen inputs for one proposer bake-off run.
data SharedRun = SharedRun
  { srCohortSize               :: Int
  , srFinalists                :: Int
  , srBootstrapCandidates      :: Int
  , srRandomReserveCandidates  :: Int
  , srTuningPairs              :: Int
  , srValidationPairBudget     :: Int
  , srProductionValidationPairs :: Int
  , srTuningEffort             :: SearchEffort
  , srValidationEffort         :: SearchEffort
  , srProductionEffort         :: SearchEffort
  , srExcludedFamilies         :: [String]
  , srEvaluatorWorkers         :: Int
  , srPairTimeoutSeconds       :: Int
  }
  deriving (Eq, Show)

-- | The frozen proposer bake-off decision rule inputs.
data BakeoffDecision = BakeoffDecision
  { bdBaseline                  :: String
  , bdChallenger                :: String
  , bdScorePracticalMargin      :: Double
  , bdRecallNoninferiorityMargin :: Double
  , bdTopSetK                   :: Int
  }
  deriving (Eq, Show)

-- | A proposer bake-off specification.
data BakeoffSpec = BakeoffSpec
  { bsExperimentId      :: String
  , bsGameBinary        :: String  -- filesystem path
  , bsObjectiveFile     :: String  -- filesystem path
  , bsProposalSeeds     :: [Int]
  , bsTaskSeed          :: Int
  , bsTuningPairBudgets :: [Int]
  , bsSharedRun         :: SharedRun
  , bsDecision          :: BakeoffDecision
  }
  deriving (Eq, Show)

-- | Held-out evidence one completed proposer bake-off child contributes.
data ChildFact = ChildFact
  { cfCellId                  :: String
  , cfBudget                  :: Int
  , cfSeed                    :: Int
  , cfPolicy                  :: String
  , cfManifestFingerprint     :: String
  , cfBestCandidateFingerprint :: String
  , cfFinalistFingerprints    :: [String]
  , cfHeldOutMeans            :: [(String, Double)]
  , cfHeldOutBestScore        :: Double
  , cfTuningPairAttempts      :: Int
  , cfTuningPhysicalGames     :: Int
  , cfTuningSearchIterations  :: Int
  , cfTuningWallTimeMs        :: Int
  }
  deriving (Eq, Show)

-- | Shared frozen inputs for one elimination bake-off run.
data EliminationSharedRun = EliminationSharedRun
  { esrProposerPolicy            :: String
  , esrCohortSize                :: Int
  , esrFinalists                 :: Int
  , esrBootstrapCandidates       :: Int
  , esrRandomReserveCandidates   :: Int
  , esrTuningPairs               :: Int
  , esrValidationPairBudget      :: Int
  , esrProductionValidationPairs :: Int
  , esrDiagnosticPairBudget      :: Int
  , esrTuningEffort              :: SearchEffort
  , esrValidationEffort          :: SearchEffort
  , esrProductionEffort          :: SearchEffort
  , esrExcludedFamilies          :: [String]
  , esrEvaluatorWorkers          :: Int
  , esrPairTimeoutSeconds        :: Int
  , esrActiveAuditProbability    :: Double
  }
  deriving (Eq, Show)

-- | The preregistered gate authorization block for active elimination.
data EliminationGate = EliminationGate
  { egDocumentId             :: String
  , egDecision               :: String
  , egAuthorizedPolicyVersion :: String
  }
  deriving (Eq, Show)

-- | The frozen elimination bake-off decision rule inputs.
data EliminationDecision = EliminationDecision
  { edScorePracticalMargin      :: Double
  , edRecallNoninferiorityMargin :: Double
  , edTopSetK                   :: Int
  }
  deriving (Eq, Show)

-- | An elimination bake-off specification.
data EliminationBakeoffSpec = EliminationBakeoffSpec
  { ebsExperimentId      :: String
  , ebsGameBinary        :: String  -- filesystem path
  , ebsObjectiveFile     :: String  -- filesystem path
  , ebsProposalSeeds     :: [Int]
  , ebsTaskSeed          :: Int
  , ebsTuningPairBudgets :: [Int]
  , ebsSharedRun         :: EliminationSharedRun
  , ebsDecision          :: EliminationDecision
  , ebsGate              :: EliminationGate
  }
  deriving (Eq, Show)

-- | Held-out and elimination-specific evidence from one completed child.
data EliminationChildFact = EliminationChildFact
  { ecfCellId                             :: String
  , ecfBudget                             :: Int
  , ecfSeed                               :: Int
  , ecfPolicy                             :: String
  , ecfManifestFingerprint                :: String
  , ecfBestCandidateFingerprint           :: String
  , ecfFinalistFingerprints               :: [String]
  , ecfHeldOutMeans                       :: [(String, Double)]
  , ecfHeldOutBestScore                   :: Double
  , ecfCompletedCohorts                   :: Int
  , ecfAcceptedUniqueCandidates           :: Int
  , ecfTerminalCandidateFailures          :: Int
  , ecfCensoredTuningAttempts             :: Int
  , ecfTuningPairAttempts                 :: Int
  , ecfTuningPhysicalGames                :: Int
  , ecfTuningSearchIterations             :: Int
  , ecfTuningWallTimeMs                   :: Int
  , ecfUnspentPairAttempts                :: Int
  , ecfOverrunPairAttempts                :: Int
  , ecfNominalEliminations                :: Int
  , ecfPruned                             :: Int
  , ecfAuditContinued                     :: Int
  , ecfAuditedBoundaryReversals           :: Int
  , ecfEstimatedBoundaryReversals         :: Double
  , ecfGrossNominalSuffixUniquePairs      :: Int
  , ecfAuditContinuationSuffixUniquePairs :: Int
  , ecfPlannedUniquePairSavings           :: Int
  , ecfSuspended                          :: Bool
  }
  deriving (Eq, Show)
