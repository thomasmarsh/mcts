module Artifacts where

import Candidate (ProposalSource)
import Deployment (ObjectiveEpoch, OpponentPanel)
import Effort (SearchEffort)
import Evaluation (TaskCorpus, TaskPrefix)
import Proposal (ProposerPolicy)
import Racing (ComputeBudget)

-- | A proposer specification: frozen policy parameters.
data ProposerSpecification = ProposerSpecification
  { psPolicy                  :: ProposerPolicy
  , psProposalSeed            :: Int
  , psTaskSeed                :: Int
  , psCohortSize              :: Int
  , psFinalists               :: Int
  , psBootstrapCandidates     :: Int
  , psRandomReserveCandidates :: Int
  , psSourceSchedule          :: [ProposalSource]
  , psChallengerSourceSchedule :: [ProposalSource]
  , psBootstrapSeed           :: Int
  , psReserveSeed             :: Int
  , psExcludedFamilies         :: [String]
  }
  deriving (Eq, Show)

-- | Shadow policy specification: paired bootstrap variant.
data PairedBootstrapPolicySpec = PairedBootstrapPolicySpec
  { pbpPracticalEffectMargin          :: Double
  , pbpEliminationProbabilityThreshold :: Double
  , pbpResamples                      :: Int
  , pbpMethodVersion                  :: String
  , pbpMinimumEligiblePrefixPairs      :: Int
  }
  deriving (Eq, Show)

-- | Shadow policy specification: successive halving variant.
data SuccessiveHalvingPolicySpec = SuccessiveHalvingPolicySpec
  { shpReductionFactor             :: Int
  , shpPracticalEffectMargin       :: Double
  , shpMinimumEligiblePrefixPairs   :: Int
  , shpSurvivorFloor               :: Int
  , shpMethodVersion               :: String
  }
  deriving (Eq, Show)

-- | Union of shadow policy specifications.
data ShadowPolicySpec
  = PairedBootstrapPolicy PairedBootstrapPolicySpec
  | SuccessiveHalvingPolicy SuccessiveHalvingPolicySpec
  deriving (Eq, Show)

-- | Active elimination specification.
data ActiveEliminationSpec = ActiveEliminationSpec
  { aesAuditProbability :: Double
  , aesSamplerVersion   :: String
  , aesSafeyRuleVersion  :: String
  }
  deriving (Eq, Show)

-- | Candidate failure policy specification.
data CandidateFailurePolicySpec = CandidateFailurePolicySpec
  { cfpsMaxPairAttempts :: Int
  }
  deriving (Eq, Show)

-- | Diagnostic policy specification.
data DiagnosticPolicySpec = DiagnosticPolicySpec
  { dpsMaximumReserveSlots :: Int
  }
  deriving (Eq, Show)

-- | The frozen manifest: everything needed to reproduce a tuning run.
data Manifest = Manifest
  { mFingerprint                :: String
  , mRunId                      :: String
  , mEpoch                      :: ObjectiveEpoch
  , mPanel                      :: OpponentPanel
  , mTuningCorpus                :: TaskCorpus
  , mProductionValidationCorpus  :: TaskCorpus
  , mTuningPrefix               :: TaskPrefix
  , mTuningBlocks                :: [TaskPrefix]
  , mValidationPrefix            :: TaskPrefix
  , mEffortValues                :: (SearchEffort, SearchEffort, SearchEffort)
  , mComputeBudget               :: ComputeBudget
  , mProposerSpec                :: ProposerSpecification
  , mShadowPolicy                :: ShadowPolicySpec
  , mCandidateFailurePolicy       :: CandidateFailurePolicySpec
  , mActiveElimination            :: Maybe ActiveEliminationSpec
  , mDiagnosticPolicy             :: DiagnosticPolicySpec
  , mGameConfigFingerprint        :: String
  , mObjectiveFingerprint         :: String
  }
  deriving (Eq, Show)