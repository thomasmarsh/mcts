module Artifacts where

import Candidate (ProposalSource)
import Constraints (Constraints)
import Deployment (ObjectiveEpoch, OpponentPanel)
import Effort (SearchEffort)
import Evaluation (TaskCorpus, TaskPrefix)
import Json (JsonValue)
import Proposal (ProposerPolicy)
import Racing (ComputeBudget)
import Schema (GameSpec)
import Shadow (ShadowMethodVersion, ShadowPolicyKind)

-- | A proposer specification: frozen policy parameters.
data ProposerSpecification = ProposerSpecification
  { psPolicy                   :: ProposerPolicy
  , psProposalSeed             :: Int
  , psTaskSeed                 :: Int
  , psCohortSize               :: Int
  , psFinalists                :: Int
  , psBootstrapCandidates      :: Int
  , psRandomReserveCandidates  :: Int
  , psSourceSchedule           :: [ProposalSource]
  , psChallengerSourceSchedule :: [ProposalSource]
  , psBootstrapSeed            :: Int
  , psReserveSeed              :: Int
  , psRuntimeVersions          :: [(String, String)]
  , psConstraints              :: Constraints
  }
  deriving (Eq, Show)

-- | Shadow policy specification: paired bootstrap variant.
data PairedBootstrapPolicySpec = PairedBootstrapPolicySpec
  { pbpKind                         :: ShadowPolicyKind
  , pbpPracticalEffectMargin        :: Double
  , pbpEliminationProbabilityThreshold :: Double
  , pbpResamples                    :: Int
  , pbpMethodVersion                :: ShadowMethodVersion
  , pbpMinimumEligiblePrefixPairs   :: Int
  }
  deriving (Eq, Show)

-- | Shadow policy specification: successive halving variant.
data SuccessiveHalvingPolicySpec = SuccessiveHalvingPolicySpec
  { shpKind                       :: ShadowPolicyKind
  , shpMethodVersion              :: ShadowMethodVersion
  , shpReductionFactor            :: Int
  , shpPracticalEffectMargin      :: Double
  , shpMinimumEligiblePrefixPairs :: Int
  , shpSurvivorFloor              :: Int
  , shpRankingRule                :: String
  , shpSpareMargin                :: Double
  }
  deriving (Eq, Show)

-- | Union of shadow policy specifications.
data ShadowPolicySpec
  = PairedBootstrapPolicy PairedBootstrapPolicySpec
  | SuccessiveHalvingPolicy SuccessiveHalvingPolicySpec
  deriving (Eq, Show)

-- | Candidate failure policy specification.
data CandidateFailurePolicySpec = CandidateFailurePolicySpec
  { cfpsMaxPairAttempts :: Int
  , cfpsPolicyVersion   :: String
  }
  deriving (Eq, Show)

-- | Active elimination specification.
data ActiveEliminationSpec = ActiveEliminationSpec
  { aesAuditProbability    :: Double
  , aesShadowPolicyKind    :: ShadowPolicyKind
  , aesShadowMethodVersion :: ShadowMethodVersion
  , aesShadowSpareMargin   :: Double
  , aesSamplerVersion      :: String
  , aesSafetyRuleVersion   :: String
  }
  deriving (Eq, Show)

-- | Diagnostic policy specification.
data DiagnosticPolicySpec = DiagnosticPolicySpec
  { dpsMaximumReserveSlots  :: Int
  , dpsEdgePolicyVersion    :: String
  , dpsSeedPolicyVersion    :: String
  , dpsGraphRuleVersion     :: String
  , dpsShortlistRuleVersion :: String
  }
  deriving (Eq, Show)

-- | The frozen manifest: everything needed to reproduce a tuning run.
data Manifest = Manifest
  { mFingerprint                :: String
  , mSpec                       :: GameSpec
  , mObjectiveSourcePath        :: String  -- filesystem path
  , mObjectiveId                :: String
  , mObjectiveFingerprint       :: String
  , mPanel                      :: OpponentPanel
  , mTuningCorpus               :: TaskCorpus
  , mProductionValidationCorpus :: TaskCorpus
  , mTuningPrefix               :: TaskPrefix
  , mTuningBlocks               :: [TaskPrefix]
  , mValidationPrefix           :: TaskPrefix
  , mEpoch                      :: ObjectiveEpoch
  , mProposerSpec               :: ProposerSpecification
  , mRunId                      :: String
  , mGameConfigFingerprint      :: String
  , mEffortValues               :: (SearchEffort, SearchEffort, SearchEffort)
  , mComputeBudget              :: ComputeBudget
  , mShadowPolicy               :: ShadowPolicySpec
  , mCandidateFailurePolicy     :: CandidateFailurePolicySpec
  , mActiveElimination          :: Maybe ActiveEliminationSpec
  , mDiagnosticPolicy           :: DiagnosticPolicySpec
  , mGameConfig                 :: String
  , mConstraints                :: Constraints
  }
  deriving (Eq, Show)

-- | Build and fingerprint a manifest from resolved inputs.
buildManifest :: Manifest
buildManifest = undefined

-- | Strictly decode a manifest transport object.
decodeManifestObject :: JsonValue -> Manifest
decodeManifestObject = undefined

-- | Return the frozen transport representation at the publishing boundary.
manifestJson :: Manifest -> JsonValue
manifestJson = undefined

-- | Classify the validation evidence as a production claim or a mechanics smoke
-- test, listing any missing production axes.
productionClaim
  :: TaskPrefix -> TaskCorpus -> SearchEffort -> SearchEffort -> (String, [String])
productionClaim = undefined
