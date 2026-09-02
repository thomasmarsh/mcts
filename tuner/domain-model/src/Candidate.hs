module Candidate where

import ConfigSpace (CanonicalConfig)
import Effort (SearchEffort)

-- | An immutable canonical configuration. Re-evaluating at another seed or
-- fidelity adds evidence to the same candidate.
data Candidate = Candidate
  { candId             :: String
  , candFingerprint    :: String
  , candCanonicalConfig :: CanonicalConfig
  }
  deriving (Eq, Show)

-- | All proposal sources in the fixed mixed-source schedule.
data ProposalSource
  = SchemaDefault
  | BootstrapRandom
  | SmacModel
  | RandomReserve
  | RandomSearch
  | QmcSearch
  | IraceModel
  deriving (Eq, Show)

-- | Provenance of a proposed candidate.
data ProposalProvenance = ProposalProvenance
  { ppSource             :: ProposalSource
  , ppProposerVersion    :: String
  , ppSourceAttempt      :: Int
  , ppOrigin             :: Maybe String
  , ppAcquisition        :: Maybe Double
  , ppPrediction         :: Maybe Double
  , ppUncertainty        :: Maybe Double
  , ppParentCandidateId   :: Maybe String
  }
  deriving (Eq, Show)

-- | An observation frontier visible to the proposer.
data ObservationFrontier = ObservationFrontier
  { ofFrontierId         :: String
  , ofObjectiveEpochId   :: String
  , ofPrefixId           :: String
  , ofTaskIds            :: [String]
  , ofSearchEffort       :: SearchEffort
  , ofObservationIds     :: [String]
  }
  deriving (Eq, Show)

-- | A proposal: a candidate with its provenance and the frontier visible at creation time.
data Proposal = Proposal
  { propIndex        :: Int
  , propCohortIndex  :: Int
  , propCohortSlot   :: Int
  , propCandidate    :: Candidate
  , propFrontier     :: ObservationFrontier
  , propProvenance   :: ProposalProvenance
  }
  deriving (Eq, Show)

-- | Disposition of a proposal (stored as (proposalIndex, status) pairs).
data ProposalDisposition = Accepted | Rejected
  deriving (Eq, Show)

-- | Validation error from a game binary.
data ValidationError = ValidationError
  { veField          :: String
  , veMessage        :: String
  , veCandidateIndex :: Maybe Int
  }
  deriving (Eq, Show)

-- | Validation result for a candidate against one opponent.
data ValidationResult = ValidationResult
  { vrValid  :: Bool
  , vrErrors :: [ValidationError]
  }
  deriving (Eq, Show)

-- | Terminal failure of a candidate: too many failed pair attempts.
data CandidateFailure = CandidateFailure
  { cfCohortIndex               :: Int
  , cfCandidateId               :: String
  , cfTriggeringPairId          :: String
  , cfStartedAttempts           :: Int
  , cfFailedAttempts            :: Int
  , cfCensoredAttempts          :: Int
  , cfCompletedTuningPairIds     :: [String]
  }
  deriving (Eq, Show)

-- | Facts about attempts for one pair.
data PairAttemptFacts = PairAttemptFacts
  { pafStartedAttempts   :: Int
  , pafFailedAttempts    :: Int
  , pafCensoredAttempts  :: Int
  , pafCompletedAttempts :: Int
  }
  deriving (Eq, Show)