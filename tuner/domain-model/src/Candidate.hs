module Candidate where

import ConfigSpace (ParamAssign, ConfigSpace)

-- | An immutable canonical configuration. Its identity never changes once created.
-- Re-evaluating at another seed or fidelity adds evidence to the same candidate.
data Candidate = Candidate
  { candId             :: String
  , candFingerprint    :: String
  , candCanonicalConfig :: ParamAssign
  }
  deriving (Eq, Show)

-- | Create a candidate from a configuration space and raw assignment.
-- Validates and canonicalizes; returns Nothing on invalid input.
mkCandidate :: ConfigSpace -> ParamAssign -> Maybe Candidate
mkCandidate = undefined

-- | A candidate's lineage: who proposed it, when, and from what context.
data CandidateLineage = CandidateLineage
  { clProposerVersion :: String
  , clAcquisitionValue :: Maybe Double
  , clUncertainty     :: Maybe Double
  , clParentCandidateId :: Maybe String
  , clEvidenceEpochId :: String
  }
  deriving (Eq, Show)

-- | Validation result from a game binary for a candidate against an opponent.
data CandidateValidation = CandidateValidation
  { cvValid  :: Bool
  , cvErrors :: [String]
  }
  deriving (Eq, Show)

-- | Terminal failure of a candidate: too many failed pair attempts.
-- The candidate is removed from the active cohort.
data CandidateFailure = CandidateFailure
  { cfCandidateId       :: String
  , cfCohortIndex       :: Int
  , cfStartedAttempts   :: Int
  , cfFailedAttempts    :: Int
  , cfCensoredAttempts  :: Int
  }
  deriving (Eq, Show)