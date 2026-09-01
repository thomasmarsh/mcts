module Proposal where

import Candidate (Candidate)
import Evidence (ObservationFrontier, Observation)

-- | Source of a proposed configuration.
data ProposalSource
  = SchemaDefault
  | BootstrapRandom
  | SmacModel
  | RandomReserve
  | LowDiscrepancyQMC
  deriving (Eq, Show)

-- | Provenance of a proposed candidate: how it was generated.
data ProposalProvenance = ProposalProvenance
  { ppSource          :: ProposalSource
  , ppProposerVersion :: String
  , ppSourceAttempt   :: Int
  , ppOrigin          :: Maybe String     -- e.g. elite-centered, random forest, TPE
  , ppAcquisition     :: Maybe Double     -- acquisition function value
  , ppPrediction      :: Maybe Double     -- predicted production score
  , ppUncertainty     :: Maybe Double     -- predictive uncertainty
  , ppParentId        :: Maybe String     -- parent/elite that spawned this
  }
  deriving (Eq, Show)

-- | A proposed configuration with its provenance and the observation frontier
-- visible to the proposer at creation time.
data Proposal = Proposal
  { pIndex          :: Int
  , pCohortIndex    :: Int
  , pCohortSlot     :: Int
  , pCandidate      :: Candidate
  , pFrontier       :: ObservationFrontier
  , pProvenance     :: ProposalProvenance
  }
  deriving (Eq, Show)

-- | The interface for a model-guided proposer. Observes completed tuning
-- results at a common frontier and proposes one new configuration.
-- @m@ is the effect context (e.g. IO, perhaps with randomness).
newtype ModelProposer m = ModelProposer
  { mpAsk
      :: [Observation]          -- ^ observations at the common frontier
      -> ObservationFrontier    -- ^ the frontier they belong to
      -> [String]               -- ^ fingerprints to exclude (already proposed)
      -> Int                    -- ^ proposal attempt number
      -> m (Maybe Candidate)    -- ^ a proposed configuration (or Nothing)
  }

-- | The allocation policy that decides what kind of proposal to request next.
-- Balances model-guided, elite-centered, random, and low-discrepancy sources.
proposalPolicy
  :: Int  -- ^ cohort size
  -> Int  -- ^ bootstrap candidates
  -> Int  -- ^ random reserve slots
  -> Int  -- ^ cohort index (0 = initial, 1+ = challenger)
  -> Int  -- ^ accepted slot in this cohort
  -> ProposalSource
proposalPolicy = undefined

-- | Compute the observation frontier visible to the model proposer
-- from completed tuning observations at a common prefix.
visibleFrontier :: [Observation] -> Maybe ObservationFrontier
visibleFrontier = undefined