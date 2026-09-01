module EventPayloads where

import Candidate (CandidateFailure)
import Effort (SearchEffort)
import Evaluation (DiagnosticPairTask, Phase)
import Racing (ResourceAllocation)
import Shadow (ShadowRaceDecision)

-- | All event types that appear in the evidence log.
data EventType
  = EvProposalCreated
  | EvProposalAccepted
  | EvProposalRejected
  | EvCohortCompleted
  | EvPairStarted
  | EvPairCompleted
  | EvPairFailed
  | EvDiagnosticPairStarted
  | EvDiagnosticPairCompleted
  | EvDiagnosticPairFailed
  | EvRunInterrupted
  | EvRunFailed
  | EvObservationCompleted
  | EvFinalistsSelected
  | EvRunCompleted
  | EvAllocationDecided
  | EvShadowRaceDecided
  | EvCandidateFailed
  deriving (Eq, Show)

-- | A proposal identity: fixed fields common to proposal events.
data ProposalIdentity = ProposalIdentity
  { piProposalIndex   :: Int
  , piCohortIndex     :: Int
  , piCohortSlot      :: Int
  , piSource          :: String
  , piSourceAttempt   :: Int
  , piCandidateId     :: String
  , piFingerprint     :: String
  , piCanonicalConfig :: String
  }
  deriving (Eq, Show)

-- | A pair identity: fixed fields common to pair events.
data PairIdentity = PairIdentity
  { piiPhase        :: Phase
  , piiCandidateId  :: String
  , piiTaskId       :: String
  , piiPairId       :: String
  , piiOpponentId   :: String
  , piiSearchEffort :: SearchEffort
  }
  deriving (Eq, Show)

-- | Union of all evidence event payloads.
data EventPayload
  = AllocationDecidedPayload
      { epAllocAllocation    :: ResourceAllocation
      , epAllocPolicyVersion :: String
      }
  | ShadowRaceDecidedPayload
      { epShadowDecision :: ShadowRaceDecision
      }
  | ProposalCreatedPayload
      { epPcIdentity               :: ProposalIdentity
      , epPcFrontierId             :: String
      , epPcFrontierObservationIds :: [String]
      , epPcProposerVersion        :: String
      , epPcOrigin                 :: Maybe String
      , epPcAcquisition            :: Maybe Double
      , epPcPrediction             :: Maybe Double
      , epPcUncertainty            :: Maybe Double
      , epPcParentCandidateId      :: Maybe String
      }
  | ProposalAcceptedPayload
      { epPaIdentity                  :: ProposalIdentity
      , epPaPanelResponseFingerprints :: [String]
      }
  | ProposalRejectedPayload
      { epPrIdentity :: ProposalIdentity
      , epPrReason   :: String
      }
  | CohortCompletedPayload
      { epCcCohortIndex          :: Int
      , epCcCandidateIds         :: [String]
      , epCcRetainedCandidateIds :: [String]
      , epCcProposalSources      :: [String]
      }
  | PairStartedPayload
      { epPsIdentity :: PairIdentity
      , epPsTaskSeed :: Int
      }
  | PairCompletedPayload
      { epPc2Identity    :: PairIdentity
      , epPc2PairUtility :: Double
      }
  | PairFailedPayload
      { epPfIdentity :: PairIdentity
      , epPfKind     :: String
      , epPfMessage  :: String
      }
  | DiagnosticPairCompletedPayload
      { epDpcTask :: DiagnosticPairTask
      }
  | CandidateFailedPayload
      { epCfFailure :: CandidateFailure
      }
  | ObservationCompletedPayload
      { epOcObservationId :: String
      , epOcCandidateId   :: String
      , epOcPhase         :: Phase
      , epOcPrefixId      :: String
      , epOcPairUtilities :: [Double]
      }
  | FinalistsSelectedPayload
      { epFsFinalistIds :: [String]
      }
  | RunCompletedPayload
      { epRcFinalistIds :: [String]
      }
  | RunInterruptedPayload
      { epRiStage  :: String
      , epRiPairId :: Maybe String
      }
  | RunFailedPayload
      { epRfKind    :: String
      , epRfMessage :: String
      }
  deriving (Eq, Show)