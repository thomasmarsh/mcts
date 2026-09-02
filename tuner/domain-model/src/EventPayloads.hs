module EventPayloads where

import Candidate (ProposalSource)
import Effort (SearchEffort)
import Evaluation (DiagnosticPairTask, Phase)
import Json (JsonValue)
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

-- | Reason a proposal was rejected.
data RejectionReason = Duplicate | SemanticValidation
  deriving (Eq, Show)

-- | A proposal identity: fixed fields common to proposal events.
data ProposalIdentity = ProposalIdentity
  { piProposalIndex   :: Int
  , piCohortIndex     :: Int
  , piCohortSlot      :: Int
  , piSource          :: ProposalSource
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

-- | A single field-level validation error from a game binary.
data PanelFieldError = PanelFieldError
  { pfeField          :: String
  , pfeMessage        :: String
  , pfeCandidateIndex :: Maybe Int
  }
  deriving (Eq, Show)

-- | Per-opponent semantic rejection errors.
data PanelRejection = PanelRejection
  { prOpponentId :: String
  , prErrors     :: [PanelFieldError]
  }
  deriving (Eq, Show)

-- | The append-only evidence envelope.
data EvidenceEvent = EvidenceEvent
  { eeSequence :: Int
  , eeType     :: EventType
  , eePayload  :: EventPayload
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
      , epPrReason   :: RejectionReason
      , epPrErrors   :: [PanelRejection]
      }
  | CohortCompletedPayload
      { epCcCohortIndex          :: Int
      , epCcCandidateIds         :: [String]
      , epCcRetainedCandidateIds :: [String]
      , epCcProposalSources      :: [String]
      , epCcScheduleVersion      :: String
      , epCcFinalFrontierId      :: String
      }
  | PairStartedPayload
      { epPsIdentity :: PairIdentity
      , epPsTaskSeed :: Int
      }
  | PairCompletedPayload
      { epPc2Identity    :: PairIdentity
      , epPc2Games       :: (JsonValue, JsonValue)
      , epPc2PairUtility :: Double
      }
  | PairFailedPayload
      { epPfIdentity      :: PairIdentity
      , epPfKind          :: String
      , epPfCommand       :: [String]
      , epPfReturncode    :: Maybe Int
      , epPfStderr        :: String
      , epPfStdout        :: String
      , epPfPartialOutput :: [String]
      }
  | DiagnosticPairStartedPayload
      { epDpsTask :: DiagnosticPairTask
      }
  | DiagnosticPairCompletedPayload
      { epDpcTask  :: DiagnosticPairTask
      , epDpcGames :: (JsonValue, JsonValue)
      }
  | DiagnosticPairFailedPayload
      { epDpfTask    :: DiagnosticPairTask
      , epDpfKind    :: String
      , epDpfMessage :: String
      }
  | CandidateFailedPayload
      { epCfPolicyVersion             :: String
      , epCfReason                    :: String
      , epCfCohortIndex               :: Int
      , epCfCandidateId               :: String
      , epCfTriggeringPair            :: PairIdentity
      , epCfStartedAttempts           :: Int
      , epCfFailedAttempts            :: Int
      , epCfCensoredAttempts          :: Int
      , epCfCompletedTuningPairIds    :: [String]
      }
  | RunInterruptedPayload
      { epRiStage  :: String
      , epRiPairId :: Maybe String
      }
  | RunFailedPayload
      { epRfKind    :: String
      , epRfMessage :: String
      }
  | ObservationCompletedPayload
      { epOcObservationId    :: String
      , epOcCandidateId      :: String
      , epOcPhase            :: Phase
      , epOcObjectiveEpochId :: String
      , epOcCorpusId         :: String
      , epOcPrefixId         :: String
      , epOcPrefixTaskIds    :: [String]
      , epOcPrefixLength     :: Int
      , epOcSearchEffort     :: SearchEffort
      , epOcPairUtilities    :: [Double]
      , epOcEstimate         :: JsonValue
      , epOcCounts           :: JsonValue
      }
  | FinalistsSelectedPayload
      { epFsFinalistIds         :: [String]
      , epFsTuningEstimates     :: JsonValue
      , epFsObjectiveEpochId    :: String
      , epFsCorpusId            :: String
      , epFsPrefixId            :: String
      , epFsPrefixTaskIds       :: [String]
      , epFsSearchEffort        :: SearchEffort
      , epFsSelectionRuleVersion :: String
      }
  | RunCompletedPayload
      { epRcManifestFingerprint   :: String
      , epRcAcceptedIds           :: [String]
      , epRcFinalistIds           :: [String]
      , epRcEvidenceCounts        :: JsonValue
      , epRcValidationClaim       :: String
      , epRcObjectiveEpochId      :: String
      , epRcValidationPrefixId    :: String
      , epRcValidationSearchEffort :: SearchEffort
      , epRcMissingProductionAxes :: [String]
      , epRcCohortFrontierId      :: String
      }
  deriving (Eq, Show)
