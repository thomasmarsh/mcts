module Diagnostic where

import Candidate (Candidate)
import Evaluation (DiagnosticPairTask, DiagnosticPairResult)
import Evidence (Observation)
import Statistics (Estimate)

-- | A directed edge in the diagnostic matchup graph.
data DiagnosticEdge = DiagnosticEdge
  { deEdgeId             :: String
  , deLeftCandidateId    :: String
  , deRightCandidateId   :: String
  , dePairResults        :: [DiagnosticPairResult]
  , deEstimate           :: Maybe Estimate
  , deMaterialDirection  :: Maybe String  -- "left_to_right" | "right_to_left" | Nothing
  }
  deriving (Eq, Show)

-- | Whether an edge is unresolved (estimate crosses 0.5).
edgeUnresolved :: DiagnosticEdge -> Bool
edgeUnresolved = undefined

-- | A connected component in the material cycle graph.
data MaterialCycleComponent = MaterialCycleComponent
  { mccCandidateIds           :: [String]
  , mccWitnessCycleCandidateIds :: [String]
  }
  deriving (Eq, Show)

-- | The diagnostic graph: direct candidate-vs-candidate evidence.
data DiagnosticGraph = DiagnosticGraph
  { dgEdges                   :: [DiagnosticEdge]
  , dgMaterialCycleComponents :: [MaterialCycleComponent]
  , dgFingerprint             :: String
  }
  deriving (Eq, Show)

-- | Build a diagnostic graph from cohort candidates and completed pairs.
buildDiagnosticGraph
  :: [Candidate] -> [DiagnosticPairResult] -> [(String, Int)] -> DiagnosticGraph
buildDiagnosticGraph = undefined

-- | Reason for choosing a diagnostic pair.
data DiagnosticReason
  = GraphConnectivity
  | PotentialCycleClosure
  | RankingBoundary
  | UnresolvedEdge
  deriving (Eq, Show)

-- | A pending diagnostic pair allocation.
data EvaluateDiagnosticPair = EvaluateDiagnosticPair
  { edpCohortIndex :: Int
  , edpReason      :: DiagnosticReason
  , edpTask        :: DiagnosticPairTask
  }
  deriving (Eq, Show)

-- | Find the next diagnostic pair to allocate.
nextDiagnosticAllocation
  :: [Candidate]
  -> [DiagnosticPairResult]
  -> Int  -- diagnostic budget remaining
  -> Int  -- finalist count
  -> Maybe EvaluateDiagnosticPair
nextDiagnosticAllocation = undefined

-- | Select validation shortlist accounting for non-transitive cycles.
selectValidationShortlist
  :: [Candidate] -> [Observation] -> Int -> DiagnosticGraph
  -> ([Candidate], Maybe String, Maybe String)
selectValidationShortlist = undefined