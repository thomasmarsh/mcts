module Constraints where

import Candidate (Candidate)
import Schema (TuningSchema)

-- | Constraint policy version baked into manifests.
constraintPolicyVersion :: String
constraintPolicyVersion = "space-constraints-v1"

-- | A parameter scalar value: the set of possible constraint operands.
data ParamScalar
  = PSBool Bool
  | PSInt Int
  | PSNumber Double
  | PSString String
  deriving (Eq, Show)

-- | Treat the parameter as a constant with the given value.
data FixOp = FixOp { fixValue :: ParamScalar }
  deriving (Eq, Show)

-- | Replace a float/int parameter's bounds with a sub-range.
data RangeOp = RangeOp
  { rangeLow  :: Either Int Double
  , rangeHigh :: Either Int Double
  }
  deriving (Eq, Show)

-- | Restrict a categorical/bool parameter to a proper subset of its choices.
data ChoicesOp = ChoicesOp { choicesValues :: [ParamScalar] }
  deriving (Eq, Show)

-- | A single-parameter narrowing.
data SetOp
  = SetFix FixOp
  | SetRange RangeOp
  | SetChoices ChoicesOp
  deriving (Eq, Show)

-- | A constraint: a set of per-parameter narrowings, optionally guarded by a
-- when-predicate over categorical parameters.
data Constraint = Constraint
  { cWhen :: [(String, [ParamScalar])]  -- sorted, may be empty (unconditional)
  , cSets :: [(String, SetOp)]          -- sorted, non-empty
  }
  deriving (Eq, Show)

-- | A list of constraints. Unconditional narrowings are baked into the schema;
-- predicated ones are enforced at the ConfigSpace/candidate-gate boundary.
type Constraints = [Constraint]

-- | Check that the constraint list is valid against a tuning schema.
validateConstraints :: TuningSchema -> Constraints -> Either String ()
validateConstraints = undefined

-- | Shrink the schema by applying statically-decidable constraints.
constrainedSchema :: TuningSchema -> Constraints -> TuningSchema
constrainedSchema = undefined

-- | Reject a candidate whose canonical config violates any active constraint.
requireCandidateAllowed :: Candidate -> Constraints -> Either String ()
requireCandidateAllowed = undefined