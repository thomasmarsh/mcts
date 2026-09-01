module ConfigSpace where

-- | A configuration space is a typed, conditional graph of parameters.
-- Parameters may be categorical, integer, numeric, or Boolean.
-- Activation conditions (conjunctions/disjunctions) define which
-- parameters are active given other parameter values.
-- Forbidden combinations and relational constraints are enforced.

-- | The kind of a parameter.
data ParamKind = Categorical | Integer_ | Numeric | Boolean
  deriving (Eq, Show)

-- | A parameter name within the configuration space.
type ParamName = String

-- | A raw parameter value before canonicalization.
data ParamValue
  = CategoricalValue String
  | IntegerValue   Int
  | NumericValue   Double
  | BooleanValue   Bool
  deriving (Eq, Show)

-- | A condition determining whether a parameter is active.
data Condition
  = And [Condition]
  | Or  [Condition]
  | Not Condition
  | Equals ParamName ParamValue
  | In    ParamName [ParamValue]
  deriving (Eq, Show)

-- | A parameter definition with its kind, default, optional scale, and activation condition.
data ParamDef = ParamDef
  { pName    :: ParamName
  , pKind    :: ParamKind
  , pDefault :: ParamValue
  , pScale   :: Maybe Scale
  , pActive  :: Maybe Condition  -- Nothing means always active
  }
  deriving (Eq, Show)

-- | Transformation applied to a numeric/integer parameter (e.g. log scaling).
data Scale = Linear | Log
  deriving (Eq, Show)

-- | A relational constraint between two parameters.
data RelConstraint
  = LessThan   ParamName ParamName
  | EqualTo    ParamName ParamName
  | Incompatible ParamName ParamName
  deriving (Eq, Show)

-- | A complete configuration space description.
data ConfigSpace = ConfigSpace
  { csParams      :: [ParamDef]
  , csForbidden   :: [Condition]  -- global forbidden combinations
  , csConstraints :: [RelConstraint]  -- relational constraints
  }
  deriving (Eq, Show)

-- | A map from parameter name to its assigned value.
type ParamAssign = [(ParamName, ParamValue)]

-- | Validate an assignment against the space: check forbidden conditions
-- and relational constraints.
validate :: ConfigSpace -> ParamAssign -> Bool
validate = undefined

-- | Canonicalize an assignment: remove inactive params, sort keys, apply
-- equivalence rules. Returns Nothing if invalid.
canonicalize :: ConfigSpace -> ParamAssign -> Maybe ParamAssign
canonicalize = undefined

-- | The default assignment for a space (all active params at their defaults).
defaults :: ConfigSpace -> ParamAssign
defaults = undefined