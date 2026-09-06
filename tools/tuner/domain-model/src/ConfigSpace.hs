module ConfigSpace where

-- | The kind of a parameter.
data ParamKind = Categorical | Integer_ | Numeric | Boolean
  deriving (Eq, Show)

-- | A raw parameter value.
data ParamValue
  = CategoricalValue String
  | IntegerValue   Int
  | NumericValue   Double
  | BooleanValue   Bool
  deriving (Eq, Show)

-- | A parameter name.
type ParamName = String

-- | A condition determining whether a parameter is active.
data Condition
  = And [Condition]
  | Or  [Condition]
  | Not Condition
  | Equals ParamName ParamValue
  | In    ParamName [ParamValue]
  deriving (Eq, Show)

-- | Transformation applied to a numeric/integer parameter.
data Scale = Linear | Log
  deriving (Eq, Show)

-- | A parameter definition.
data ParamDef = ParamDef
  { pName    :: ParamName
  , pKind    :: ParamKind
  , pDefault :: ParamValue
  , pScale   :: Maybe Scale
  , pActive  :: Maybe Condition
  }
  deriving (Eq, Show)

-- | A relational constraint.
data RelConstraint
  = LessThan   ParamName ParamName
  | EqualTo    ParamName ParamName
  | Incompatible ParamName ParamName
  deriving (Eq, Show)

-- | A complete configuration space.
data ConfigSpace = ConfigSpace
  { csParams      :: [ParamDef]
  , csForbidden   :: [Condition]
  , csConstraints :: [RelConstraint]
  }
  deriving (Eq, Show)

-- | A canonicalized JSON configuration string.
type CanonicalConfig = String

-- | Validate an assignment.
validate :: ConfigSpace -> [(ParamName, ParamValue)] -> Bool
validate = undefined

-- | Canonicalize an assignment to a JSON string.
canonicalize :: ConfigSpace -> [(ParamName, ParamValue)] -> Maybe CanonicalConfig
canonicalize = undefined