module Schema where

import ConfigSpace (ConfigSpace)
import Json (JsonValue)

-- | The kind of a game-host tuning parameter.
data ParameterKind
  = FloatParam
  | IntParam
  | CategoricalParam
  | BoolParam
  | ConstantParam
  deriving (Eq, Show)

-- | A named AI preset exposed by the game host.
data AiPresetSpec = AiPresetSpec
  { apsId          :: String
  , apsLabel       :: String
  , apsDescription :: String
  }
  deriving (Eq, Show)

-- | A single tuning parameter as decoded from the game host's describe
-- response. Numeric bounds, categorical/bool choices, defaults, and constant
-- values are all canonical JSON values.
data ParameterSpec = ParameterSpec
  { psName          :: String
  , psKind          :: ParameterKind
  , psBounds        :: Maybe (Double, Double)  -- float/int parameters
  , psChoices       :: Maybe [JsonValue]       -- categorical/bool parameters
  , psDefault       :: Maybe JsonValue
  , psConstantValue :: Maybe JsonValue         -- constant parameters
  }
  deriving (Eq, Show)

-- | A conditional activation edge: when the parent parameter takes one of the
-- given values, the listed child parameters become active.
data ActivationCondition = ActivationCondition
  { acParent   :: String
  , acValues   :: [JsonValue]
  , acChildren :: [String]
  }
  deriving (Eq, Show)

-- | The typed, conditional tuning schema of one game host.
data TuningSchema = TuningSchema
  { tsId         :: String
  , tsBaselines  :: [String]
  , tsEvalRounds :: Int
  , tsParameters :: [ParameterSpec]
  , tsConditions :: [ActivationCondition]
  , tsGameConfig :: String  -- canonical JSON game config
  }
  deriving (Eq, Show)

-- | Everything the game host's describe response carries, decoded into strict
-- typed values.
data GameSpec = GameSpec
  { gsKind                  :: String
  , gsLabel                 :: String
  , gsDescription           :: String
  , gsDefaultGameConfig     :: String  -- canonical JSON
  , gsAiPresets             :: [AiPresetSpec]
  , gsTuning                :: TuningSchema
  , gsDescriptionFingerprint :: String
  , gsSchemaFingerprint     :: String
  , gsBinaryPath            :: String  -- filesystem path
  , gsBinarySha256          :: String
  , gsEngineFingerprint     :: String
  , gsRawDescription        :: String
  }
  deriving (Eq, Show)

-- | Decode a game-host describe response into a strict GameSpec.
decodeGameSpec :: JsonValue -> String -> String -> GameSpec
decodeGameSpec = undefined

-- | Build a concrete conditional space from a schema and seed, honoring the
-- frozen family-exclusion policy.
buildSpace :: TuningSchema -> Int -> [String] -> ConfigSpace
buildSpace = undefined

-- | Narrow one hyperparameter value to a JSON scalar.
paramValue :: JsonValue -> String -> JsonValue
paramValue = undefined

-- | Active (non-inactive) parameter values of a configuration.
activeValues :: ConfigSpace -> [(String, JsonValue)]
activeValues = undefined

-- | The schema default configuration's active values.
defaultValues :: ConfigSpace -> [(String, JsonValue)]
defaultValues = undefined

-- | One random configuration's active values.
randomValues :: ConfigSpace -> [(String, JsonValue)]
randomValues = undefined

-- | Project proposed values through the schema's conditional order, omitting
-- inactive parameters so candidate identity stays canonical.
conditionalValues :: TuningSchema -> [(String, JsonValue)] -> [(String, JsonValue)]
conditionalValues = undefined

-- | Non-constant parameters of a schema.
nonconstantParameters :: TuningSchema -> [ParameterSpec]
nonconstantParameters = undefined
