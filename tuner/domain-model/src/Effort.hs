module Effort where

import Json (JsonValue)

-- | Search effort per move: a kind and a positive integer value.
data EffortKind = Iterations | TimeMs
  deriving (Eq, Show)

data SearchEffort = SearchEffort
  { effortKind  :: EffortKind
  , effortValue :: Int
  }
  deriving (Eq, Show)

-- | Smart constructor rejecting non-positive values.
mkSearchEffort :: EffortKind -> Int -> Either String SearchEffort
mkSearchEffort kind value
  | value <= 0 = Left "search effort value must be a positive integer"
  | otherwise  = Right (SearchEffort kind value)

-- | Whether an observed effort exceeds a production effort of the same kind.
exceedsSameKind :: SearchEffort -> SearchEffort -> Bool
exceedsSameKind observed production =
  effortKind observed == effortKind production && effortValue observed > effortValue production

-- | Transport representation of a frozen search effort.
encodeEffort :: SearchEffort -> JsonValue
encodeEffort = undefined

-- | Strict decode of a frozen search effort.
decodeEffort :: JsonValue -> String -> SearchEffort
decodeEffort = undefined