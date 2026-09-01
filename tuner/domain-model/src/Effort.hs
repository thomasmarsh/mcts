module Effort where

-- | Search effort per move: a kind and a positive integer value.
data EffortKind = Iterations | TimeMs
  deriving (Eq, Show)

data SearchEffort = SearchEffort
  { effortKind  :: EffortKind
  , effortValue :: Int
  }
  deriving (Eq, Show)

-- | Whether an observed effort exceeds a production effort of the same kind.
exceedsSameKind :: SearchEffort -> SearchEffort -> Bool
exceedsSameKind observed production =
  effortKind observed == effortKind production && effortValue observed > effortValue production