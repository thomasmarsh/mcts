module Deployment where

import Effort (SearchEffort)

-- | Role of an opponent in the panel.
data OpponentRole = Default | HistoricalReference
  deriving (Eq, Show)

-- | Source of an opponent configuration.
data OpponentSource = OpponentSchemaDefault | Inline
  deriving (Eq, Show)

-- | An opponent identity with its configuration, role, and panel weight.
data Opponent = Opponent
  { oppId                     :: String
  , oppSourceId               :: OpponentSource
  , oppLabel                  :: String
  , oppRole                   :: OpponentRole
  , oppWeight                 :: Int
  , oppCanonicalConfig        :: String  -- canonical JSON
  , oppConfigurationFingerprint :: String
  }
  deriving (Eq, Show)

-- | A frozen set of opponents with weights.
data OpponentPanel = OpponentPanel
  { panelId         :: String
  , panelFingerprint :: String
  , panelOpponents   :: [Opponent]
  , panelTotalWeight  :: Int
  }
  deriving (Eq, Show)

-- | An objective epoch: the frozen reference frame for all observations within it.
data ObjectiveEpoch = ObjectiveEpoch
  { epochId         :: String
  , epochFingerprint :: String
  }
  deriving (Eq, Show)

-- | A deployment case: everything needed to reproduce one comparison.
data DeploymentCase = DeploymentCase
  { dcGameConfig     :: String  -- canonical JSON game config
  , dcOpening        :: String  -- starting position/opening prefix
  , dcOpponentConfig :: String  -- opponent's canonical config
  , dcSeed           :: Int
  }
  deriving (Eq, Show)

-- | A versioned distribution over deployment cases.
data DeploymentDistribution = DeploymentDistribution
  { ddVersion :: String
  , ddCases   :: [(Double, DeploymentCase)]
  }
  deriving (Eq, Show)

-- | A declared optimization objective.
data TuningObjective = TuningObjective
  { objEpoch            :: ObjectiveEpoch
  , objProductionBudget :: SearchEffort
  }
  deriving (Eq, Show)