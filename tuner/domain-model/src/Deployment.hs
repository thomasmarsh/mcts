module Deployment where

import ConfigSpace (ParamAssign)

-- | A deployment case contains every variable needed to reproduce one comparison:
-- the game configuration, starting position or opening prefix, opponent config,
-- random seed, and rules/adjudication policy.
data DeploymentCase = DeploymentCase
  { dcGameConfig      :: ParamAssign  -- game/rules configuration
  , dcOpening         :: String       -- starting position or opening prefix
  , dcOpponentConfig  :: ParamAssign  -- opponent's strategy configuration
  , dcSeed            :: Int
  }
  deriving (Eq, Show)

-- | A versioned distribution over deployment cases. Tuning results are always
-- relative to one version of this distribution.
data DeploymentDistribution = DeploymentDistribution
  { ddVersion   :: String
  , ddCases     :: [(Double, DeploymentCase)]  -- weight + case
  }
  deriving (Eq, Show)

-- | Role of an opponent in the panel.
data OpponentRole
  = DefaultOpponent
  | HistoricalReference
  | CalibrationAnchor
  | FrontierStyle
  deriving (Eq, Show)

-- | An opponent identity with its configuration, role, and panel weight.
data Opponent = Opponent
  { oppId          :: String
  , oppLabel       :: String
  , oppRole        :: OpponentRole
  , oppWeight      :: Int
  , oppConfig      :: ParamAssign
  , oppFingerprint :: String
  }
  deriving (Eq, Show)

-- | A frozen set of opponents with weights, representing the deployment distribution
-- of opposition to evaluate against.
data OpponentPanel = OpponentPanel
  { panelId        :: String
  , panelFingerprint :: String
  , panelOpponents :: [Opponent]
  }
  deriving (Eq, Show)

-- | An objective epoch: a frozen opponent panel and start distribution that
-- defines the reference frame for all observations within it. Cross-epoch
-- comparisons use explicit models, never raw scalar mixing.
data ObjectiveEpoch = ObjectiveEpoch
  { epochId        :: String
  , epochFingerprint :: String
  , epochPanel     :: OpponentPanel
  , epochDeployment :: DeploymentDistribution
  }
  deriving (Eq, Show)

-- | A declared optimization objective: the expectation of seat-balanced utility
-- over the deployment distribution at a given production search budget.
data TuningObjective = TuningObjective
  { objEpoch          :: ObjectiveEpoch
  , objProductionBudget :: SearchEffort
  }
  deriving (Eq, Show)

-- | Search effort per move: either a fixed iteration count or a time budget in ms.
data SearchEffort
  = Iterations Int
  | TimeMs     Int
  deriving (Eq, Show)