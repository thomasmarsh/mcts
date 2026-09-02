module FamilyExclusions where

import Candidate (Candidate)
import Schema (TuningSchema)

-- | A family name in the configuration schema.
type FamilyName = String

-- | The frozen named-family exclusion policy version.
familyExclusionPolicyVersion :: String
familyExclusionPolicyVersion = "named-family-exclusions-v1"

-- | Normalize excluded families: nonempty, trimmed, sorted, duplicate-free.
normalizeFamilyExclusions :: [FamilyName] -> [FamilyName]
normalizeFamilyExclusions = undefined

-- | Validate excluded families against a tuning schema.
validateFamilyExclusions :: TuningSchema -> [FamilyName] -> Either String ()
validateFamilyExclusions = undefined

-- | Reject a candidate whose family is excluded.
requireCandidateFamilyAllowed :: Candidate -> [FamilyName] -> Either String ()
requireCandidateFamilyAllowed = undefined
