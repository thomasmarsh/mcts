module FamilyExclusions where

-- | Frozen named-family exclusion policy. Excluded families must be sorted and
-- duplicate-free, and must not exclude every family.
type FamilyName = String

validateFamilyExclusions :: [FamilyName] -> [FamilyName] -> Bool
validateFamilyExclusions _choices _excluded = undefined

requireCandidateFamilyAllowed :: FamilyName -> [FamilyName] -> Bool
requireCandidateFamilyAllowed _family _excluded = undefined