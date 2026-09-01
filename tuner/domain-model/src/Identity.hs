module Identity where

-- | Fingerprint (SHA-256 hex digest) of a canonical JSON value.
type Fingerprint = String

-- | A stable, deterministic ID built from a kind tag and a fingerprint.
stableId :: String -> String -> String
stableId kind fp = kind <> "-" <> fp