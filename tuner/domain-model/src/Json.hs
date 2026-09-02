module Json where

-- | A canonical JSON value used at the transport and fingerprinting boundary.
--
-- Integers and floats are kept distinct so fingerprints match the Python
-- implementation, which never coerces an integer literal to a float.
data JsonValue
  = JsonNull
  | JsonBool Bool
  | JsonInt Int
  | JsonNumber Double
  | JsonString String
  | JsonArray [JsonValue]
  | JsonObject [(String, JsonValue)]
  deriving (Eq, Show)

-- | A JSON scalar: everything that may appear as a parameter value.
isScalar :: JsonValue -> Bool
isScalar JsonNull       = True
isScalar (JsonBool _)   = True
isScalar (JsonInt _)    = True
isScalar (JsonNumber _) = True
isScalar (JsonString _) = True
isScalar _              = False
