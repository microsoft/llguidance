from typing import Literal

TokenId = int
StopReason = Literal["NotStopped", "MaxTokensTotal", "MaxTokensParser", "ParserNotAccepting", "NoExtension", "NoExtensionBias", "EndOfSentence", "InternalError"]
