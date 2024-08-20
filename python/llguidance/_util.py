from typing import Literal

TokenId = int
StopReason = Literal[
    "NotStopped",
    "MaxTokensTotal",
    "MaxTokensParser",
    "ParserTooComplex",
    "LexerTooComplex",
    "NoExtension",
    "NoExtensionBias",
    "EndOfSentence",
    "InternalError",
]
