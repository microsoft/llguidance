from typing import List, Tuple, Mapping, Optional, Sequence, Union
from ._util import TokenId, StopReason
from ._tokenizer import TokenizerWrapper


class LLTokenizer:
    vocab_size: int
    eos_token: TokenId

    def __new__(
        cls,
        tokenizer: TokenizerWrapper,
    ) -> "LLTokenizer":
        """
        Create a new tokenizer.
        """

    def greedy_tokenize(self, text: str) -> List[int]:
        """
        Tokenize the text using a greedy algorithm.
        This will not necesserily match BPE.
        """

    def tokenize_bytes(self, utf8bytes: bytes) -> List[int]:
        """
        Tokenize the text as bytes.
        This will use the underlaying Python tokenizer to tokenize valid UTF8
        prefix of the text, and then fallback to greedy_tokenize() for the last
        few bytes.
        """

    def tokenize_str(self, text: str) -> List[int]:
        """
        Same as tokenize_bytes, but for strings.
        """

    def dbg_tokens(self, tokens: List[int]) -> str:
        """
        Return a debug string representation of the tokens.
        The result is double-quoted and tokens are separated by '‧'.
        """

    def test_trace_tokens(self, tokens: List[int]) -> str:
        """
        Return a debug string representation of the tokens
        for test traces.
        """

    def decode_str(self, tokens: List[int]) -> str:
        """
        Decode the tokens into a string.
        Invalid UTF-8 will be replaced with the Unicode replacement character.
        """

    def decode_bytes(self, tokens: List[int]) -> bytes:
        """
        Decode the tokens into a bytes object.
        """


class LLInterpreter:

    def __new__(
        cls,
        tokenizer: LLTokenizer,
        llguidance_json: str,
        log_level: int = 1,
    ) -> "LLInterpreter":
        """
        Create a new interpreter.
        Args:
            tokenizer: LLTokenizer - the tokenizer to use
            llguidance_json: str - the JSON representation of the AG2 grammar/constraint
            log_level: int - the verbosity level of the interpreter
                0 is silent, 1 is warnings, 2 is verbose
        """

    def deep_copy(self) -> "LLInterpreter":
        """
        Create a deep copy of the interpreter.
        """

    def is_accepting(self) -> bool:
        """
        Check if the last mid_process() call resulted in overall accepting state
        of the parser.
        """

    def stop_reason(self) -> StopReason:
        """
        Get the reason why the parser stopped.
        Returns:
            "NotStopped" - Parser has not emitted stop() yet.
            "MaxTokensTotal" - max_tokens limit on the total number of tokens has been reached.
            "MaxTokensParser" - max_tokens limit on the number of tokens in the top-level parser has been reached.
            "ParserNotAccepting" - LLM generated tokens that were not accepted by the parser.
            "NoExtension" - Top-level parser indicates that no more bytes can be added.
            "NoExtensionBias" - Top-level parser indicates that no more bytes can be added, however it was recognized late.
            "EndOfSentence" - Top-level parser allowed EOS (as it was in an accepting state), and EOS was generated.
            "InternalError" - Something went wrong with creating a nested parser.
        """

    def process_prompt(self, prompt: List[TokenId]) -> List[TokenId]:
        """
        Perform any adjustments to the prompt before completion.
        Returns the adjusted prompt.
        """

    def mid_process(self) -> Tuple[Optional[bytes], str]:
        """
        Perform next parsing step.
        Returns: optional token mask and a JSON string.
        """

    def post_process(
            self,
            sampled_token: Optional[TokenId]) -> Tuple[int, List[TokenId]]:
        """
        Perform any adjustments to the sampled token.
        Returns the number of tokens to remove from the prompt and the
        list of tokens to append.
        If mid_process() returned None, this should be called immedietly with None.
        """
