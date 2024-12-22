from typing import List, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from zllm.core.datatypes.sequence import Sequence
from zllm.core.sequence_manager.base_sequence_manager import BaseSequenceManager
from zllm.transformer_utils.tokenizer import detokenize_incrementally
# from sarathi.transformers_utils.tokenizer import detokenize_incrementally


class EngineSequenceManager(BaseSequenceManager):

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config,
    ):
        super().__init__(config)
        self.tokenizer = tokenizer

    def _decode_seq(self, seq: Sequence) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset, read_offset) = (
            detokenize_incrementally(
                self.tokenizer,
                all_input_ids=seq.get_token_ids(),
                prev_tokens=seq.tokens,
                prefix_offset=seq.prefix_offset,
                read_offset=seq.read_offset,
                skip_special_tokens=True,
            )
        )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text
        
        # todo 重写这段代码 
        # (new_tokens, new_output_text, prefix_offset, read_offset) = (
        #     detokenize_incrementally(
        #         self.tokenizer,
        #         all_input_ids=seq.get_token_ids(),
        #         prev_tokens=seq.tokens,
        #         prefix_offset=seq.prefix_offset,
        #         read_offset=seq.read_offset,
        #         skip_special_tokens=True,
        #     )
        # )

        # all_token_ids = seq.get_token_ids()
        # new_token_ids = all_token_ids[seq.read_offset:]

        # new_output_text = self.tokenizer.decode(new_token_ids)
        # if not new_output_text.endswith("�"):
        #     (new_tokens, new_output_text, prefix_offset, read_offset) = (
        #         [new_output_text], new_output_text, seq.prefix_offset+len(new_token_ids) , seq.read_offset+len(new_token_ids),
        #     )
        # else:
        #     (new_tokens, new_output_text, prefix_offset, read_offset) = (
        #         [], "", seq.prefix_offset , seq.read_offset,
        #     )

        # if seq.tokens is None:
        #     seq.tokens = new_tokens
        # else:
        #     seq.tokens.extend(new_tokens)

        # seq.prefix_offset = prefix_offset
        # seq.read_offset = read_offset
        # seq.output_text += new_output_text

    def _on_append_token(self, seq: Sequence) -> None:
        self._decode_seq(seq)

    def _get_block_table(self, seq: Sequence) -> List[int]:
        return []
