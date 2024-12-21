

from typing import List, Optional, Tuple, Union

from zllm.logger import init_logger

logger = init_logger(__name__)

def _convert_tokens_to_string_with_added_encoders(
    tokenizer,
    output_tokens: List[str],
    skip_special_tokens: bool,
) -> str:
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts = []
    current_sub_text = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if token in tokenizer.get_added_vocab():
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    return " ".join(sub_texts)


def detokenize_incrementally(
    tokenizer,
    all_input_ids: List[int],
    prev_tokens: Optional[List[str]],
    prefix_offset: int = 0,
    read_offset: int = 0,
    skip_special_tokens: bool = False,
) -> Tuple[List[str], str, int, int]:
    new_token_id = all_input_ids[-1]

    # This is the first iteration for this sequence
    if prev_tokens is None:
        try:
            new_tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids[-6:], skip_special_tokens=skip_special_tokens
            )
        except ValueError as e:
            new_tokens = ["[UNK]"] * 6
            # logger.warning(f"Warning: {e}")

        output_tokens = new_tokens
        # 5 is an arbitrary value that should work for all
        # tokenizers (bigger = more conservative).
        # Subtract 1 extra to account for the generated token.
        prefix_offset = max(len(output_tokens) - 6, 0)
        read_offset = max(len(output_tokens) - 1, 0)
    else:
        # Put new_token_id in a list so skip_special_tokens is respected
        try:
            new_tokens = tokenizer.convert_ids_to_tokens(
                [new_token_id], skip_special_tokens=skip_special_tokens
            )
        except ValueError as e:
            new_tokens = [prev_tokens[-1]]
            logger.warning(f"Warning: {e}")
        output_tokens = prev_tokens + new_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset]
        )
        new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
        )

    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        new_text = new_text[len(prefix_text) :]
        return new_tokens, new_text, read_offset, len(output_tokens)
    else:
        return new_tokens, "", prefix_offset, read_offset
