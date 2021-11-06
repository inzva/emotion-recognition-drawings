import math
import torch


def remove_redundant_spaces_in_sentence(sentence) -> str:
    return " ".join(sentence.split())


def merge_comic_texts(texts):
    merged_text = ""
    for text in texts:
        for sentence in text:
            if not isinstance(sentence, str) and math.isnan(sentence):
                continue
            merged_text += (" " + sentence)
    return merged_text


def text_transform_for_tokenizer(tokenizer_func, texts):
    """ transforms all text data to transformer tokenizer output.

    @param tokenizer_func: tokenizer partial ready to be used with only text
    @param texts: narratives and dialogs
    @return: dict of token ids and attention masks
    """
    merged_text = merge_comic_texts(texts)
    inputs = tokenizer_func(text=merged_text)
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long)
    }
