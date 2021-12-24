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


def text_transform_for_tokenizer(tokenizer_func,
                                 texts,
                                 get_type_ids: bool = False):
    """ transforms all text data to transformer tokenizer output.

    @param tokenizer_func: tokenizer partial ready to be used with only text
    @param texts: narratives and dialogs
    @param get_type_ids: returns token type ids, useful for multimodal processes
    @return: dict of token ids and attention masks
    """
    merged_text = merge_comic_texts(texts)
    inputs = tokenizer_func(text=merged_text)
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"] if get_type_ids else None
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
    }


def text_transform_for_elmo_embeddings_to_character_indexes(tokenizer_func, texts):
    """ Transforms text data into Elmo Embeddings

    @param tokenizer_func: find_character_indexes function of ElmoTextEmbedder
    @param texts:  list of texts in a panel
    @return: elmo character indexes
    """
    merged_text = merge_comic_texts(texts)
    character_ids = tokenizer_func(sentences=merged_text)
    return character_ids
    # elmo_representations = inputs['elmo_representations'][0]
    # return elmo_representations[0]
