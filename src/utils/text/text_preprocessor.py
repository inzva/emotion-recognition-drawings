import math
import string
from collections import Counter
from typing import List
import pathlib

import spacy
import re
from tqdm import tqdm
from os.path import exists
from src.datamodules.datasets.emoreccom import EmoRecComDataset
from src.utils.data_persistance_utils import save_object, load_object
import numpy as np

from src.utils.text.text_utils import remove_redundant_spaces_in_sentence

spacy_en = spacy.load('en_core_web_sm')


class TextPreprocessor:
    def __init__(self,
                 dataset: EmoRecComDataset,
                 minimum_word_count: int = 2,
                 unk_token_position: int = 1,
                 empty_token_position: int = 0,
                 encoding_max_length: int = 70,
                 words_path: str = pathlib.Path(__file__).parent.resolve().joinpath("emoreccom_word.pickle"),
                 vocab2index_path: str = pathlib.Path(__file__).parent.resolve().joinpath(
                     "emoreccom_vocab2index.pickle")
                 ):
        self.dataset = dataset
        self.minimum_word_count = minimum_word_count
        self.unk_token_position = unk_token_position
        self.empty_token_position = empty_token_position
        self.encoding_max_length = encoding_max_length
        # creating vocabulary
        self.vocab2index = {'': self.empty_token_position, "UNK": self.unk_token_position}
        self.words = ['', "UNK"]
        self.words_path = words_path
        self.vocab2index_path = vocab2index_path
        # load word dict and vocab2index if already saved
        if exists(words_path) and exists(vocab2index_path):
            self.words = load_object(file_name=self.words_path)
            self.vocab2index = load_object(file_name=self.vocab2index_path)

    @staticmethod
    def tokenize(text):
        if not isinstance(text, str) and math.isnan(text):
            return []
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
        nopunct = regex.sub(" ", text.lower())
        return [token.text.strip() for token in spacy_en.tokenizer(nopunct)]

    def encode_sentence(self, text: str):
        text = remove_redundant_spaces_in_sentence(text)
        tokenized = TextPreprocessor.tokenize(text)
        encoded = np.zeros(self.encoding_max_length, dtype=int)
        enc1 = np.array([self.vocab2index.get(word, self.vocab2index["UNK"]) for word in tokenized])
        length = min(self.encoding_max_length, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded, length

    def decode_sentence(self, encoded: np.ndarray) -> str:
        return " ".join([self.words[int(idx)] for idx in encoded])

    def text_transform(self, texts) -> np.ndarray:
        """ transforms all text data to encoded version.

        @param texts: narratives and dialogs
        @return: encoded version of concatanation of all text
        """
        merged_text = ""
        for text in texts:
            for sentence in text:
                if not isinstance(sentence, str) and math.isnan(sentence):
                    continue
                merged_text += (" " + sentence)
        return self.encode_sentence(merged_text)[0]

    def create_vocabulary(self,
                          save: bool = True,
                          force_reload: bool = False):
        """
        Creates vocabulary for dataset.
        EmoRecCom training takes 17 seconds on my PC. (@gsoykan)
        """
        if len(self.words) > 3 \
                and len(self.vocab2index) > 3 \
                and force_reload is False:
            return
        self.vocab2index = {'': self.empty_token_position, "UNK": self.unk_token_position}
        self.words = ['', "UNK"]
        # count number of occurences of each word
        counts = Counter()
        for item in tqdm(iter(self.dataset)):
            _, _, _, (narrative, dialog) = item
            for d in dialog:
                counts.update(TextPreprocessor.tokenize(d))
            for n in narrative:
                counts.update(TextPreprocessor.tokenize(n))

        # deleting infrequent words
        print("num_words before:", len(counts.keys()))
        for word in list(counts):
            if counts[word] < self.minimum_word_count:
                del counts[word]
        print("num_words after:", len(counts.keys()))

        # updating vocabulary
        for word in counts:
            self.vocab2index[word] = len(self.words)
            self.words.append(word)

        if save:
            save_object(self.words, self.words_path)
            save_object(self.vocab2index, self.vocab2index_path)


if __name__ == '__main__':
    emoreccom_path = "/home/gsoykan20/Desktop/datasets/multimodal_emotion_recognition_on_comics_scenes/"  # "/userfiles/comics_grp/multimodal_emotion_recognition_on_comics_scenes/"
    dataset = EmoRecComDataset(emoreccom_path, train=True)
    text_preprocessor = TextPreprocessor(dataset)
    text_preprocessor.create_vocabulary()
    save_object(text_preprocessor.words, file_name="emoreccom_word.pickle")
    save_object(text_preprocessor.vocab2index, file_name="emoreccom_vocab2index.pickle")
    sample_text = "wow !! i ' ve never liked fighting ray . but knock their bloomin ' blocks"
    encoded, encoded_length = text_preprocessor.encode_sentence(sample_text)
    decoded = text_preprocessor.decode_sentence(encoded=encoded)
    print(encoded)
