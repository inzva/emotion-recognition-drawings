import torch
from allennlp.modules.elmo import Elmo, batch_to_ids


class ElmoTextEmbedder(torch.nn.Module):
    def __init__(self,
                 options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                 weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"):
        """
        source: https://www.reddit.com/r/LanguageTechnology/comments/b2n5jn/how_to_get_sentence_embeddings_from_elmo/
        default options returns embeddings of size: [B, T_MAX, 1024]
        @rtype: object
        """
        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        super().__init__()
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)

    @staticmethod
    def find_character_indexes(sentences,
                               max_sentence_length: int):
        # use batch_to_ids to convert sentences to character ids
        if not isinstance(sentences, list):
            sentences = [sentences]
        sentences = [d.split() for d in sentences]
        character_ids = batch_to_ids(sentences)
        if character_ids.shape == torch.Size([1, 0]):
            # let's assume word length is 50
            character_ids = torch.zeros([1, max_sentence_length, 50], dtype=torch.int64)
        b, sentence_length, word_length = character_ids.shape
        sentence_length_diff = max_sentence_length - sentence_length
        alternative_character_ids = torch.zeros([b, max_sentence_length, word_length], dtype=torch.int64)
        if sentence_length_diff > 0:
            alternative_character_ids[:, :sentence_length, :] = character_ids
        elif sentence_length_diff < 0:
            alternative_character_ids[:, :, :] = character_ids[:, :max_sentence_length, :]
        else:
            alternative_character_ids = character_ids
        return alternative_character_ids

    def forward(self, character_ids):
        return self.elmo(character_ids)['elmo_representations'][0]


if __name__ == '__main__':
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo_embedder = ElmoTextEmbedder(options_file, weight_file)
    sentences = ["ELMo loves you so mucchhh", "ddf df d", "let's see what that means", "oh come on!",
                 "this is a very very looooooooooooooooooooooooooong word"]
    embeddings = elmo_embedder.get_embeddings(sentences, max_sentence_length=100)
    print(embeddings)
