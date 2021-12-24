import enum
import numpy as np
import torch


class AdditionalInfoExtractorForDataset(enum.Enum):
    FaceBodyEmbeddings = 1

    @classmethod
    def get_info(cls,
                 element_name: str,
                 train: bool,
                 max_seq_len: int = 8,
                 base_embedding_path: str = "/home/gsoykan20/Desktop/self_development/emotion-recognition-drawings/data/emoreccom_face_body_embeddings_96d/",
                 embedding_dim: int = 96
                 ):
        if train:
            base_embedding_path += "train/"
        else:
            base_embedding_path += "test/"
        embedding_path = base_embedding_path + element_name + ".npy"
        lean_face_body_embedding = np.load(embedding_path)
        if len(lean_face_body_embedding) == 0:
            lean_face_body_embedding = np.zeros((0, embedding_dim))
        seq_len, current_embedding_dim = lean_face_body_embedding.shape
        assert current_embedding_dim == embedding_dim
        face_body_embedding_for_batch = np.zeros((max_seq_len, embedding_dim))
        filled_until_index = min(max_seq_len, seq_len)
        face_body_embedding_for_batch[:filled_until_index, :] = lean_face_body_embedding[:filled_until_index, :]
        visual_embeds = torch.from_numpy(face_body_embedding_for_batch.astype(np.float32))
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        visual_attention_mask[filled_until_index:] = 0
        return visual_embeds, visual_token_type_ids, visual_attention_mask


if __name__ == '__main__':
    info = AdditionalInfoExtractorForDataset.FaceBodyEmbeddings.get_info(element_name="37_8_5.jpg",
                                                                         train=True)
    print(info)
