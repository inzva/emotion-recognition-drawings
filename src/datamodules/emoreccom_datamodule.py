from typing import Optional, Tuple
from functools import partial

import transformers
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.datamodules.transform_datamodule import ValTransform, \
    SlightTransform, MediumTransform, FullTransform
from src.datamodules.datasets.dataset_modality import DatasetModality
from src.datamodules.datasets.emoreccom import EmoRecComDataset
from src.utils.additional_info_extractor_for_dataset import AdditionalInfoExtractorForDataset
from src.utils.emoreccom_label_transforms import normalize_and_take_top_n
from src.utils.text.elmo_embedder import ElmoTextEmbedder
from src.utils.text.text_preprocessor import TextPreprocessor
from src.utils.text.text_utils import text_transform_for_tokenizer, \
    text_transform_for_elmo_embeddings_to_character_indexes


class EmoRecComDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            modality: DatasetModality = DatasetModality.VisionAndText,
            # Activate If you are using BERT Tokenizer
            use_tokenizer_instead_text_preprocessor: bool = True,
            tokenizer_name: str = "squeezebert/squeezebert-uncased",
            tokenizer_get_token_type_ids: Optional[bool] = None,
            # Use Elmo Embeddings, Note that use_tokenizer_instead_text_preprocessor and tokens are exclusive to each other.
            use_elmo_tokens: bool = False,
            tokenizer_max_len: int = 100,
            use_label_transform: bool = False,
            damp_labels_if_text_is_empty: bool = False,
            # Train dataset length 6112
            train_val_test_split: Tuple[int, int, int] = (5112, 500, 500),
            text_encoding_max_length: int = 120,
            use_private_test_set: bool = False,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            # 0 for no augmentation, 1 for slight, 2 for medium and 3 for full, 4 for old
            # check datamodule > transform_datamodule.py for details
            image_augment_strength: int = 0,
            # When provided it injects additional info extractor to Dataset. Hence
            #  creates new data for batches
            additional_info_extractor: Optional[int] = None,
            face_body_embedding_base_path: Optional[str] = None,
            face_body_embedding_max_seq_len: Optional[int] = None,
            use_clip = False
    ):
        super().__init__()
        assert image_augment_strength in [0, 1, 2, 3, 4]
        # Handling Additional Info Extractor
        if additional_info_extractor:
            additional_info_extractor = AdditionalInfoExtractorForDataset(
                additional_info_extractor)
            if additional_info_extractor is AdditionalInfoExtractorForDataset.FaceBodyEmbeddings:
                self.additional_info_extractor = lambda element_name, train: additional_info_extractor. \
                    get_info(element_name,
                             train,
                             face_body_embedding_max_seq_len,
                             face_body_embedding_base_path)
            else:
                raise Exception(
                    "Unknown additional_info_extractor for EmoRecCom DataModule"
                )
        else:
            self.additional_info_extractor = None

        self.use_tokenizer_instead_text_preprocessor = use_tokenizer_instead_text_preprocessor
        self.tokenizer_name = tokenizer_name
        self.tokenizer_max_len = tokenizer_max_len
        self.tokenizer_get_token_type_ids = tokenizer_get_token_type_ids
        self.use_elmo_tokens = use_elmo_tokens
        self.data_dir = data_dir
        self.modality = modality
        self.use_private_test_set = use_private_test_set
        self.train_val_test_split = train_val_test_split
        self.text_encoding_max_length = text_encoding_max_length
        self.damp_labels_if_text_is_empty = damp_labels_if_text_is_empty
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_label_transform = use_label_transform

        image_augment_types = {
            0: ValTransform,
            1: SlightTransform,
            2: MediumTransform,
            3: FullTransform,
            4: transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,)),
                 transforms.Resize((224, 224))
                 ]
            )
        }

        self.vision_transform = image_augment_types[image_augment_strength]
        if image_augment_strength < 4:
            self.vision_transform = self.vision_transform()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.text_transform = None
        self.label_transform = None

        self.use_clip = use_clip

    @property
    def num_classes(self) -> int:
        return 8

    def prepare_data(self):
        self.label_transform = normalize_and_take_top_n if self.use_label_transform else None
        if self.use_tokenizer_instead_text_preprocessor:
            tokenizer = self.get_tokenizer()
            self.tokenizer_max_len = self.tokenizer_max_len
            tokenizer_func = partial(tokenizer,
                                     text_pair=None,
                                     add_special_tokens=True,
                                     max_length=self.tokenizer_max_len,
                                     padding="max_length",
                                     truncation=True,
                                     )
            self.text_transform = lambda texts: text_transform_for_tokenizer(tokenizer_func,
                                                                             texts,
                                                                             self.tokenizer_get_token_type_ids)
        elif self.use_elmo_tokens:
            tokenizer_func = partial(ElmoTextEmbedder.find_character_indexes,
                                     max_sentence_length=self.tokenizer_max_len)
            self.text_transform = lambda texts: text_transform_for_elmo_embeddings_to_character_indexes(tokenizer_func,
                                                                                                        texts)
        elif self.use_clip:
            self.text_transform=None
            dataset = EmoRecComDataset(self.data_dir, train=True, plain_text=self.use_clip, text_transform=self.text_transform)
                
        else:
            dataset = EmoRecComDataset(self.data_dir, train=True)
            text_preprocessor = TextPreprocessor(dataset,
                                                 encoding_max_length=self.text_encoding_max_length)
            text_preprocessor.create_vocabulary()
            self.text_transform = text_preprocessor.text_transform

    def get_tokenizer(self):
        if self.tokenizer_name == "squeezebert/squeezebert-uncased":
            tokenizer = transformers. \
                SqueezeBertTokenizer. \
                from_pretrained(self.tokenizer_name,
                                do_lower_case=True)
        elif self.tokenizer_name == "bert-base-uncased":
            tokenizer = transformers. \
                BertTokenizer. \
                from_pretrained("bert-base-uncased",
                                do_lower_case=True)
        else:
            raise Exception(
                "Unknown tokenizer_name for GoEmotionsDataModule"
            )
        return tokenizer

    def setup(self, stage: Optional[str] = None):
        if not self.data_train or not self.data_val or not self.data_test:
            dataset = partial(EmoRecComDataset,
                              self.data_dir,
                              train=not self.use_private_test_set,
                              modality=self.modality,
                              text_transform=self.text_transform,
                              vision_transform=self.vision_transform,
                              label_transform=self.label_transform,
                              damp_labels_if_text_is_empty=self.damp_labels_if_text_is_empty,
                              additional_info_extractor=self.additional_info_extractor,
                              plain_text=self.use_clip)
            self.data_train = dataset(specific_slice=slice(0, self.train_val_test_split[0]))
            self.data_val = dataset(specific_slice=slice(self.train_val_test_split[0],
                                                         self.train_val_test_split[0] + self.train_val_test_split[1]))
            self.data_test = dataset(specific_slice=slice(self.train_val_test_split[0] + self.train_val_test_split[1],
                                                          self.train_val_test_split[0] + self.train_val_test_split[1] +
                                                          self.train_val_test_split[2]))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == '__main__':
    use_transformer_tokenizer = True
    emoreccom_path = "/home/gsoykan20/Desktop/datasets/multimodal_emotion_recognition_on_comics_scenes/"  # "/userfiles/comics_grp/multimodal_emotion_recognition_on_comics_scenes/"
    datamodule = EmoRecComDataModule(data_dir=emoreccom_path,
                                     modality=DatasetModality.Text,
                                     use_tokenizer_instead_text_preprocessor=use_transformer_tokenizer)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = iter(datamodule.train_dataloader())
    batch = next(train_dataloader)  # img, img_info, (hard labels, polarities), text encodings
    print(batch)
