from typing import Optional, Tuple
from functools import partial

import transformers
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.datamodules.datasets.dataset_modality import DatasetModality
from src.datamodules.datasets.emoreccom_bb import EmoRecComBBDataset
from src.utils.text.text_preprocessor import TextPreprocessor
from src.utils.text.text_utils import text_transform_for_tokenizer


class EmoRecComBBDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            modality: DatasetModality = DatasetModality.VisionAndText,
            use_tokenizer_instead_text_preprocessor: bool = True,
            tokenizer_name: str = "squeezebert/squeezebert-uncased",
            tokenizer_max_len: int = 100,
            # Train dataset length 6112
            train_val_test_split: Tuple[int, int, int] = (5112, 500, 500),
            text_encoding_max_length: int = 120,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            bb_path: str = "selected_bbox.json",
            person_bb: bool =True
    ):
        super().__init__()
        self.use_tokenizer_instead_text_preprocessor = use_tokenizer_instead_text_preprocessor
        self.tokenizer_name = tokenizer_name
        self.tokenizer_max_len = tokenizer_max_len
        self.data_dir = data_dir
        self.modality = modality
        self.train_val_test_split = train_val_test_split
        self.text_encoding_max_length = text_encoding_max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.vision_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((300,300))]
        )
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.text_transform = None
        self.bb_path = bb_path
        self.person_bb = person_bb


    @property
    def num_classes(self) -> int:
        return 8

    def prepare_data(self):
        if self.use_tokenizer_instead_text_preprocessor:
            if self.tokenizer_name == "squeezebert/squeezebert-uncased":
                tokenizer = transformers. \
                    SqueezeBertTokenizer. \
                    from_pretrained(self.tokenizer_name,
                                    do_lower_case=True)
            else:
                raise Exception(
                    "Unknown tokenizer_name for GoEmotionsDataModule"
                )

            self.tokenizer_max_len = self.tokenizer_max_len
            tokenizer_func = partial(tokenizer,
                                     text_pair=None,
                                     add_special_tokens=True,
                                     max_length=self.tokenizer_max_len,
                                     padding="max_length",
                                     truncation=True,
                                     )
            self.text_transform = lambda texts: text_transform_for_tokenizer(tokenizer_func, texts)
        else:
            dataset = EmoRecComBBDataset(self.data_dir, train=True, bb_path=self.bb_path, person_bb=self.person_bb )
            text_preprocessor = TextPreprocessor(dataset,
                                                 encoding_max_length=self.text_encoding_max_length)
            text_preprocessor.create_vocabulary()
            self.text_transform = text_preprocessor.text_transform

    def setup(self, stage: Optional[str] = None):
        if not self.data_train or not self.data_val or not self.data_test:
            dataset = partial(EmoRecComBBDataset, self.data_dir,
                              train=True,
                              modality=self.modality,
                              text_transform=self.text_transform,
                              vision_transform=self.vision_transform)
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
    datamodule = EmoRecComBBDataModule(data_dir=emoreccom_path,
                                     modality=DatasetModality.Text,
                                     use_tokenizer_instead_text_preprocessor=use_transformer_tokenizer)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = iter(datamodule.train_dataloader())
    batch = next(train_dataloader)  # img, img_info, (hard labels, polarities), text encodings
    print(batch)
