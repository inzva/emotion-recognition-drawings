from typing import Optional
import transformers
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

from src.datamodules.datasets.goemotions import GoEmotions


class GoEmotionsDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer_name: str = "squeezebert/squeezebert-uncased",
                 tokenizer_max_len: int = 40,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False
                 ):
        super().__init__()
        if tokenizer_name == "squeezebert/squeezebert-uncased":
            self.tokenizer = transformers. \
                SqueezeBertTokenizer. \
                from_pretrained(tokenizer_name,
                                do_lower_case=True)
        else:
            raise Exception(
                "Unknown tokenizer_name for GoEmotionsDataModule"
            )
        self.tokenizer_max_len = tokenizer_max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data = None

    @property
    def num_classes(self) -> int:
        return 28  # 27 + 1(neutral)

    def prepare_data(self) -> None:
        go_emotions = load_dataset("go_emotions")
        self.data = go_emotions.data

    def setup(self, stage: Optional[str] = None):
        train = self.data["train"].to_pandas()
        valid = self.data["validation"].to_pandas()
        test = self.data["test"].to_pandas()
        print(train.shape, valid.shape, test.shape)

        # (43410, 3) (5426, 3) (5427, 3)
        def one_hot_encoder(df):
            one_hot_encoding = []
            for i in tqdm(range(len(df))):
                temp = [0] * self.num_classes
                label_indices = df.iloc[i]["labels"]
                for index in label_indices:
                    temp[index] = 1
                one_hot_encoding.append(temp)
            return pd.DataFrame(one_hot_encoding)

        train_ohe_labels = one_hot_encoder(train)
        valid_ohe_labels = one_hot_encoder(valid)
        test_ohe_labels = one_hot_encoder(test)
        train = pd.concat([train, train_ohe_labels], axis=1)
        valid = pd.concat([valid, valid_ohe_labels], axis=1)
        test = pd.concat([test, test_ohe_labels], axis=1)
        self.data_train = GoEmotions(train.text.tolist(),
                                     train[range(self.num_classes)].values.tolist(),
                                     self.tokenizer,
                                     self.tokenizer_max_len)
        self.data_val = GoEmotions(valid.text.tolist(),
                                   valid[range(self.num_classes)].values.tolist(),
                                   self.tokenizer,
                                   self.tokenizer_max_len)
        self.data_test = GoEmotions(test.text.tolist(),
                                    test[range(self.num_classes)].values.tolist(),
                                    self.tokenizer,
                                    self.tokenizer_max_len)

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
    datamodule = GoEmotionsDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = iter(datamodule.train_dataloader())
    batch = next(train_dataloader)  # img, img_info, (hard labels, polarities), text encodings
    print(batch)
