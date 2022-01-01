import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset
from src.datamodules.datasets.dataset_modality import DatasetModality
from src.utils.text.text_utils import merge_comic_texts

class EmoRecComDataset(Dataset):
    def __init__(
            self,
            data_dir=None,
            train=True,
            modality: DatasetModality = DatasetModality.VisionAndText,
            text_transform=None,
            vision_transform=None,
            label_transform=None,
            specific_slice=None,
            damp_labels_if_text_is_empty: bool = False,
            additional_info_extractor=None,
            plain_text = False
    ):
        """EmoRecCom pytorch dataset
        @param data_dir: Directory of the dataset.
        @param train: If false, the dataset is in test mode and
            test mode does not have ground truth labels so it returns all 0's.
        @param modality: Dataset modality.
            It can take following values => DatasetModality.VisionAndText,
                                            DatasetModality.Vision,
                                            DatasetModality.Text
        @param text_transform: transform function for text data,
            usually this is encoding from raw text to token indexes.
        @param text_transform: transform function for vision data(image),
            usually this is includes normalization and various augmentations.
        @param specific_slice: Optional. If provided it slices dataset to provided value.
        @param damp_labels_if_text_is_empty: bool. There are instances with empty 
        text inputs. They should not indicate any labels as 1. If this is true, labels of
        such instances will be all zeroes.
        @param additional_info_extractor: when provided this should be function that takes
        filename & train for the index and returns tensor transformable additional information
        to the element, e.g. face_body embeddings extracted from YOLOX_FB model.
        """
        if isinstance(modality, int):
            modality = DatasetModality(modality)
        # Fixing label transform options to enum if they are int's.
        super().__init__()
        self.train = train
        self.modality = modality
        self.damp_labels_if_text_is_empty = damp_labels_if_text_is_empty
        self.specific_slice = specific_slice
        self.text_transform = text_transform
        self.vision_transform = vision_transform
        self.label_transform = label_transform
        self.additional_info_extractor = additional_info_extractor
        self.emotion_dict = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "sad": 4,
            "surprise": 5,
            "neutral": 6,
            "other": 7
        }
        self.emotion_list = list(self.emotion_dict.keys())
        self.files, self.annotations = self.load_annotations(data_dir, train)
        if specific_slice is not None:
            self.files = self.files[specific_slice]

        self.plain_text = plain_text

    def __len__(self):
        return len(self.files)

    def pull_item(self, index):
        img, img_info, labels, texts = [], [], [[], []], [[], []]
        if self.modality in [DatasetModality.VisionAndText]:
            img, img_info = self.load_image(index)
            labels, texts = self.load_anno(index)
        elif self.modality in [DatasetModality.Vision]:
            img, img_info = self.load_image(index)
            labels, _ = self.load_anno(index)
        elif self.modality in [DatasetModality.Text]:
            labels, texts = self.load_anno(index)
        elif self.modality in [DatasetModality.TextAndFaceBodyEmbeddings]:
            labels, texts = self.load_anno(index)
        return img, img_info, labels, texts

    def __getitem__(self, index):
        img, img_info, labels, texts = self.pull_item(index)
        if self.damp_labels_if_text_is_empty and \
                not texts[0] and \
                not texts[1]:
            labels[0] = np.zeros_like(labels[0])
        if self.modality in [DatasetModality.VisionAndText, DatasetModality.Vision] \
                and self.vision_transform is not None:
            img = self.vision_transform(img)
        if self.modality in [DatasetModality.VisionAndText,
                             DatasetModality.Text,
                             DatasetModality.TextAndFaceBodyEmbeddings] \
                and self.text_transform is not None:
            texts = self.text_transform(texts)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        if self.additional_info_extractor:
            return img, img_info, labels, texts, self.additional_info_extractor(self.files[index], self.train)
        if self.plain_text == True:
            texts = merge_comic_texts(texts=texts)
            return img, img_info, labels, texts
        return img, img_info, labels, texts

    def load_anno(self, index):
        # given an index, it loads the annotations of the file at that index
        file = self.files[index]
        annots = self.annotations[file]

        label = annots["labels"]
        polarity = annots["polarity"]
        narrative = annots["narrative"]
        dialog = annots["dialog"]

        return [label, polarity], [narrative, dialog]

    def load_image(self, index):
        img_path = os.path.join(self.annotations["root_dir"], self.files[index])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None
        h, w, c = img.shape
        return img, [h, w]

    def load_annotations(self, root_path, train: bool):

        # root_path is the path where emoreccom dataset root is.
        # The rest of the paths should be the same.
        if train:
            imgs_path = os.path.join(root_path, "public_train", "train")
            labels_path = os.path.join(root_path, "public_train", "train_emotion_labels.csv")
            polarity_path = os.path.join(root_path, "public_train", "train_emotion_polarity.csv")
            texts_path = os.path.join(root_path, "public_train", "train_transcriptions.json")
        else:
            imgs_path = os.path.join(root_path, "public_train", "test")
            labels_path = None
            polarity_path = None
            texts_path = os.path.join(root_path, "public_train", "test_transcriptions.json")

        # Files will keep the list of files available in the whole dataset partition
        files = os.listdir(imgs_path)

        # Annotation will include the labels, dialog, narrative and polarity information.
        # The key will be the filename, the values will be the ones listed above.
        annotations = {"root_dir": imgs_path}
        for file in files:
            annotations[file] = {
                "labels": np.zeros(len(self.emotion_list)),
                "polarity": np.zeros(len(self.emotion_list)),
                "narrative": [],
                "dialog": []
            }

        # Reading the text data
        with open(texts_path, "r") as f:
            texts = json.load(f)

        for text in texts:
            filename = text["img_id"] + ".jpg"
            dialog = text["dialog"]
            narrative = text["narration"]

            assert filename in annotations

            annotations[filename]["narrative"] = narrative
            annotations[filename]["dialog"] = dialog

        # Reading the labels 
        if labels_path is not None:

            with open(labels_path, "r") as f:
                lines = f.readlines()

            # first extract the emotion order in the CSV file to correctly assign it
            titles = "index" + lines[0]
            if titles[-1] == "\n":
                titles = titles[:-1]
            titles = titles.split(",")
            emotions = titles[2:]

            # for each new line in the emotions CSV file
            for line in lines[1:]:
                if len(line) < 3:
                    # avoids any additional line ends and skips
                    continue
                elif line[-1] == "\n":
                    line = line[:-1]
                line = line.split(",")
                # get the filename and check if it is in the annotations dict
                img_id = line[1] + ".jpg"
                assert img_id in annotations

                # for each emotion in the line we read
                for i, one_hot in enumerate(line[2:]):
                    binary = float(one_hot)
                    # if the emotion is present as 1.0 (bigger than 0)
                    if binary > 0.0:
                        # get the emotion and give its value to the annotation
                        emotion = emotions[i]
                        annotations[img_id]["labels"][self.emotion_dict[emotion]] = binary

        # Reading the polarity values
        if polarity_path is not None:
            with open(polarity_path, "r") as f:
                lines = f.readlines()

            # for each new line in the polarity CSV file
            for line in lines[1:]:
                if len(line) < 3:
                    # avoids any additional line ends and skips
                    continue
                elif line[-1] == "\n":
                    line = line[:-1]
                line = line.split(",", 2)
                # get the filename and check if it is in the annotations dict
                img_id = line[1] + ".jpg"
                assert img_id in annotations

                if line[2][0] == '"':
                    line[2] = line[2][1:-1]

                polarity_dict = json.loads(line[2].replace("'", '"'))

                for emotion in polarity_dict.keys():
                    annotations[img_id]["polarity"][self.emotion_dict[emotion]] = polarity_dict[emotion]

        return files, annotations
