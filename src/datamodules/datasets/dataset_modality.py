import enum


class DatasetModality(enum.Enum):
    Text = 1
    Vision = 2
    VisionAndText = 3
    TextAndFaceBodyEmbeddings = 4
