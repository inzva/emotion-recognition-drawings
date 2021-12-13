goemotions_emotion_list = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral"
]

emoreccom_goemotions_emotion_mapping = {
    "happy": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration",
              "gratitude", "relief", "approval"],
    "angry": ["anger", "annoyance"],
    "disgust": ["disappointment",
                "disapproval",
                "disgust",
                "embarrassment"],
    "fear": [
        "fear",
        "grief",
        "confusion",
    ],
    "sad": ["remorse",
            "sadness",
            "nervousness",
            ],
    "surprise": [
        "realization",
        "curiosity",
        "surprise",
    ],
    "neutral": ["neutral"],
    # from @gsoykan, could not think of an emotion to put into "other" category
    # if the model(BERT trained on GoEmotions) output is empty then the result can be other
    "other": []
}
