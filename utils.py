import pandas as pd
import pickle


def get_encoder(k):
    """
    Get Label Encoder object for a given column
    Return Type - LabelEncoder object for the specified column
    """

    with open(f"encoders/{k}_encoder.pkl", "rb") as fp:
        encoder = pickle.load(fp)

    return encoder


def get_model():
    
    with open(f"models/classifier.pkl", "rb") as fp:
        model = pickle.load(fp)

    return model