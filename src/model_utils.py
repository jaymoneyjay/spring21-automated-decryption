""" Helper functions to load and store pytorch models. """

import torch

from pathlib import Path
from src.models import CharCNN


def load_model(model_name, len_alphabet):
    """ Load the specified pytorch model """
    
    model = CharCNN(len_alphabet)

    if not torch.cuda.is_available():
        state = torch.load(model_name, map_location=torch.device("cpu"))

    else:
        state = torch.load(model_name)

    model.load_state_dict(state)

    return model


def save_model(model, fpath):
    """ Save the specified pytorch model """
    
    torch.save(model.state_dict(), fpath)
