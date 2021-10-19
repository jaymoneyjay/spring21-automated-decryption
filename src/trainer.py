""" Specifies an interface for code efficient training and testing of pytorch models. """

import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, nb_epochs):
        """Create a trainer by specifying the number of epochs to train
        Args:
            nb_epochs: int. Number of epochs to train
        """

        self.nb_epochs = nb_epochs
        self.logs = {}

    def fit(self, model, dl_train, dl_val, verbose=True, thresh=5e-2):
        """Train the model on the specified data and print the training and validation loss and accuracy.
        Args:
            model: Module. Model to train
            dl_train: DataLoader. DataLoader containing the training data
            dl_val: DataLoader. DataLoader containing the validation data
            verbose: bool. Whether or not to output training information
            thresh: float. Threshold to test convergence of the model
        """

        optimizer = model.configure_optimizers()
        train_loss_epochs = []
        val_loss_epochs = []

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)

        for e in range(self.nb_epochs):
            loss_train = []
            for batch_idx, batch in enumerate(tqdm(dl_train)):
                model.train()
                optimizer.zero_grad()
                loss = model.training_step(batch, batch_idx, device)
                loss.backward()
                optimizer.step()

                loss_train.append(loss.item())

            loss_val = []

            for batch_idx, batch in enumerate(dl_val):
                model.eval()
                with torch.no_grad():
                    loss = model.validation_step(batch, batch_idx, device)
                    loss_val.append(loss.item())
            avg_loss_train = round(sum(loss_train) / len(loss_train), 2)
            train_loss_epochs.append(avg_loss_train)

            avg_loss_val = round(sum(loss_val) / len(loss_val), 2)
            val_loss_epochs.append(avg_loss_val)

            if verbose:
                print(
                    f"# Epoch {e+1}/{self.nb_epochs}:\t loss={avg_loss_train}\t loss_val={avg_loss_val}"
                )

            if (
                len(train_loss_epochs) > 1
                and abs(train_loss_epochs[e] - train_loss_epochs[e - 1]) < thresh
            ):
                return train_loss_epochs, val_loss_epochs

        return train_loss_epochs, val_loss_epochs

    def score(self, model, dl, measure):
        """Measure performance of model"""
        
        predictions, labels = self.predict(model, dl)

        return measure(labels, predictions)

    def predict(self, model, dl):
        """Predict scores for data in dl"""
        
        predictions, labels = [], []

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)

        for batch_idx, batch in enumerate(tqdm(dl)):
            model.eval()
            with torch.no_grad():
                p, l = model.predict_step(batch, batch_idx, device)
                p = p.cpu().detach().tolist()
                l = l.cpu().detach().tolist()
                predictions.extend(p)
                labels.extend(l)

        return predictions, labels

    def _log(self, key, value):
        """ Log a performance metric """
        
        self.logs[key] = value

    def get_metric(key):
        """ Retrieve a logged performance metric """
        
        return self.logs[key]
