""" Predictive models for text classification. """

from numpy import empty, argmax
from torch import nn
import numpy as np
import torch
from decouple import config
from torchmetrics import ConfusionMatrix, Accuracy
from src.data_utils import encode_msgs


class CharCNN(nn.Module):
    """

    A character-level CNN for text classification.
    This architecture is an implementation of Zhang et al., 'Character-level Convolutional Networks for Text Classification', NeurIPS, 2016

    """

    def __init__(
        self,
        alphabet_size,
        num_conv_filters=256,
        num_fc_filters=1024,
        loss=torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0], dtype=torch.float32)
        ),
    ):

        super(CharCNN, self).__init__()

        self.__name__ = "CharCNN"

        self.alphabet_size = alphabet_size

        self.max_seq_length = config("MAX_SEQ_LEN", cast=int)
        self.num_classes = config("N_CLASSES", cast=int)
        self.loss = loss

        self.num_conv_filters = num_conv_filters
        self.num_fc_filters = num_fc_filters
        self.conv_kernel_sizes = [7, 7, 3, 3, 3, 3]
        self.pool_kernel_sizes = [3, 3, None, None, None, 3]

        # Metrics
        self.accuracy = Accuracy()
        self.confusion = ConfusionMatrix(num_classes=self.num_classes, normalize="true")

        # Calculate output length of last conv. layer
        self.conv_seq_length = self._calculate_conv_seq_length()

        # Define convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.alphabet_size, num_conv_filters, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        # Define fully-connected output layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.conv_seq_length, num_fc_filters), nn.ReLU(), nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(num_fc_filters, num_fc_filters), nn.ReLU(), nn.Dropout(0.5)
        )

        self.fc_out = nn.Linear(num_fc_filters, self.num_classes)

        self._initialise_weights()

    def forward(self, x):
        """Forward pass"""

        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # Reshape
        x = x.view(x.size(0), -1)

        # Fully-connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)

        return x

    def _calculate_conv_seq_length(self):
        """Calculate number of units in output of last convolutional layer."""
        
        conv_seq_length = self.max_seq_length

        for fc, fp in zip(self.conv_kernel_sizes, self.pool_kernel_sizes):
            conv_seq_length = (conv_seq_length - fc) + 1

            if fp is not None:
                conv_seq_length = (conv_seq_length - fp) // fp + 1

        return conv_seq_length * self.num_conv_filters

    def _initialise_weights(self, mean=0.0, std=0.05):
        """Initialise weights with Gaussian distribution."""
        
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def predict(self, msg, alphabet, use_loss):
        """ Predict the class scores for the specified message """
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.eval()
        self.to(device)
        
        y0 = torch.Tensor([0]).long()
        y1 = torch.Tensor([1]).long()

        inp = encode_msgs([msg], alphabet)
        if torch.cuda.is_available():
            inp = inp.to(device)
            y0 = y0.to(device)
            y1 = y1.to(device)

        with torch.no_grad():
            log = self(inp)
            if use_loss:
                loss0 = -self.loss(log, y0)
                loss1 = -self.loss(log, y1)
                return torch.Tensor([loss0, loss1])
            else:
                p = torch.nn.functional.softmax(log, dim=1)
                return p

    def predict_all(self, msgs, alphabet, use_loss=False, batch_size=64):
        """ Predict the class scores for all specified messages. """
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.eval()
        self.to(device)

        inputs = encode_msgs(msgs, alphabet)

        probabilities = []
        losses = []
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction='none'
        )

        n_inputs = inputs.size(0)
        for start in range(0, n_inputs, batch_size):
            end = min(n_inputs, start + batch_size)
            batch_inp = inputs[start:end]
            
            y0 = torch.Tensor([0 for _ in range(start, end)]).long()
            y1 = torch.Tensor([1 for _ in range(start, end)]).long()

            if torch.cuda.is_available():
                batch_inp = batch_inp.to(device)
                y0 = y0.to(device)
                y1 = y1.to(device)

            with torch.no_grad():
                logits = self(batch_inp)
                
                if use_loss:
                    loss0 = -loss_fn(logits, y0)
                    loss1 = -loss_fn(logits, y1)
                    for l0, l1 in zip(loss0, loss1):
                        losses.append(torch.Tensor([l0.item(), l1.item()]).cpu().detach().numpy())
                else:
                    p = torch.nn.functional.softmax(logits, dim=1)
                    probabilities.extend(p)
        if use_loss:
            return losses
        else:
            probabilities = [p.cpu().detach().numpy() for p in probabilities]
            return probabilities
        
    def _get_labels(self, logits):
        """Get class labels from predicted logits."""
        
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        labels = torch.argmax(probabilities, 1)
        return labels

    ####
    # Methods to be used within the Trainer interface, see trainer.py
    ####

    def training_step(self, batch, batch_idx, device):
        """ Specifies the behaviour of a single train step. """
        
        x, y = batch

        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)

        out = self(x)
        loss = self.loss(out, y)
        return loss

    def validation_step(self, batch, batch_idx, device):
        """ Specifies the behaviour for a single validation step. """
        
        x, y = batch

        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)

        out = self(x)
        loss = self.loss(out, y)
        return loss

    def predict_step(self, batch, batch_idx, device):
        """ Specifies the behaviour for a single prediction step. """
        
        x, y = batch

        if torch.cuda.is_available():
            x = x.to(device)
            y = y.to(device)

        out = self(x)
        pred = self._get_labels(out)
        return pred, y

    def test_step(self, batch, batch_idx):
        """ Specifies the behaviour for a single test step. """
        
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """ Set the adam optimizer as optimizer. """
        
        return torch.optim.Adam(self.parameters(), lr=1e-3)
