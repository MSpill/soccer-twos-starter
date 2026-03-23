import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel():
    def __init__(self, weights_path: str):
        """
        A fully connected neural network with two hidden layers.

        Parameters
        ----------
        state_size (int): Observation dimension
        action_size (int): Action dimension
        seed (int): random seed
        """
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        hidden_wt = weights['_hidden_layers.0._model.0.weight']
        hidden_bias = weights['_hidden_layers.0._model.0.bias']
        logit_wt = weights['_logits._model.0.weight']
        logit_bias = weights['_logits._model.0.bias']
        num_hidden = hidden_wt.shape[0]
        print(f'num hidden is {num_hidden}')
        self.hidden = nn.Linear(336, num_hidden)
        self.logit = nn.Linear(num_hidden, 27)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(hidden_wt))
            self.hidden.bias.copy_(torch.from_numpy(hidden_bias))
            self.logit.weight.copy_(torch.from_numpy(logit_wt))
            self.logit.bias.copy_(torch.from_numpy(logit_bias))

    def get_action(self, x):
        """Forward pass"""
        with torch.no_grad():
            x = torch.from_numpy(x)
            logits = self.logit(F.tanh(self.hidden(x)))
            action = logits.argmax().item()
            return action

if __name__ == '__main__':
    model = MyModel("striker_agent/weights/goalie.pkl")