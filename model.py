import torch.nn as nn
import torch

class FeedforwardNetworks(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(FeedforwardNetworks, self).__init__()
        embed_dim = embed_dim * 2 + 7
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4_ratio = nn.Linear(128, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img_encoding, text_encoding, one_hot):
        x = torch.cat((text_encoding, img_encoding, one_hot), dim=1)
        x = self.relu(self.fc1(self.norm(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        ratio_prediction = self.fc4_ratio(x)
        ratio_prediction = self.softmax(ratio_prediction) * 100
        return ratio_prediction

class SalesPredictionModel(nn.Module):
    def __init__(self, embed_dim):
        super(SalesPredictionModel, self).__init__()
        self.feedforwardnetworks = FeedforwardNetworks(embed_dim)

    def forward(self, text_embed, image_embed, one_hot):
        ratio_output = self.feedforwardnetworks(text_embed, image_embed, one_hot)
        return ratio_output
