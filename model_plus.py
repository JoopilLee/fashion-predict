import torch.nn as nn
import torch

class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, dropout=0.2):
        super(FusionNetwork, self).__init__()
        self.use_img = use_img
        self.use_text = use_text
        input_dim = embedding_dim * 2 + 7 
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, img_encoding, text_encoding, one_hot):
        decoder_inputs = []
        decoder_inputs.append(img_encoding)
        decoder_inputs.append(text_encoding)
        decoder_inputs.append(one_hot) 
        concat_features = torch.cat(decoder_inputs, dim=1)
        final = self.feature_fusion(concat_features)
        return final

class FeedforwardNetworks(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(FeedforwardNetworks, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4_ratio = nn.Linear(128, 7)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, final):
        x = self.norm(final)
        x = self.relu(self.fc1(x))
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
        hidden_dim = embed_dim * 2 + 7
        self.fusionnetwork = FusionNetwork(embed_dim, hidden_dim, 1, 1, dropout=0.2)
        self.feedforwardnetworks = FeedforwardNetworks(hidden_dim) 

    def forward(self, text_embed, image_embed, one_hot):
        x = self.fusionnetwork(text_embed, image_embed, one_hot)
        ratio_output = self.feedforwardnetworks(x)
        return ratio_output