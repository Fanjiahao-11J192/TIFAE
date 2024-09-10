import copy
from Transformer.transformer import *

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, layers, n_blocks, embed_dim, input_dim,attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.input_dim =input_dim

        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))

    def get_encoder(self, layers):
        all_layers = [] #[256,128,64]
        input_dim = self.input_dim #384
        for i in range(0, len(layers)):
            all_layers.append(TransformerEncoderLayer())
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)

    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer) - 2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i + 1]))
            all_layers.append(nn.ReLU())  # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        # x_out = transformer_fusion
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            x_in = x_in + x_out
            # x_in = x_in + transformer_fusion
            # x_in = transformer_fusion + x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in + x_out), latents