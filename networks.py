import torch
import torch.nn as nn
import torch.nn.functional as torchfunc


# implements the concatenated relu activation function
class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x,-x),1)
        return torchfunc.relu(x)

# Note their default cnn_embed_dim is 50
class Encoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=100):
        super(Encoder, self).__init__()
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # 3x64x64
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64, stride=2, kernel_size=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # 64x31x31
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        # 128x14x14
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=4),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        # 256x6x6
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, stride=2, kernel_size=4),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        # 512x2x2
        self.fullyconnected1 = nn.Sequential(
            nn.Linear(512*2*2, 256),
            nn.BatchNorm1d(256),
            CReLU()
        )
        # 512
        self.fullyconnected2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            CReLU()
        )
        self.stylemulayer = nn.Linear(512, self.CNN_embed_dim)
        self.contentmulayer = nn.Linear(512, self.CNN_embed_dim)
        self.stylelogvarlayer = nn.Sequential(
            nn.Linear(512, self.CNN_embed_dim),
            nn.Tanh()
        )
        self.contentlogvarlayer = nn.Sequential(
            nn.Linear(512, self.CNN_embed_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.flatten(start_dim=1)
        x = self.fullyconnected1(x)
        x = self.fullyconnected2(x)

        # The 5x multiplier is from the paper. No clue why
        style_mu, style_logvar = self.stylemulayer(x), 5*self.stylelogvarlayer(x)
        content_mu, content_logvar = self.contentmulayer(x), 5*self.contentlogvarlayer(x)

        return style_mu, style_logvar, content_mu, content_logvar


# This is messed up, but their instructions don't actually describe a network that works
class Decoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=100):
        super(Decoder, self).__init__()
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        self.deconvblock1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*CNN_embed_dim, out_channels=256, stride=2, kernel_size=4, output_padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.deconvblock2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, stride=2, kernel_size=4, output_padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.deconvblock3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, output_padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        self.deconvblock4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=4, output_padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.Sigmoid()
        )

        self.deconvblockmu = nn.ConvTranspose2d(in_channels=64, out_channels=3, stride=1, kernel_size=4)

        self.deconvlogvar = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=3, stride=1, kernel_size=3),
            nn.Tanh()
        )

    def forward(self, style_z, content_z):
        x = torch.cat((style_z, content_z), dim=1)
        x = x.view(-1, self.CNN_embed_dim*2, 1,1)
        x = self.deconvblock1(x)
        x = self.deconvblock2(x)
        x = self.deconvblock3(x)
        x = self.deconvblock4(x)

        output_mu = self.deconvblockmu(x)
        output_logvar = 5*self.deconvlogvar(x)

        return output_mu


def main():
    test_encoder_input = torch.zeros((16,3,64,64))
    test_decoder_input = torch.zeros((16,100))
    test_decoder_input2 = torch.zeros((16,100))
    test_encoder = Encoder(CNN_embed_dim=100)
    test_decoder = Decoder(CNN_embed_dim=100)

    print(test_encoder(test_encoder_input)[0].shape)
    print(test_decoder(test_decoder_input, test_decoder_input2).shape)

if __name__=="__main__":
    main()