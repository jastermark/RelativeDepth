from torch import nn
import torch


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except:
            pass
    return


# Define model
class RelScaleNet(nn.Module):
    def __init__(self, cls_range=None, width_multiplier=2):
        super(RelScaleNet, self).__init__()

        assert cls_range is None or isinstance(cls_range, (list, tuple)), "cls_range must be None or a list/tuple"
        self.cls_range = cls_range

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32*width_multiplier, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32*width_multiplier, 32*width_multiplier, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32*width_multiplier, 64*width_multiplier, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64*width_multiplier, 64*width_multiplier, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64*width_multiplier, 128*width_multiplier, 3, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*16*128*width_multiplier, 128*width_multiplier),
        )
    
        if cls_range is None:
            # Direct regression
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(128*width_multiplier, 1),
                nn.ReLU(),
            )
        else:
            self.head = nn.Softmax(dim=1)

    @staticmethod
    def _expectation(probs, cls_range):
        assert isinstance(cls_range, (list, tuple)), "cls_range must be a list/tuple"
        assert len(cls_range) == 2, "cls_range must have length 2"
        assert cls_range[0] < cls_range[1], "cls_range must be increasing"
        assert torch.allclose(torch.sum(probs, dim=1), torch.ones_like(probs[:, 0])), "probs must sum to 1"

        values = torch.linspace(cls_range[0], cls_range[1], probs.shape[1], device=probs.device)
        return torch.sum(probs * values, dim=1)

    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x):
        normed_x = self.input_norm(x)
        latent = self.encoder(normed_x)
        value = self.head(latent)

        if self.cls_range is None:
            # Direct regression
            return value
        else:
            # Classification output
            return self._expectation(value, self.cls_range)
