import torch.nn as nn


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.f1 = self._make_layer(1, 64, 8, 2, 3)
        self.f2 = self._make_layer(64, 128, 6, 2)
        self.f3 = self._make_layer(128, 128, 5)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128, 10)
        )
        
    def _make_layer(self, *cfg):
        return nn.Sequential(
            nn.Conv2d(*cfg),
            nn.ReLU(),
            nn.Dropout()
        )
                                
    def forward(self, x):
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = self.f3(out2)
        out = self.classifier(out3)
        
        return out