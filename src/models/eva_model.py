import sys
import timm
import torch.nn as nn

sys.path.append('../..')

class EVA02(nn.Module):
    def __init__(self, model_name, in_channels=3, num_classes=1, pretrained=True, checkpoint=None):
        super(EVA02, self).__init__()

        # create model
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_channels, checkpoint_path=checkpoint)

        # get in_features
        in_features = self.model.head.in_features

        # set classifier
        self.model.head = nn.Linear(in_features, num_classes) 

        # sigmoid function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        # forward pass
        y = self.model(x)

        # apply sigmoid
        y = self.sigmoid(y)

        return y