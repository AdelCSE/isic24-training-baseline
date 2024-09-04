import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x)
    
    def gem(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class EffNet(nn.Module):
    def __init__(self, model_name, in_channels=3, num_classes=1, pretrained=True, checkpoint=None):
        super(EffNet, self).__init__()

        # create model
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_channels, num_classes=num_classes, checkpoint_path=checkpoint)

        # Pooling and classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.fc = nn.Linear(in_features, num_classes)

        # sigmoid function
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        # forward pass
        feats = self.model(x)

        # pooling
        pooled_feats = self.pooling(feats).flatten(1)

        # apply sigmoid
        y = self.sigmoid(self.fc(pooled_feats))

        return y