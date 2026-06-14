# Importing libraries
import torch
import torch.nn as nn
import torchvision.models as models
import timm 


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet152(pretrained=True)
        
        modules = list(resnet.children())[:-2]
        self.get_features = nn.Sequential(*modules)
        
        # Freeze all ResNet weights
        for p in self.get_features.parameters():
            p.requires_grad = False

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        with torch.no_grad():            # ensure no grads are computed
            feat_map = self.get_features(x)
            a = self.avg_pool(feat_map).view(x.size(0), -1)
            m = self.max_pool(feat_map).view(x.size(0), -1)
            
        return torch.cat((a, m), dim=1)
    
    


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="efficientnet_b5"):
        super().__init__()
        
        # Load EfficientNet-B5
        self.model = timm.create_model(model_name, pretrained=True)
        
        # Remove the classification head
        self.model.reset_classifier(0)  

        # Freeze EfficientNet parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # EfficientNet-B5 outputs 2048 channels before the classifier
        self.out_channels = 2048

        # Global pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        with torch.no_grad():
            
            # Extract feature maps using EfficientNet's built-in method
            feat_map = self.model.forward_features(x)  # Shape: (B, 2048, H, W)

            # Safety check
            if feat_map.ndim != 4:
                raise ValueError(f"Expected 4D feature map, got {feat_map.shape}")

            # Pool and flatten
            a = self.avg_pool(feat_map).view(x.size(0), -1)
            m = self.max_pool(feat_map).view(x.size(0), -1)
            features = torch.cat((a, m), dim=1)  # Shape: (B, 4096)

        return features
    
    
    
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pretrained VGG19
        vgg19 = models.vgg19(pretrained=True)
        
        # Use only convolutional layers, exclude classifier
        self.features = vgg19.features 
        
        # Freeze parameters
        for p in self.features.parameters():
            p.requires_grad = False

        self.out_channels = 512

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        with torch.no_grad():
            feat_map = self.features(x)  # Shape: (B, 512, H, W)

            a = self.avg_pool(feat_map).view(x.size(0), -1)
            m = self.max_pool(feat_map).view(x.size(0), -1)
            features = torch.cat((a, m), dim=1)  # (B, 1024)

        return features
    
    
    
class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)

        # Keep only convolutional layers (exclude final FC)
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )

        # Freeze parameters
        for p in self.features.parameters():
            p.requires_grad = False

        self.out_channels = 2048

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        with torch.no_grad():
            feat_map = self.features(x)  # Shape: (B, 2048, H, W)

            a = self.avg_pool(feat_map).view(x.size(0), -1)
            m = self.max_pool(feat_map).view(x.size(0), -1)
            features = torch.cat((a, m), dim=1)  # (B, 4096)

        return features
    
    
    
    
    
    
    
    
    
    

'''
# Load Pretrained ResNet with Cat Pooling
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet_model = models.resnet152(pretrained=True) 
        
        # Remove AvgPool and Dense Layer
        modules = list(resnet_model.children())[:-2]
        self.get_features = nn.Sequential(*modules) # Model with just the convolutional layers and last residual block
        
        self.get_features.eval()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Avg Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling
    
    def forward(self, x):
        features = self.get_features(x)  # Shape: [batch, 2048, h, w]
        
        avg_pooled = self.avg_pool(features).view(features.size(0), -1)  # Shape: [batch, 2048]
        max_pooled = self.max_pool(features).view(features.size(0), -1)  # Shape: [batch, 2048]

        cat_features = torch.cat((avg_pooled, max_pooled), dim=1)  # Concatenate [batch, 4096]
        
        return cat_features
    '''