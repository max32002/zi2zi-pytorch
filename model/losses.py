import torch
import torch.nn as nn


class CategoryLoss(nn.Module):
    def __init__(self, category_num):
        super(CategoryLoss, self).__init__()
        emb = nn.Embedding(category_num, category_num)
        emb.weight.data = torch.eye(category_num)
        self.emb = emb
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, category_logits, labels):
        target = self.emb(labels)
        return self.loss(category_logits, target)


class BinaryLoss(nn.Module):
    def __init__(self, real):
        super(BinaryLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.real = real

    def forward(self, logits):
        if self.real:
            labels = torch.ones(logits.shape[0], 1)
        else:
            labels = torch.zeros(logits.shape[0], 1)
        if logits.is_cuda:
            labels = labels.cuda()
        return self.bce(logits, labels)


class PerceptualLoss(nn.Module):
    def __init__(self, layer_index=15, device=None):
        # 15 is relu3_3 usually
        super(PerceptualLoss, self).__init__()
        from torchvision.models import vgg16, VGG16_Weights
        
        # Load VGG16 with default pre-trained weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        # We only need the features part
        
        # features list commonly used for perceptual loss:
        # relu1_2: 3
        # relu2_2: 8
        # relu3_3: 15
        # relu4_3: 22
        
        self.vgg_features = vgg.features[:layer_index + 1].eval()
        
        if device:
             self.vgg_features.to(device)

        for param in self.vgg_features.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()
        
        # ImageNet mean and std for normalization
        # VGG expects normalized input
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_img, target_img):
        # Input images are usually [-1, 1] in GANs (tanh output)
        # Convert to [0, 1]
        input_img = (input_img + 1) / 2
        target_img = (target_img + 1) / 2
        
        if input_img.is_cuda and not self.mean.is_cuda:
            self.mean = self.mean.to(input_img.device)
            self.std = self.std.to(input_img.device)

        # If input is 1 channel, repeat to 3
        if input_img.shape[1] == 1:
            input_img = input_img.repeat(1, 3, 1, 1)
        if target_img.shape[1] == 1:
            target_img = target_img.repeat(1, 3, 1, 1)

        # Normalize
        input_img = (input_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std
        
        # Get features
        input_features = self.vgg_features(input_img)
        target_features = self.vgg_features(target_img)
        
        return self.criterion(input_features, target_features)
