import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 定义一个平衡代理类，用于生成规范化的中心点
class balanced_proxies(nn.Module):
    def __init__(self, dim, proxy_num=3):
        super(balanced_proxies, self).__init__()
        # 初始化代理点参数
        protos = torch.nn.Parameter(torch.empty([proxy_num, dim]))
        # 使用Xavier均匀初始化方法初始化代理点
        self.proxies = torch.nn.init.xavier_uniform_(protos, gain=1)

    def forward(self):
        # 规范化代理点并返回
        centers = F.normalize(self.proxies, dim=-1)
        return centers

# 定义一个基于ResNet50的神经网络模型类，用于特征提取和分类
class ECL_model(nn.Module):
    def __init__(self, num_classes=8, feat_dim=512):
        super(ECL_model, self).__init__()
        # 加载预训练的ResNet50模型
        cnns = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 保留ResNet50模型除最后全连接层外的所有层作为特征提取的backbone
        self.backbone = torch.nn.Sequential(*(list(cnns.children())[:-1]))

        self.num_classes = num_classes
        # 计算全连接层输入的维度
        dimension = 512 * 4

        # 定义一个头部网络，包含线性层、批量归一化、ReLU激活和线性层
        self.head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        # 定义一个线性层，用于输出分类结果
        self.fc = nn.Linear(dimension, self.num_classes)

    def forward(self, x):
        if isinstance(x, list):
            # 处理输入是列表（双输入），分别提取两个输入的特征
            feat1 = self.backbone(x[0])
            feat1 = feat1.view(feat1.shape[0], -1)
            feat1_mlp = F.normalize(self.head(feat1))
            logits = self.fc(feat1)

            feat2 = self.backbone(x[1])
            feat2 = feat2.view(feat2.shape[0], -1)
            feat2_mlp = F.normalize(self.head(feat2))

            return logits, [feat1_mlp, feat2_mlp]
        else:
            # 处理单输入，提取特征并输出分类结果
            feat1 = self.backbone(x)
            feat1 = feat1.view(feat1.shape[0], -1)
            logits = self.fc(feat1)

            return logits
