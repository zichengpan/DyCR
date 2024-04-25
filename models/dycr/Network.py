import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *

class MYNET(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','PlantVillage']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.cos_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        x = self.encode(x)
        feat = x
        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
        x = self.args.temperature * x

        return x, feat

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def update_fc(self,dataloader,class_list,session, exemplar=None, dim=4):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if exemplar == None:
            new_fc, _ = self.update_fc_avg(data, label, class_list)
            return
        else:
            new_fc, exemplar = self.update_fc_avg(data, label, class_list, exemplar, dim=dim)

        return exemplar

    def update_fc_avg(self,data,label,class_list, exemplar=None, dim=4):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto

            if exemplar is not None:
                for item in range(embedding.shape[0]):
                    tmp = embedding.view(embedding.shape[0], dim,
                                                         int(self.num_features / dim))
                    Q, R = torch.linalg.qr(tmp[item, :, :], mode="complete")
                    exemplar[class_index, item, :] = torch.reshape(Q, [dim * dim])

        new_fc=torch.stack(new_fc,dim=0)
        return new_fc, exemplar
