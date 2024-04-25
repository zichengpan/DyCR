import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

identity = lambda x: x

class PlantVillage(Dataset):

    def __init__(self, root='./filelists', train=True,
                 index_path=None, index=None, base_sess=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.IMAGE_PATH = os.path.join(root, 'PlantVillage/images')
        self.SPLIT_PATH = os.path.join(root, 'PlantVillage/split')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(';')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[2])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


class SubDataset:
    def __init__(self, sub_meta, cl, transform, identity, noise_rate):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = identity
        self.noise_rate = noise_rate

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        if self.noise_rate > 0.:
            if np.random.random() > (1 - self.noise_rate):
                img = np.array(img)
                img = np.random.randint(0, 255, size=img.shape)
                img = Image.fromarray(img.astype('uint8'))
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)
