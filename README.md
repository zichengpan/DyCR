# DyCR
This repo is the implementation of the following paper:

**DyCR: A Dynamic Clustering and Recovering Network for Few-Shot Class-Incremental Learning** (TNNLS 2024), [[paper]](https://ieeexplore.ieee.org/document/10531293)

## Abstract
Few-shot class-incremental learning (FSCIL) aims to continually learn novel data with limited samples. One of the major challenges is the catastrophic forgetting problem of old knowledge while training the model on new data. To alleviate this problem, recent state-of-the-art methods adopt a well-trained static network with fixed parameters at incremental learning stages to maintain old knowledge. These methods suffer from the poor adaptation of the old model with new knowledge. In this work, a dynamic clustering and recovering network (DyCR) is proposed to tackle the adaptation problem and effectively mitigate the forgetting phenomena on FSCIL tasks. Unlike static FSCIL methods, the proposed DyCR network is dynamic and trainable during the incremental learning stages, which makes the network capable of learning new features and better adapting to novel data. To address the forgetting problem and improve the model performance, a novel orthogonal decomposition mechanism is developed to split the feature embeddings into context and category information. The context part is preserved and utilized to recover old class features in future incremental learning stages, which can mitigate the forgetting problem with a much smaller size of data than saving the raw exemplars. The category part is used to optimize the feature embedding space by moving different classes of samples far apart and squeezing the sample distances within the same classes during the training stage. Experiments show that the DyCR network outperforms existing methods on four benchmark datasets.

## Dataset Preparation Guidelines

### CIFAR-100, CUB200, and Mini-ImageNet:
Follow the guidelines provided in the [CEC](https://github.com/icoz69/CEC-CVPR2021) to download and prepare these datasets. Detailed steps are available in the repository.

### PlantVillage Dataset:
Download the PlantVillage dataset from [this link](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color). Organize the dataset according to the structure required for incremental learning as outlined below:

```
data
|–– CUB_200_2011/
|–– .../
|–– PlantVillage/
|   |–– images/
|   |   |–– 1.JPG
|   |   |–– 2.JPG
|   |   |–– ...
|   |–– split/
|   |   |–– train.csv
|   |   |–– test.csv
```

## Training Scripts

- Train CUB200

    ```
    python train.py -project dycr -dataset cub200 -lr_base 0.005 -lr_new 0.001 -decay 0.0005 -epochs_base 120 -schedule Milestone -milestones 60 80 100 -gpu 0,1,2,3 -temperature 16 -batch_size_base 256
    ```
- Train CIFAR-100
    ```
    python train.py -projec dycr -dataset cifar100 -gamma 0.1 -lr_base 0.1 -lr_new 0.001 -decay 0.0005 -epochs_base 600 -schedule Cosine -gpu 0,1,2,3 -temperature 16 -batch_size_base 256
    ```
- Train Mini-ImageNet
    ```
    python train.py -project dycr -dataset mini_imagenet -gamma 0.1 -lr_base 0.1 -lr_new 0.001 -decay 0.0005 -epochs_base 600 -schedule Cosine -gpu 0,1,2,3 -temperature 16 -batch_size_base 64
    ```
- Train PlantVillage
    ```
    python train.py -projec dycr -dataset PlantVillage -gamma 0.1 -lr_base 0.1 -decay 0.0005 -epochs_base 200 -schedule Cosine -gpu 0,1,2,3 -temperature 16 -batch_size_base 32
    ```

## Citation
If you find our code or paper useful, please give us a citation.
```bash
@ARTICLE{10531293,
  author={Pan, Zicheng and Yu, Xiaohan and Zhang, Miaohua and Zhang, Weichuan and Gao, Yongsheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={DyCR: A Dynamic Clustering and Recovering Network for Few-Shot Class-Incremental Learning}, 
  year={2024},
  pages={1-14},
  doi={10.1109/TNNLS.2024.3394844}
}
```

## Acknowledgment
We thank the following repos for providing helpful components/functions in our work.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)

- [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)

- [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset/tree/master)
