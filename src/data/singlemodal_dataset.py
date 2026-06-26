import os
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler
import random
from src.utils import *


class CompositeTableDataset(Dataset):
    def __init__(self, split, root_path, fea_col, tar_col, cate_col, scaler, use_normalize=False):
        super().__init__()
        tabular_data_path = os.path.join(root_path, f"{split}.csv")
        df = pd.read_csv(tabular_data_path)
        tabular_data = df[fea_col].to_numpy()
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        labels = df[tar_col].to_numpy()
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mean = None
        self.std = None
        self.set_mean_and_std()
        self.n_tasks = labels.shape[1]
        self.category_cols = cate_col
        self.feature_cols = fea_col
        self.data = df
        self.use_normalize = use_normalize
        self.scaler = scaler
        self.normalize(self.scaler)

    def normalize(self, scaler):
        if self.category_cols is None:
            continue_fea = self.data[self.feature_cols].to_numpy()
        else:
            continue_fea = self.data[[i for i in self.feature_cols if i not in self.category_cols]].to_numpy()
            cat_fea = self.data[self.category_cols].to_numpy()
            
        if self.use_normalize:
            if scaler is None:
                scaler = StandardScaler()
                continue_fea = scaler.fit_transform(continue_fea)
                self.scaler = scaler
            else:
                continue_fea = scaler.transform(continue_fea)
            
        if self.category_cols is None:
            self.features = continue_fea
        else:
            self.features = np.concatenate([continue_fea, cat_fea], axis=1)

    def __getitem__(self, item):
        return torch.tensor(self.features[item], dtype=torch.float32), torch.tensor(self.labels[item], dtype=torch.float32)

    def __len__(self):
        return self.labels.size()[0]

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std


class TableDataset(Dataset):
    def __init__(self, split, dataset_type, root_path="../datasets", scaler=None):
        super().__init__()
        tabular_data_path = os.path.join(root_path, f"table/{split}/{dataset_type}.csv")
        df = pd.read_csv(tabular_data_path)
        tabular_data = df.values[:, 1:8]
        if scaler is not None:
            tabular_data_cont = scaler.transform(tabular_data[:, :-1])
        else:
            scaler = StandardScaler()
            tabular_data_cont = scaler.fit_transform(tabular_data[:, :-1])
            self.scaler = scaler
        tabular_data = np.concatenate([tabular_data_cont, tabular_data[:, -1][:, np.newaxis]], axis=1)
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)

        labels = df.values[:, 8:13]
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mean = None
        self.std = None
        self.set_mean_and_std()
        self.n_tasks = labels.shape[1]

    def __getitem__(self, item):
        return self.tabular_data[item], self.labels[item]

    def __len__(self):
        return self.labels.size()[0]

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std


class CompositeImageDataset(Dataset):
    def __init__(self, split, table_root, image_dir, image_extension='.png', feature_cols=None, target_cols=None, id_col='Image index'):
        self.csv_file = os.path.join(table_root, f'{split}.csv')
        self.image_dir = image_dir
        self.image_extension = image_extension
        self.data = pd.read_csv(self.csv_file)
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        
        self.features = self.data[self.feature_cols].values.astype(np.float32)
        self.targets = self.data[self.target_cols].values.astype(np.float32)
        
        self.sample_ids = self.data[id_col].values
        self.mean, self.std = None, None
        self.set_mean_and_std()

        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.targets, axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.targets, axis=0))
        self.mean = mean
        self.std = std
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        sample_id = self.sample_ids[idx]      
        image_path = os.path.join(self.image_dir, f"{sample_id:03d}{self.image_extension}")
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, targets


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class FibreImageDataset(CompositeImageDataset):
    def __init__(self, split, table_root, image_dir, image_extension='.png',
                 feature_cols=None, target_cols=None, id_col='ID', imagenet_norm=False):
        super().__init__(split, table_root, image_dir, image_extension,
                         feature_cols, target_cols, id_col)
        norm = ([transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
                if imagenet_norm else [])
        self.transform_train = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()] + norm)
        self.transform_test  = transforms.Compose(
            [transforms.Resize((224, 224)), Rotate90(), transforms.ToTensor()] + norm)

    def get_image(self, sample_id, transform_method):
        img_path = os.path.join(self.image_dir, str(sample_id))
        fn = random.choice(os.listdir(img_path))
        img_full_path = os.path.join(img_path, fn)
        image = Image.open(img_full_path)
        image = transform_method(image)
        return image

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        sample_id = self.sample_ids[idx]
        direction = int(features[-1])
        image = self.get_image(sample_id,
                               self.transform_train if direction == 1 else self.transform_test)
        return image, targets


if __name__ == "__main__":
    pass
