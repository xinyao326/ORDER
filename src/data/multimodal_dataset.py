import os
from PIL import Image
import torch
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as F


class Rotate90:
    def __call__(self, img):
        return F.rotate(img, 90)


class MultiModalCompositeDataset(Dataset):
    def __init__(
        self, 
        csv_file,
        image_dir,
        train_transform=None,
        test_transform=None,
        image_extension='.png',
        feature_cols=None,
        target_cols=None,
        idx_col=None,
        category_cols=None,
        extracted_fea=None,
        istrain=True,
        scaler=None,
        use_normalize=False
    ):
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.image_extension = image_extension
        self.istrain = istrain
        self.data = pd.read_csv(csv_file)
        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.category_cols = category_cols
        self.scaler = scaler
        self.use_normalize = use_normalize
        
        self.targets = self.data[self.target_cols].values.astype(np.float32)
        
        self.sample_ids = self.data[idx_col].values
        self.mean = torch.from_numpy(self.targets.mean(0))
        self.std = torch.from_numpy(self.targets.std(0))
        
        if train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.train_transform = train_transform
        if test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.test_transform = test_transform
        
        self.extracted_fea = extracted_fea
        if self.extracted_fea is not None:
            assert isinstance(self.extracted_fea, dict)

        self.normalize(scaler)

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
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_image(self, sample_id, transform_method):
        image_path = os.path.join(self.image_dir, f"{sample_id:03d}{self.image_extension}")
        image = Image.open(image_path).convert('RGB')
        image = transform_method(image)
        return image
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        sample_id = self.sample_ids[idx]      
        image = self.get_image(sample_id, self.train_transform if self.istrain else self.test_transform)
        return (idx, targets, features, image)


class MultiModalFibreDataset(MultiModalCompositeDataset):
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
        image = self.get_image(sample_id, self.train_transform if int(direction)==1 else self.test_transform)
        return (idx, targets, features, image)


if __name__ == "__main__":
    pass