# data/base_dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd  # 데이터 로드용

class BaseDataset(Dataset):
    
    def __init__(self, data_path=None, data=None, labels=None, transform=None, target_col='label'):
        """
        data_path: CSV/파일 경로 (우선 로드)
        data/labels: 직접 전달 (NumPy/Pandas/Tensor)
        transform: 이미지 등 전처리 (torchvision.transforms)
        target_col: 레이블 컬럼명 (CSV 경우)
        """
        
        self.transform = transform or transforms.Compose([])  # 기본 transform
        self.target_col = target_col
        
        if data_path:
            self.data = pd.read_csv(data_path) if data_path.endswith('.csv') else np.load(data_path)
            self.labels = self.data[target_col].values if isinstance(self.data, pd.DataFrame) else None
        else:
            self.data = data
            self.labels = labels
        
        self.n_samples = len(self.data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.labels[idx] if self.labels is not None else None
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target


if __name__ == "__main__":    
    from torchvision import transforms

    # test dataset
    dummy_data = np.random.rand(100, 28, 28)  #  28x28로 정의된 100개의 데이터 포인터
    dummy_labels = np.random.randint(0, 10, size=(100,)) # 0-9 사이의 레이블

    # data prerocessing : transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # BaseDataset Insdtance 생성
    dataset = BaseDataset(data=dummy_data, labels=dummy_labels, transform=transform)

    # 
    sample, label = dataset[0]
    print("Total Dataset length:", len(dataset))
    print("Sample shape:", sample.shape, "Label:", label)
    print("Sample pixel range:", sample.min().item(), "to", sample.max().item())
    print("sample type:", type(sample), "label type:", type(label))
