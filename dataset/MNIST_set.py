from torchvision.datasets import MNIST
from base import BaseDataset  # 상대 import

class MNISTDataset(BaseDataset):
    def __init__(self, root='./data', train=True, download=True, transform=None):
        # torchvision MNIST 로드
        mnist = MNIST(root=root, train=train, download=download, transform=transform)
        super().__init__(data=mnist.data.float() / 255.0,  # 정규화
                        labels=mnist.targets,
                        transform=None)  # 추가 transform은 init에서 처리