import torch
import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):

    @abstractmethod
    def forward(self, *inputs):
        pass

    def __str__(self):
        """
            설명 : 학습에 사용할 파라미터의 개수를 출력하는 함수
        """
        model_params = [p for p in self.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in model_params)

        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(n_params)


if __name__ == "__main__":

    # Test code
    class MLP(BaseModel):
        def __init__(self, input_size=784, hidden_size=256, output_size=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()

        # Forward Override 구현(필수 : @abstractmethod)
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Model 정의
    model = MLP(input_size=784, hidden_size=256, output_size=10)
    
    # __str__ 테스트
    print(model)

    # 더미 데이터로 Forward 테스트
    test_x = torch.randn(16, 784)  # 배치 크기 16, 입력 크기 784
    print("Test input shape:", test_x.shape)
    
    test_output = model(test_x)
    print("Test output shape:", test_output.shape)


        

