import numpy as np  # 선형대수 모듈
import matplotlib.pyplot as plt  # 시각화 모듈
import torch  # 파이토치
import torch.nn as nn  # Pytorch의 모듈을 모아놓은 것. from~이 아닌 저렇게 임포트를 하는것이 거의 관습이라고 함
import torch.nn.functional as F  # toch.nn 중에서 자주 쓰는 함수를 F로 임포트
import torch.nn.init as init  # 초기화 관련 모둘
import torchvision  # TorchVision 임포트
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 32  # 한번에 학습을 실행할 떄 사용할 데이터의 크기를 정함
EPOCHS = 30 # 전체 데이터를 한 번 학습 돌리는 것이 1EPOCH이므로 여기에서는 30번 학습 돌리는것을 의미함

print('Using PyTorch version:', torch.__version__, 'Device:', DEVICE)

train_dataset = datasets.MNIST(
    root="../data/MNIST",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
test_dataset = datasets.MNIST(
    root="../data/MNIST",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False

) #DataLoader는 위에서 불러온 dataset을 사전에 설정한 batch size(32)로 미니 배치 설정을 할수 있게 도움

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

def imshow(img):
    # img = img / 2 + 0.5     # denormalize
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)  # iterator
images, labels = next(dataiter)  # next() 함수 사용


imshow(torchvision.utils.make_grid(images))

