import torch
from torchsummary import summary
import torchvision
import torchvision.transforms as transfroms
import torch.nn as nn
from model import TeacherNet, Classifier,StudentNet, BarlowTwinsLoss
from torchsummary import summary
from torchmetrics import ConfusionMatrix
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device + " is available")

learning_rate = 0.001
batch_size = 100
num_classes = 10
epochs = 5


# MNIST 데이터셋 로드
train_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)
test_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=False,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

# train_loader, test_loader 생성
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

examples = enumerate(train_set)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

path2weights = './models/weight.pt'
path2model = './models/model.pt'
t_model = torch.load(path2model)
weights = torch.load(path2weights)
t_model.load_state_dict(weights)

s_model = StudentNet().to(device)

path2w_c  = './models/weight_c.pt'
path2m_c = './models/model_c.pt'
classifier = torch.load(path2m_c)
weights_c = torch.load(path2w_c)
classifier.load_state_dict(weights_c)

# Cost Function과 Optimizer 선택
for param in t_model.parameters():
    param.requires_grad = False
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(s_model.parameters(), lr=learning_rate)

for epoch in range(epochs):  # epochs수만큼 반복
    avg_cost = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # 모든 model의 gradient 값을 0으로 설정
        t_latent = t_model(data) # 모델을 forward pass해 결과값 저장
        s_latent = s_model(data)
        cost = criterion(s_latent, t_latent)  # output과 target의 loss 계산
        cost.backward()  # backward 함수를 호출해 gradient 계산
        optimizer.step()  # 모델의 학습 파라미터 갱신
        avg_cost += cost / len(train_loader)  # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))



# train_loader, test_loader 생성
s_model.eval()  # evaluate mode로 전환 dropout 이나 batch_normalization 해제
with torch.no_grad():  # grad 해제
    correct = 0
    total = 0
    zeros = torch.zeros(9,9)
    y_true = []
    y_pred = []
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        y_true.extend(target.numpy())
        latent =s_model(data)
        out = classifier(latent)

        preds = torch.max(out.data, 1)[1] # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
        total += len(target)  # 전체 클래스 개수
        y_pred.extend(preds.cpu().numpy())
        correct += (preds == target).sum().item()  # 예측값과 실제값이 같은지 비교

    cf_matrix = confusion_matrix(y_true, y_pred)


    print(cf_matrix)
    acc = np.sum(cf_matrix,axis=1)
    for i in range(len(acc)):
        acc1 =cf_matrix[i,:]/acc[i]
        print(acc1)
        print(np.sum(acc1))

    print('Test Accuracy: ', 100. * correct / total, '%')

