import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import os
from model import TeacherNet, Classifier

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

# input size를 알기 위해서
examples = enumerate(train_set)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

model = TeacherNet().to(device)  # CNN instance 생성
classifier = Classifier().to(device)
# Cost Function과 Optimizer 선택
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):  # epochs수만큼 반복
    avg_cost = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # 모든 model의 gradient 값을 0으로 설정
        latent = model(data)
        # 모델을 forward pass해 결과값 저장
        hypothesis  = classifier(latent)
        cost = criterion(hypothesis, target)  # output과 target의 loss 계산
        cost.backward()  # backward 함수를 호출해 gradient 계산
        optimizer.step()  # 모델의 학습 파라미터 갱신
        avg_cost += cost / len(train_loader)  # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# test
model.eval()  # evaluate mode로 전환 dropout 이나 batch_normalization 해제
with torch.no_grad():  # grad 해제
    correct = 0
    total = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        latent = model(data)
        out = classifier(latent)
        preds = torch.max(out.data, 1)[1]  # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
        total += len(target)  # 전체 클래스 개수
        correct += (preds == target).sum().item()  # 예측값과 실제값이 같은지 비교

    print('Test Accuracy: ', 100. * correct / total, '%')

path2weights = './models/weight.pt'
path2model = './models/model.pt'
path2w_c  = './models/weight_c.pt'
path2m_c = './models/model_c.pt'
# check directory
torch.save(model.state_dict(), path2weights)
# store model and weights into a file
torch.save(model, path2model)


torch.save(classifier.state_dict(), path2w_c)
# store model and weights into a file
torch.save(classifier, path2m_c)