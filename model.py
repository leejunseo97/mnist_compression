import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary

class TeacherNet(nn.Module):
    def __init__(self):  # layer 정의
        super(TeacherNet, self).__init__()

        # input size = 28x28
        self.conv1 = nn.Conv2d(1, 10,
                               kernel_size=5)  # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((W-K+2P)/S)+1 공식으로 인해 ((28-5+0)/1)+1=24 -> 24x24로 변환
        # maxpooling하면 12x12

        self.conv2 = nn.Conv2d(10, 20,
                               kernel_size=5)  # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((12-5+0)/1)+1=8 -> 8x8로 변환
        # maxpooling하면 4x4

        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)  # 랜덤하게 뉴런을 종료해서 학습을 방해해 학습이 학습용 데이터에 치우치는 현상을 막기 위해 사용
        self.mp = nn.MaxPool2d(2)  # 오버피팅을 방지하고, 연산에 들어가는 자원을 줄이기 위해 maxpolling
        self.fc = nn.Linear(320, 112)  # 4x4x20 vector로 flat한 것을 100개의 출력으로 변경


    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))  # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
        x = F.relu(self.mp(self.conv2(x)))  # convolution layer 2번에 relu를 씌우고 maxpool, 결과값은 4x4x20
        x = self.drop2D(x)
        x = x.view(x.size(0), -1)  # flat
        x = self.fc(x)  # fc1 레이어에 삽입
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()


        self.fc1 = nn.Linear(112,100)
        self.fc2 = nn.Linear(100,10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x)

class StudentNet(nn.Module):
    def __init__(self, Linear_num):
        super(StudentNet, self).__init__()

        for i in range(Linear_num):
            self.row{i}' = nn.Linear(28,1)
            f'self.col{i}' = nn.Linear(28, 1)


        #self.projector1 = nn.Linear(112,30)

        #self.projector2 = nn.Linear(20,60)

    def forward(self,x,Linear_num):
        cat = []
        for i in range(Linear_num):
            f'r_{i}' = F.relu(f'self.row{i}'(x))
            f'r_{i}'  = f'r_{i}' .view(f'r_{i}' .size(0), -1)


            f'c_{i}'  = torch.transpose(F.relu(f'self.col{i}'(torch.transpose(x,0,1))),0,1)
            f'c_{i}' = f'c_{i}'.view(f'c_{i}'.size(0), -1)
            concat = torch.cat(f'r_{i}',f'c_{i}', dim = 1)
            cat.append(concat)


        #out1 = self.projector1(concat)
        #out2 = self.projector2(out1)
        return cat

class ex_t_model(nn.Module):
    def __init__(self,output_layer, *args):
        super().__init__(*args)
        self.path2weights = './models/weight.pt'
        self.path2model = './models/model.pt'

        # define the teacher model
        self.t_model = TeacherNet()

        self.t_model = torch.load(self.path2model)
        self.weights = torch.load(self.path2weights)
        self.t_model.load_state_dict(self.weights)

        self.pretrained = self.t_model
        print(self.pretrained)
        self.output_layer = output_layer
        self.selected_out = OrderedDict()
        self.fhooks = []

        for i, l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layer:
                self.fhooks.append(getattr(self.pretrained, l).register_forward_hook(self.forward_hook(l)))

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size = 100, lambda_coeff=5e-3, z_dim=56):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag