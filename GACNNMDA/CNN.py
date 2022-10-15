import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.l1=nn.Sequential(
            nn.Conv2d(1,3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.l2=nn.Sequential(
            nn.Conv2d(3,6,kernel_size=3,padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        self.l3=nn.Linear(38640 ,2)
    def forward(self,x):
        input_size=x.size(0)
        x=self.l1(x)
        x=self.l2(x)
        x=x.view(input_size,-1)
        x=self.l3(x)
        x=F.log_softmax(x)
        return x


def train1(x_train,y_train,x_test,y_test):
    transform = transforms.ToTensor()
    nepoch=8
    class trainDataset(Dataset):
        def __init__(self):
            self.transforms = transform
            self.x_data, self.y_data = x_train, y_train

        def __getitem__(self, index):
            x = self.transforms(self.x_data[index])
            y = self.y_data[index]
            return x, y

        def __len__(self):
            return len(self.x_data)
    class testDataset(Dataset):
        def __init__(self):
            self.transforms = transform
            self.x_data, self.y_data = x_test, y_test

        def __getitem__(self, index):
            x = self.transforms(self.x_data[index])
            y = self.y_data[index]
            return x, y
        def __len__(self):
            return len(self.x_data)
    traindataset=trainDataset()
    testdataset=testDataset()
    trainloader = DataLoader(dataset=traindataset, batch_size=8, shuffle=True)
    testloader=DataLoader(dataset=testdataset,batch_size=8,shuffle=False)
    model = CNN()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Train
    total_step = len(trainloader)
    for epoch in range(nepoch):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(torch.float32) #[8,1,2,128]
            # inputs=torch.tensor(item for item in inputs)
            # labels=torch.tensor(item for item in labels)
            outputs = model(inputs)  #[8,3,2,128]
            labels = labels.long()
            #outputs=torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 8 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}'
                      .format(epoch + 1, nepoch, i + 1, total_step, loss.item()))
    model.eval()
    score = []
    tlabel = []
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in testloader:
            inputs = inputs.to(torch.float32)
            labels = labels.long()
            outputs = model(inputs)
            data = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(data, 1)
            total += labels.size(0)
            tmp = data.numpy()
            tmp1 = labels.numpy()
            for i in range(len(tmp)):
                score.append(tmp[i][1])
                tlabel.append(tmp1[i])
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    return score, tlabel