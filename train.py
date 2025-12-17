import pickle

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from predata import *
import torch
import os

os.makedirs("./model",exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path = r'train/part.0'
test = analyspromption()
text , label = test.load_text(path)
clean_text = test.clean_text(text)
fenci = test.fenci(clean_text)
vector, vectorizer = test.tezheng(fenci,2)


#开始进行神经网络训练
x_dense = vector.toarray()
x_dense = torch.tensor(x_dense, dtype=torch.float)
#将label中的字符0  1 转换为int型，然后再转换为tensor类型
y_dense = torch.tensor([int(i) for i in label], dtype=torch.long)


dataset = TensorDataset(x_dense, y_dense)
loder = DataLoader(dataset=dataset,batch_size = 32,shuffle=True)

model = nn.Sequential(
    nn.Linear(x_dense.shape[1], 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.ReLU(),
    nn.Linear(10,2)

).to(device)

if __name__ == '__main__':
    # 计算损失
    loss = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        for x_dense, y_dense in loder:
            x_b = x_dense.to(device)
            y_b = y_dense.to(device)

            optimizer.zero_grad()
            output = model(x_b)
            loss_1 = loss(output, y_b)
            loss_1.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}: loss={loss_1.item():.4f}")
            torch.save(model.state_dict(), f"./model/model{epoch}.pth")
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)