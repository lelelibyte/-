from torch import nn

from predata import *
import torch

from train import vectorizer, device

#读取测试文件，并且不读取最后的标签，通过最后的标签来计算最后的准确率
path = r'test/part.0'
text = []
label = []




test = analyspromption()
text,label = test.load_text(path)

count = label.length()
cleaned = test.clean_text([text])
fenci = test.fenci(cleaned)


x_test = vectorizer.transform([' '.join(fenci[0])])

x_tensor = torch.tensor(x_test.toarray(), dtype=torch.float32).to(device)
feature_dim = x_tensor.shape[1]
model = nn.Sequential(
    nn.Linear(feature_dim, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
).to(device)

# 加载参数
model.load_state_dict(torch.load("./model/model90.pth", map_location=device))
model.eval()
with torch.no_grad():
    output = model(x_tensor)
    pred = torch.argmax(output, dim=1).item()

