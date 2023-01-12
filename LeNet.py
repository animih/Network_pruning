import torch
import torch.nn.functional as F
from torch import nn, optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, 1)


def train_model(model, optimizer, crit, loaders,
    reg = None, display = False,
    N_epochs = 30):

    t = 0

    accuracy = 0

    for epoch in range(N_epochs):
        for k, dataloader in loaders.items():
            epoch_correct = 0
            epoch_all = 0

            for x_batch, y_batch in dataloader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                tmp = time.time()

                if k == "train":
                    model.train()
                    optimizer.zero_grad()
                    outp = model(x_batch)
                    loss = crit(outp, y_batch)
                    if reg is not None:
                        loss += reg(model.parameters())
                    loss.backward()
                    optimizer.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)
                preds = outp.argmax(-1)
                correct = torch.sum(preds == y_batch)
                all = preds.shape[0]
                epoch_correct += correct.item()
                epoch_all += all

                tmp = time.time() - tmp
                t += tmp

            if (display) :
                if k == "train":
                    print(f"Epoch: {epoch+1}")
                print(f"Loader: {k}. Accuracy: {epoch_correct/epoch_all}")

        if k != "train" and epoch == N_epochs-1:
            accuracy = epoch_correct/epoch_all

    return accuracy, t