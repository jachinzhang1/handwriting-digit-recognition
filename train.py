import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from model import Model
from dataset import transform, trainData, trainDataLoader, testData, testDataLoader
import os
import csv
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

model = Model().to(device)
dummy_input = torch.randn(13, 1, 28, 28).to(device)
with SummaryWriter(comment='Net') as w:
    w.add_graph(model, (dummy_input,))
# TODO: warm start

loss = nn.CrossEntropyLoss()
lr = 1e-4
# TODO: lr decay
epochs = 30
optimizer = optim.Adam(model.parameters(), lr=lr)
# TODO: try other optimizers (SGD, RMSprop, etc.)

history = {
    'train_loss': [],
    'train_step_loss': [],
    'train_acc': [],
    'train_step_acc': [],
    'test_loss': [],
    'test_step_loss': [],
    'test_acc': [],
    'test_step_acc': []
}

# train
def train():
    print('format: train loss | train acc | test loss | test acc')
    for epoch in range(1, epochs + 1):
        processBar = tqdm(trainDataLoader, unit='step')
        model.train()

        for step, (trainImgs, labels) in enumerate(processBar):
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            pred = model(trainImgs)
            trainLoss = loss(pred, labels)
            result = torch.argmax(pred, dim=1)
            train_accuracy = (result == labels).sum() / len(labels)
            history['train_step_loss'].append(trainLoss)
            history['train_step_acc'].append(train_accuracy.item())
            writer.add_scalar('train_loss', trainLoss, step + epoch * len(processBar))

            trainLoss.backward()
            optimizer.step()

            processBar.set_description('[%2d/%d] %.4f | %.4f' % (epoch, epochs, trainLoss, train_accuracy.item()))

            if step == len(processBar) - 1:
                history['train_loss'].append(trainLoss.item())
                history['train_acc'].append(train_accuracy.item())

                correct, testTotalLoss = 0, 0
                model.eval()
                for testImgs, labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        pred = model(testImgs)
                        testLoss = loss(pred, labels)
                        result = torch.argmax(pred, dim=1)

                    testTotalLoss += testLoss.item()
                    correct += (result == labels).sum().item()
                    history['test_step_loss'].append(testLoss.item())
                    history['test_step_acc'].append((result == labels).sum().item() / len(labels))
                    # TODO: early stopping

                test_accuracy = correct / len(testData)
                testLoss_avg = testTotalLoss / len(testDataLoader)
                history['test_loss'].append(testLoss.item())
                history['test_acc'].append(test_accuracy)
                processBar.set_description(
                    '[%2d/%d] %.4f | %.4f | %.4f | %.4f'
                    % (epoch, epochs, trainLoss, train_accuracy.item(), testLoss_avg, test_accuracy)
                )

        writer.add_scalars('epoch losses', {
            'train_loss': trainLoss,
            'test_loss': testLoss_avg
        }, epoch)
        processBar.close()

    # save model
    if os.path.exists('./models') is False:
        os.mkdir('./models')
    file_num = len(os.listdir('./models'))
    torch.save(model.state_dict(), './models/model-%d.pth' % (file_num + 1))


def plot_figs():
    file_num = len(os.listdir('./models'))
    plt.figure(figsize=(12, 6))
    plt.title('Model-%d' % (file_num + 1))
    plt.axis('off')
    # plot the loss value
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['test_loss'], label='test_loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # plot the train_accuracy value
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['test_acc'], label='test_acc')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    if os.path.exists('./fig') is False:
        os.mkdir('./fig')
    plt.savefig('./fig/model-%d.png' % (file_num + 1))
    plt.show()


def main():
    train()
    plot_figs()


if __name__ == '__main__':
    main()