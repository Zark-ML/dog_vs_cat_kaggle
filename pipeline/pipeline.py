import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Normalize, Resize, Compose

from pipeline.model import SimpleCNN
from pipeline.dataset import DogCatDataset

class Pipeline:
    def __init__(self, train_dir, val_dir, size, batch_size, epochs, learning_rate, log_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_dir = log_dir

        self.train_loader, self.val_loader = self.get_data_loaders(train_dir, val_dir)
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(log_dir=log_dir)

    def get_data_loaders(self, train_dir, val_dir):
        transform = Compose([
            Resize(self.size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = DogCatDataset(root_dir=train_dir, size=self.size, transform=transform)
        val_dataset = DogCatDataset(root_dir=val_dir, size=self.size, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader

    def train_and_evaluate(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.evaluate(epoch)

        self.writer.close()

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_loader.dataset)

        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

        print(f'Epoch {epoch}/{self.epochs - 1}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    def evaluate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(self.val_loader.dataset)
        val_acc = val_corrects.double() / len(self.val_loader.dataset)

        self.writer.add_scalar('Validation/Loss', val_loss, epoch)
        self.writer.add_scalar('Validation/Accuracy', val_acc, epoch)

        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
