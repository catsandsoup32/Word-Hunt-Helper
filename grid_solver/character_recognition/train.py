import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from image_transform import get_torch_transform
from model import SmallCNN

device = torch.device("cuda") 

class Runner():
    def __init__(self, model, data_dir, num_epochs, transform, lr=3e-4, 
                 batch_size=16, num_workers=8, train_split=0.80):
        self.model = model.to(device)
        self.num_epochs = num_epochs
        dataset = ImageFolder(data_dir, transform=transform)
        train_set, val_set = random_split(dataset, [train_split, 1-train_split])
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.accuracy = Accuracy(task="multiclass", num_classes=26).to(device)
    
        
    def train(self):
        for i in range(1, self.num_epochs+1):
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(self.train_loader, desc=f"  Train loop {i}: "):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                running_loss += loss.item()
            print(f"Avg loss: {round(running_loss/len(self.train_loader), 2)}")
            self.val()
        torch.save(model.state_dict(), "weights/3_epochs.pth")

    def val(self):
        self.model.eval()
        running_loss = 0.0 
        with torch.no_grad():        
            for images, labels in tqdm(self.val_loader, desc="  Val loop: "):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                self.accuracy.update(outputs, labels)
        total_acc = self.accuracy.compute().item()
        self.accuracy.reset()
        print(f"Avg val loss: {round(running_loss/len(self.val_loader), 2)}")
        print(f"Val accuracy: {round(total_acc, 2)}")

if __name__ == "__main__":
    model = SmallCNN()
    transform = get_torch_transform()

    runner = Runner(
        model, 
        "data", 
        num_epochs=3,
        transform=transform
        )
    runner.train()