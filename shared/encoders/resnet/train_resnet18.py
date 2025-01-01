import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb

from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from shared.encoders.resnet.resnet_18 import get_resnet18


@dataclass
class Config:
    """
    Simple training config
    """

    data_dir = "/dataset/..."
    num_classes = 1000
    batch_size = 256
    num_epochs = 100
    learning_rate = 0.1
    weight_decay = 1e-4
    momentum = 0.9
    log_interval = 10  # Log every n batches
    save_dir = "./checkpoints"
    pretrained = False  # Set to True if you want to fine-tune pretrained weights


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # Mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        if batch_idx % Config.log_interval == 0:
            wandb.log(
                {"Train Loss": loss.item(), "Train Accuracy": 100.0 * correct / total}
            )

    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return total_loss / len(dataloader), accuracy


def main():
    # Should be doing something better
    wandb.init(project="resnet18-imagenet", config=Config.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(Config.data_dir, "train"), transform=transform_train
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(Config.data_dir, "val"), transform=transform_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = get_resnet18(
        num_classes=Config.num_classes, pretrained=Config.pretrained
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.learning_rate,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    os.makedirs(Config.save_dir, exist_ok=True)

    for epoch in range(Config.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        wandb.log(
            {
                "Epoch": epoch + 1,
                "Validation Loss": val_loss,
                "Validation Accuracy": val_acc,
                "Learning Rate": scheduler.get_last_lr()[0],
            }
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(Config.save_dir, "best_resnet18.pth")
            )
            print(f"Saved best model with accuracy: {best_acc:.2f}%")

        scheduler.step()

    print("Training complete. Best accuracy:", best_acc)
    wandb.finish()


if __name__ == "__main__":
    main()
