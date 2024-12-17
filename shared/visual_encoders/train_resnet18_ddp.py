import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from dataclasses import dataclass
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist

from shared.visual_encoders.resnet_18 import get_resnet18


@dataclass
class Config:
    """
    Simple training config
    """

    data_dir: str = "/dataset/..."  # Replace with your dataset path
    num_classes: int = 200
    batch_size: int = 256
    num_epochs: int = 2
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    log_interval: int = 10
    save_dir: str = "./checkpoints"
    pretrained: bool = False
    backend: str = "nccl"
    num_workers: int = 8


def train_one_epoch(
    model, dataloader, optimizer, criterion, device, scaler, epoch, rank, world_size
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # Mixed precision training, easier on VRAM
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss and accuracy
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if rank == 0:
            current_loss = loss.item()
            current_acc = 100.0 * correct / total
            progress_bar.set_postfix(loss=current_loss, acc=current_acc)

            if batch_idx % Config.log_interval == 0:
                wandb.log({"Train Loss": current_loss, "Train Accuracy": current_acc})

    # Aggregate metrics across all processes
    total_loss_tensor = torch.tensor(total_loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)

    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / total_tensor.item()
    accuracy = 100.0 * correct_tensor.item() / total_tensor.item()

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, rank, world_size):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", disable=(rank != 0)):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    total_loss_tensor = torch.tensor(total_loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)

    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / total_tensor.item()
    accuracy = 100.0 * correct_tensor.item() / total_tensor.item()

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_dir):
    checkpoint = {
        "model_state_dict": model.module.state_dict(),  # DDP wraps the model
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    torch.save(model.module.state_dict(), os.path.join(save_dir, "best_resnet18.pth"))


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = (
        "12355"  # Choose an open port, apparently this a common choice
    )

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.manual_seed(42)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, config):
    setup(rank, world_size, config.backend)

    if rank == 0:
        wandb.init(project="resnet18-imagenet", config=config.__dict__)

    device = torch.device(f"cuda:{rank}")

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
        os.path.join(config.data_dir, "train"), transform=transform_train
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(config.data_dir, "val"), transform=transform_val
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size // world_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size // world_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    model = get_resnet18(
        num_classes=config.num_classes, pretrained=config.pretrained
    ).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    if rank == 0:
        os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            epoch,
            rank,
            world_size,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, rank, world_size
        )

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            wandb.log(
                {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_acc,
                    "Learning Rate": scheduler.get_last_lr()[0],
                }
            )

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_acc, config.save_dir
                )
                print(f"Saved best model with accuracy: {best_acc:.2f}%")

        scheduler.step()

    if rank == 0:
        print("Training complete. Best accuracy:", best_acc)
        wandb.finish()

    cleanup()


def main():
    config = Config()
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs available for training.")

    mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
