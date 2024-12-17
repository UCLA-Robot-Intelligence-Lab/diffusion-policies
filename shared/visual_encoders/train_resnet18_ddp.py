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
    Training configuration for Tiny ImageNet
    """

    data_dir: str = "./dataset/tiny-imagenet-200"
    num_classes: int = 200
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    log_interval: int = 10
    save_dir: str = "./checkpoints_tinyimagenet"
    pretrained: bool = False
    backend: str = "nccl"
    num_workers: int = 8
    patience: int = 10


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

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

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
    correct_top5 = 0

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

            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(targets.view(1, -1).expand_as(pred_top5)).sum().item()

    total_loss_tensor = torch.tensor(total_loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    correct_top5_tensor = torch.tensor(correct_top5).to(device)
    total_tensor = torch.tensor(total).to(device)

    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_top5_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / total_tensor.item()
    accuracy_top1 = 100.0 * correct_tensor.item() / total_tensor.item()
    accuracy_top5 = 100.0 * correct_top5_tensor.item() / total_tensor.item()

    return avg_loss, accuracy_top1, accuracy_top5


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_dir):
    checkpoint = {
        "model_state_dict": model.module.state_dict(),  # DDP wraps the model
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    torch.save(model.module.state_dict(), os.path.join(save_dir, "best_resnet18.pth"))


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # apparently this is standard practice

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.manual_seed(42)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, config):
    setup(rank, world_size, config.backend)

    if rank == 0:
        wandb.init(project="resnet18-tinyimagenet", config=config.__dict__)

    device = torch.device(f"cuda:{rank}")

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size // world_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler()

    best_acc = 0.0
    trigger_times = 0

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
        val_loss, val_acc_top1, val_acc_top5 = validate(
            model, val_loader, criterion, device, rank, world_size
        )

        if rank == 0:
            print(
                f"Epoch {epoch+1}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc Top-1: {val_acc_top1:.2f}%, Val Acc Top-5: {val_acc_top5:.2f}%"
            )

            wandb.log(
                {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Validation Loss": val_loss,
                    "Validation Accuracy Top-1": val_acc_top1,
                    "Validation Accuracy Top-5": val_acc_top5,
                    "Learning Rate": scheduler.get_last_lr()[0],
                }
            )

            if val_acc_top1 > best_acc:
                best_acc = val_acc_top1
                trigger_times = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_acc, config.save_dir
                )
                print(f"Saved best model with Top-1 accuracy: {best_acc:.2f}%")
            else:
                trigger_times += 1
                print(f"No improvement for {trigger_times} epochs.")
                if trigger_times >= config.patience:
                    print("Early stopping triggered.")
                    break

        scheduler.step()

    if rank == 0:
        print("Training complete. Best Top-1 accuracy:", best_acc)
        wandb.finish()

    cleanup()


def main():
    config = Config()
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs available for training.")

    mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    # Validate dataset directories before starting training
    train_dir = os.path.join(Config.data_dir, "train")
    val_dir = os.path.join(Config.data_dir, "val")

    print("Number of training classes:", len(os.listdir(train_dir)))
    print("Number of validation classes:", len(os.listdir(val_dir)))

    print("CUDA Available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    main()
