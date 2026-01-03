#!/usr/bin/env python3
"""
Kaya Dog Classifier Pipeline

End-to-end pipeline to:
1. Download training images from Bing (Kaya + other dogs)
2. Fine-tune ResNet-50 for binary classification
3. Classify YouTube frames
4. Sort into kaya/not_kaya folders and compress

Usage:
    python kaya_pipeline.py --epochs 15 --batch-size 32 --threshold 0.7
"""

import argparse
import shutil
import tarfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Enable cuDNN auto-tuner for optimal convolution algorithms
torch.backends.cudnn.benchmark = True


class ImageFolderDataset(Dataset):
    """Custom dataset for loading images from class folders."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"kaya": 1, "not_kaya": 0}

        # Collect all images from class folders
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                        self.samples.append((img_path, class_idx))

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            blank = Image.new("RGB", (224, 224), (128, 128, 128))
            if self.transform:
                blank = self.transform(blank)
            return blank, label


class FrameDataset(Dataset):
    """Dataset for loading frames to classify (no labels)."""

    def __init__(self, frames_dir: Path, transform=None):
        self.frames_dir = Path(frames_dir)
        self.transform = transform
        self.frame_paths = []

        if self.frames_dir.exists():
            for img_path in self.frames_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    self.frame_paths.append(img_path)

        self.frame_paths.sort()
        print(f"Found {len(self.frame_paths)} frames to classify")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img_path = self.frame_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, str(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            blank = Image.new("RGB", (224, 224), (128, 128, 128))
            if self.transform:
                blank = self.transform(blank)
            return blank, str(img_path)


def download_training_images(output_dir: Path, kaya_limit: int = 100, dog_limit: int = 100):
    """Download training images from Bing."""
    from bing_image_downloader import downloader

    training_dir = output_dir / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Downloading Kaya images ===")
    downloader.download(
        "Hasan Piker dog Kaya",
        limit=kaya_limit,
        output_dir=str(training_dir),
        adult_filter_off=False,
        force_replace=False,
        timeout=60,
        verbose=True
    )

    print("\n=== Downloading other dog images (negative class) ===")
    downloader.download(
        "dog",
        limit=dog_limit,
        output_dir=str(training_dir),
        adult_filter_off=False,
        force_replace=False,
        timeout=60,
        verbose=True
    )

    # Rename folders to match our class names
    kaya_src = training_dir / "Hasan Piker dog Kaya"
    kaya_dst = training_dir / "kaya"
    dog_src = training_dir / "dog"
    dog_dst = training_dir / "not_kaya"

    if kaya_src.exists() and not kaya_dst.exists():
        kaya_src.rename(kaya_dst)
        print(f"Renamed {kaya_src} -> {kaya_dst}")

    if dog_src.exists() and not dog_dst.exists():
        dog_src.rename(dog_dst)
        print(f"Renamed {dog_src} -> {dog_dst}")

    # Count images
    kaya_count = len(list(kaya_dst.glob("*"))) if kaya_dst.exists() else 0
    not_kaya_count = len(list(dog_dst.glob("*"))) if dog_dst.exists() else 0
    print(f"\nTraining data: {kaya_count} Kaya images, {not_kaya_count} not-Kaya images")

    return training_dir


def get_transforms():
    """Get training and validation transforms."""
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def build_model(num_classes: int = 2, freeze_layers: bool = True) -> nn.Module:
    """Build ResNet-50 model with modified classification head."""
    print("\n=== Building ResNet-50 model ===")

    # Load pretrained ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_layers:
        # Freeze all layers except layer4 and fc
        for name, param in model.named_parameters():
            if not name.startswith(("layer4", "fc")):
                param.requires_grad = False
        print("Frozen layers: layer1, layer2, layer3")
        print("Trainable layers: layer4, fc")

    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-4,
    save_path: Path = None
) -> nn.Module:
    """Fine-tune the model with mixed precision training."""
    print(f"\n=== Training for {epochs} epochs ===")
    print(f"Device: {device}")
    print(f"Learning rate: {lr}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision training
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*train_correct/train_total:.1f}%"
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)

                if use_amp:
                    with autocast("cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  -> New best model! Val Acc: {val_acc:.1f}%")

        scheduler.step()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation accuracy: {best_val_acc:.1f}%")

    # Save model
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_val_acc": best_val_acc,
        }, save_path)
        print(f"Model saved to {save_path}")

    return model


def classify_frames(
    model: nn.Module,
    frames_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    threshold: float = 0.7
) -> dict[str, list[Path]]:
    """Classify all frames and return paths grouped by prediction."""
    print(f"\n=== Classifying frames from {frames_dir} ===")

    _, val_transform = get_transforms()
    dataset = FrameDataset(frames_dir, transform=val_transform)

    if len(dataset) == 0:
        print("No frames found to classify!")
        return {"kaya": [], "not_kaya": []}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    model = model.to(device)
    model.eval()

    results = {"kaya": [], "not_kaya": []}
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Classifying"):
            images = images.to(device)

            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            kaya_probs = probs[:, 1]  # Probability of being Kaya

            for prob, path in zip(kaya_probs, paths):
                if prob.item() >= threshold:
                    results["kaya"].append(Path(path))
                else:
                    results["not_kaya"].append(Path(path))

    print(f"Classification results: {len(results['kaya'])} Kaya, {len(results['not_kaya'])} not Kaya")
    return results


def sort_and_compress(
    results: dict[str, list[Path]],
    output_dir: Path
) -> Path:
    """Sort frames into folders and create compressed archive."""
    print("\n=== Sorting and compressing results ===")

    classified_dir = output_dir / "classified"
    kaya_dir = classified_dir / "kaya"
    not_kaya_dir = classified_dir / "not_kaya"

    # Create directories
    kaya_dir.mkdir(parents=True, exist_ok=True)
    not_kaya_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to respective directories
    print("Copying Kaya frames...")
    for src_path in tqdm(results["kaya"], desc="Kaya"):
        dst_path = kaya_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    print("Copying not-Kaya frames...")
    for src_path in tqdm(results["not_kaya"], desc="Not Kaya"):
        dst_path = not_kaya_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    # Create tar.gz archive
    archive_path = output_dir / "classified.tar.gz"
    print(f"Creating archive: {archive_path}")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(classified_dir, arcname="classified")

    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Archive created: {archive_path} ({archive_size_mb:.1f} MB)")

    return archive_path


def main():
    parser = argparse.ArgumentParser(description="Kaya Dog Classifier Pipeline")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training and inference (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Classification threshold for Kaya (default: 0.7)")
    parser.add_argument("--kaya-images", type=int, default=100,
                        help="Number of Kaya images to download (default: 100)")
    parser.add_argument("--dog-images", type=int, default=100,
                        help="Number of other dog images to download (default: 100)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading images (use existing)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (use existing model)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected, using CPU (training will be slow)")

    # Step 1: Download training images
    if not args.skip_download:
        training_dir = download_training_images(
            output_dir,
            kaya_limit=args.kaya_images,
            dog_limit=args.dog_images
        )
    else:
        training_dir = output_dir / "training_data"
        print(f"Skipping download, using existing data in {training_dir}")

    # Step 2: Prepare datasets
    print("\n=== Preparing datasets ===")
    train_transform, val_transform = get_transforms()

    full_dataset = ImageFolderDataset(training_dir, transform=train_transform)

    if len(full_dataset) == 0:
        print("ERROR: No training images found!")
        return

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val transform to validation set
    val_dataset.dataset.transform = val_transform

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    # Step 3: Build model
    model = build_model(num_classes=2, freeze_layers=True)

    # Step 4: Train or load model
    model_path = output_dir / "kaya_model.pth"

    if not args.skip_training:
        model = train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=model_path
        )
    else:
        if model_path.exists():
            print(f"\nLoading existing model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model with validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}%")
        else:
            print(f"ERROR: No model found at {model_path}")
            return

    # Step 5: Classify frames
    frames_dir = output_dir / "frames"
    results = classify_frames(
        model,
        frames_dir,
        device,
        batch_size=args.batch_size,
        threshold=args.threshold
    )

    # Step 6: Sort and compress
    if results["kaya"] or results["not_kaya"]:
        archive_path = sort_and_compress(results, output_dir)

        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE!")
        print("=" * 50)
        print(f"Kaya frames: {len(results['kaya'])}")
        print(f"Not Kaya frames: {len(results['not_kaya'])}")
        print(f"Output archive: {archive_path}")
        print(f"Classified folders: {output_dir / 'classified'}")
    else:
        print("\nNo frames were classified. Make sure frames exist in output/frames/")


if __name__ == "__main__":
    main()
