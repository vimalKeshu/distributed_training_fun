import os
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertTokenizerFast
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from model import EncoderOnlyTransformer
from news_dataset import NewsDataset
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import defaultdict

current_directory = os.path.dirname(os.path.abspath(__file__))

def log_model_weights(model, epoch):
    for name, param in model.named_parameters():
        if param.requires_grad:
            wandb.log({f"weights/{name}": wandb.Histogram(param.cpu().detach().numpy())}, step=epoch)
            wandb.log({
                f"weight_stats/{name}_mean": param.mean().item(),
                f"weight_stats/{name}_std": param.std().item(),
            }, step=epoch)
        if param.grad is not None:
            wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().detach().numpy())}, step=epoch)            

def train(model, train_loader, criterion, optimizer, device, epoch, cfg):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad() # reset the gradients
        outputs = model(input_ids, masks) # predict the output, forward pass
        loss = criterion(outputs, labels) # compute the loss
        loss.backward() # backpropagate the loss, calculate the gradient
        optimizer.step() # update the weights
        total_loss += loss.item() # calculate the total loss
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")
    log_model_weights(model, epoch)
    return avg_loss

def evaluate(epoch, model, test_loader, tokenizer, class_names, device):
    torch.cuda.empty_cache()
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    samples_table = wandb.Table(columns=["Text", "True Label", "Predicted Label"])  # Initialize table

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, masks)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Per-class tracking
            for label, pred in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
            
            # Log only 5 samples per batch
            for i in range(min(5, len(labels))):
                decoded_text = tokenizer.decode(input_ids[i].cpu().numpy(), skip_special_tokens=True)
                samples_table.add_data(decoded_text, class_names[labels[i].item()], class_names[predictions[i].item()])

    # Compute overall accuracy
    accuracy = correct / total * 100
    wandb.log({"test_accuracy": accuracy, "epoch": epoch})

    # Compute per-class accuracy
    class_acc = {class_names[i]: (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0 for i in range(len(class_names))}
    wandb.log({"eval/per_class_accuracy": class_acc, "epoch": epoch})

    # Log sample predictions
    wandb.log({f"eval/sample_predictions": samples_table, "epoch": epoch})

    # Compute & log confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    fig, _ = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    wandb.log({f"eval/confusion_matrix": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)  # Avoid memory leak

    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(f'config: {cfg}')
    wandb.init(project=cfg.wandb.project_name)
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True)) # Convert Hydra DictConfig to a standard dictionary for wandb

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_ms or torch.backends.mps.is_available else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device=device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    tokenizer:BertTokenizerFast = BertTokenizerFast.from_pretrained(cfg.model.tokenizer, cache_dir="./cache")

    dataset = load_dataset(cfg.dataset.name)
    if cfg.mode == "test" and cfg.dataset.subset_size:
        dataset["train"] = dataset["train"].select(range(cfg.dataset.subset_size))
        dataset["test"] = dataset["test"].select(range(cfg.dataset.subset_size))

    train_ds = NewsDataset(ds=dataset["train"], tokenizer=tokenizer, max_seq_len=cfg.model.max_seq_len)
    test_ds = NewsDataset(ds=dataset["test"], tokenizer=tokenizer, max_seq_len=cfg.model.max_seq_len)
    
    print(f'batch size: {cfg.training.batch_size}')

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval.batch_size)

    classes:list = cfg.dataset.classes.split(',')

    model = EncoderOnlyTransformer(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        num_layers=cfg.model.num_layers,
        num_classes=cfg.model.num_classes,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr)
    
    checkpoint_path = os.path.join(current_directory, cfg.training.checkpoint_dir_name)
    Path(f"{checkpoint_path}").mkdir(parents=True, exist_ok=True)
    latest_checkpoint = None
    initial_epoch = 0
    global_step = 0    

    if os.path.exists(checkpoint_path):
        latest_checkpoint = max([os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path)], default=None)
        if latest_checkpoint:
            print(f"Loading checkpoint from {latest_checkpoint}")
            state = torch.load(latest_checkpoint)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1 
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step'] 
    else:
        print('No model to preload, starting from scratch')

    for epoch in range(initial_epoch, cfg.training.epochs):
        train(model, train_loader, criterion, optimizer, device, epoch, cfg)
        evaluate(epoch, model, test_loader, tokenizer, classes, device)
        if (epoch + 1) % cfg.training.checkpoint_interval == 0:
            checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, checkpoint_file)            
            print(f"Checkpoint saved at {checkpoint_file}")
        global_step += 1
    wandb.finish()

if __name__ == "__main__":
    main()
