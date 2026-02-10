import torch
from tqdm.auto import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import logging  # 确保导入 logging 以记录日志


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=100, patience=10, output_dir=None, device=None):

    global best_model_path
    if device is None:  # 如果没有传入 device，则自动选择
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_model_params = None

    # 训练/微调
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        # 训练阶段
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_accuracies.append(val_accuracy)

        epoch_duration = time.time() - epoch_start

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 打印日志
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], "
                     f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, "
                     f"Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%, "
                     f"LR: {current_lr:.6f}, Duration: {epoch_duration:.2f} seconds")

        # 调度器步进
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # 早停
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_params = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

        # 更新最佳模型权重
        if best_model_params is not None and output_dir is not None:
            model.load_state_dict(best_model_params)
            best_model_path = os.path.join(output_dir, 'best_model_weights.pth')
            torch.save(model.state_dict(), best_model_path)
 
    # 绘制并保存损失曲线
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    if any(val_losses):
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    logging.info(f"最佳模型已保存到 {best_model_path}")

    return model, train_losses, val_losses
