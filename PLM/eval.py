import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from transformers import T5ForSequenceClassification, T5Tokenizer


def load_eval_data(file_path):
    """加载评估数据集"""
    contents = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 处理可能存在的不规范分割情况
            parts = line.strip().rsplit("\t", 1)
            if len(parts) != 2:
                continue
            content, label = parts
            contents.append(content)
            labels.append(int(label))
    return contents, labels


def prepare_eval_dataset(file_path, tokenizer, batch_size=16):
    """预处理评估数据"""
    contents, true_labels = load_eval_data(file_path)
    
    # Tokenize文本
    encoded_data = tokenizer(
        contents,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    
    # 创建Tensor数据集
    dataset = TensorDataset(
        encoded_data["input_ids"],
        encoded_data["attention_mask"],
        torch.tensor(true_labels),
    )
    
    # 创建数据加载器
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model, dataloader, device):
    """执行模型评估"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            
            # 获取预测结果（使用阈值0进行二值化）
            preds = (outputs.logits.squeeze() > 0).long().cpu().numpy()

            
            # 检查 preds 的形状和类型
            if preds.ndim == 0:  # 如果 preds 是标量（0 维数组）
                all_preds.append(int(preds))  # 将标量转换为整数并添加到列表
            else:  # 如果 preds 是数组
                all_preds.extend(preds.tolist())  # 将数组转换为列表并扩展
            
            all_labels.extend(labels.cpu().numpy().tolist())
            
    
    return all_labels, all_preds


def print_metrics(true_labels, predictions):
    """打印评估指标"""
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print(f"Precision: {precision_score(true_labels, predictions):.4f}")
    print(f"Recall: {recall_score(true_labels, predictions):.4f}")
    print(f"F1 Score: {f1_score(true_labels, predictions):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))


def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--eval_file", type=str, required=True, help="评估数据路径")
    parser.add_argument("--model_dir", type=str, required=True, help="训练好的模型目录")
    parser.add_argument("--batch_size", type=int, default=16, help="评估批次大小")
    args = parser.parse_args()

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForSequenceClassification.from_pretrained(args.model_dir).to(device)

    # 准备数据
    eval_loader = prepare_eval_dataset(args.eval_file, tokenizer, args.batch_size)

    # 执行评估
    true_labels, predictions = evaluate_model(model, eval_loader, device)
    print("global truth:",true_labels)
    print("pre",predictions)

    # 输出结果
    print_metrics(true_labels, predictions)


if __name__ == "__main__":
    main()
    
"""
python eval.py \
  --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/test1.txt \          
  --model_dir /home/lijiajun/train_T5/save_model/fin/ep7 \
  --batch_size 4

#fin
python eval.py \
  --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/fin/test1v1.txt \
  --model_dir /home/lijiajun/train_T5/save_model/total/ep7 \
  --batch_size 4 

#wikiart
python eval.py \
  --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/wikiart/test1v1.txt \
  --model_dir /home/lijiajun/train_T5/save_model/total/ep7 \
  --batch_size 4 

#wikiart sigle

python eval.py \
  --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/wikiart/test1v1.txt \
  --model_dir /home/lijiajun/train_T5/save_model/wikiart/ep8 \
  --batch_size 4 

  
python eval.py --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/fin/test1v1.txt --model_dir /home/lijiajun/train_T5/save_model/fin/ep8 --batch_size 4

#fin
CUDA_VISIBLE_DEVICES=1 python eval.py --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/fin/test1v1.txt --model_dir /home/lijiajun/train_T5/save_model/fin/ep7 --batch_size 4


## legal   0.95
CUDA_VISIBLE_DEVICES=1 python eval.py --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/legal/test1v1.txt --model_dir /home/lijiajun/train_T5/save_model/legal/ep10 --batch_size 4


## wikiart
CUDA_VISIBLE_DEVICES=1 python eval.py --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/wikiart/test.txt --model_dir /home/lijiajun/train_T5/save_model/wikiart/ep8 --batch_size 4


## wikiart
CUDA_VISIBLE_DEVICES=1 python eval.py --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/nba/test.txt --model_dir /home/lijiajun/train_T5/save_model/nba/ep8  --batch_size 4


"""