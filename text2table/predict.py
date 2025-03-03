import torch
from transformers import T5ForSequenceClassification, T5Tokenizer

def predict_samples(samples, model_path, batch_size=16, device=None, return_probs=False):
    """
    使用T5分类模型进行预测，返回得分或概率值

    参数：
    - samples (list of str): 输入文本样本的列表
    - model_path (str): 训练好的模型路径
    - batch_size (int): 批量大小，默认为16
    - device (str): 设备，可选 'cuda' 或 'cpu'
    - return_probs (bool): 是否返回概率值（默认为 True）

    返回：
    - scores (list of float): 每个样本的预测分数（logits 或概率）
    """

    # 设备配置
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型和 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # 处理输入样本
    encoded_data = tokenizer(
        samples,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded_data["input_ids"].to(device)
    attention_mask = encoded_data["attention_mask"].to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze().cpu().numpy()

    # 如果 return_probs=True，则返回概率值（使用 sigmoid 归一化）
    if return_probs:
        scores = torch.sigmoid(torch.tensor(logits)).tolist()
    else:
        scores = logits.tolist()  # 直接返回原始 logits 分数
        scores = [ (score + 1)*0.5 for score in scores]
    
    

    return scores  # 返回每个样本的得分或概率


# =======  示例  =======
if __name__ == "__main__":
    # 假设的样本数据
    sample_texts = ["Birth_Country [sep] Kounellis was born in Piraeus, Greece in 1936.",
                    "Birth_City [sep] Kounellis was born in Piraeus, Greece in 1936.",
                    "Art_Movement [sep] A key figure associated with Arte Povera , he studied at the Accademia di Belle Arti in Rome. ",
                    "Age [sep] Saul Steinberg (June 15, 1914 – May 12, 1999) was a Romanian and American cartoonist and illustrator, best known for his work for The New Yorker, most notably View of the World from 9th Avenue.",
]

    # 设定模型路径
    model_directory = "/home/lijiajun/train_T5/save_model/wikiart/ep8"

    # 调用预测函数
    predictions = predict_samples(sample_texts, model_directory)

    # 输出结果
    print("Predictions:", predictions)
