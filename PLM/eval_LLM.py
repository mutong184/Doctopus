import argparse
import numpy as np
import openai
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score, accuracy_score

def predict_LLm(sample,model_name):
    prompt = """
    Task: Given a sentence and an attribute, determine if the attribute's value can be derived from the sentence. Return `1` if yes, `0` if no.

    Examples:
    1. Sentence: "This phone has a battery capacity of 5000mAh."
    Attribute: battery capacity
    Output: 1

    Now, evaluate:
    Sentence: {input_sentence}
    Attribute: {input_attribute}
    Output:
    """

    # 从输入的 sample 中提取 attribute 和 sentence
    attribute = sample.split("[sep]")[0].strip()
    sentence = sample.split("[sep]")[1].strip()

    # 设置你的 OpenAI API 密钥
    client = openai.OpenAI(base_url="https://api.gptsapi.net/v1",
                            api_key="sk-Hwv0998366e81e4a9a4cd0cd7f38890f3f7417d34134xAbD")
    
    # 格式化 prompt 内容，替换其中的 {sentence} 和 {attribute}
    formatted_prompt = prompt.format(input_sentence=sentence, input_attribute=attribute)

    # 调用 OpenAI API 获取预测结果
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": formatted_prompt},
        ]
    )

    response_text = completion.choices[0].message.content.strip()
    ans = response_text.split(":")[-1].strip()

    return ans



def evaluate_performance(file_path,model_name):
    # 读取文件并解析数据
    groundtruth = []
    predictions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 分割数据和标签
            try:
                data_part, label = line.strip().rsplit('\t', 1)
                    # 调用预测函数（需自行实现）
                pred = predict_LLm(data_part.strip(),model_name)  # 注意：这里要传入处理后的数据部分

                # 确保预测结果是整数类型
                gd = int(label)
                pd = int(pred)
                groundtruth.append(gd)
                predictions.append(pd)
            except ValueError:
                print(f"格式错误行: {line}")
                print(f"{line} prdict result {label}")
                continue
             
    # 计算指标
    f1 = f1_score(groundtruth, predictions)
    acc = accuracy_score(groundtruth, predictions)
    
    return f1, acc

def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--eval_file", type=str, required=True, help="评估数据路径")
    parser.add_argument("--model_name", type=str, required=True, help="训练好的模型目录")
    args = parser.parse_args()
    
    
    file_del = args.eval_file
    # file = "/home/lijiajun/train_T5/unique_valiadtionset/train_dataset/fin/test1v1.txt"
    f1, acc = evaluate_performance(file_del,"gpt-4o")
    
    print("f1",f1)
    print("acc",acc)
    
if __name__ == "__main__":
    main()
    
""""
## fin eva 
python eval_LLM.py --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/fin/test1v1.txt --model_name gpt4o


## legal
nohup python -u eval_LLM.py  --eval_file /home/lijiajun/train_T5/unique_valiadtionset/train_dataset/legal/test1v1.txt --model_name gpt4o  > legal1v1_model_gpt4o.log 2>&1 &

"""
    
    
