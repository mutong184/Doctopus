
from collections import Counter, defaultdict
import csv
import json
import sys
import time
import numpy as np
import pandas as pd

def text_f1(preds=[], golds=[]):
    """Compute average F1 of text spans.
    Taken from Squad without prob threshold for no answer.
    """
    total_f1 = 0
    total_recall = 0
    total_prec = 0
    f1s = []
    for pred, gold in zip(preds, golds):
        if isinstance(pred, list):
            pred = ' '.join(pred)  # Example way to convert list to string
        if isinstance(gold, list):
            gold = ' '.join(gold)  # Example way to convert list to string
        pred_toks = pred.split()
        gold_toks = gold.split()
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            total_f1 += int(gold_toks == pred_toks)
            f1s.append(int(gold_toks == pred_toks))
        elif num_same == 0:
            total_f1 += 0
            f1s.append(0)
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            total_f1 += f1
            total_recall += recall
            total_prec += precision
            f1s.append(f1)
    f1_avg = total_f1 / len(golds)
    f1_median = np.percentile(f1s, 50)     
    return f1_avg,f1_median


def get_f1_dataset(groundtruth_file,predict_file,rate):
    with open(groundtruth_file, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    with open(predict_file, "r", encoding="utf-8") as file:
        predict = json.load(file)
        

    all_file = list(predict.keys())
    
    ten_percent_index = max(1, int(len(all_file) * rate))  # 计算10%的元素数量，确保至少取一个元素
    all_file=  all_file[:ten_percent_index]  # 返回前10%的元素
    # print(all_file)
    attributes = list(predict[all_file[0]].keys())

    data = defaultdict(list)

    for attribute in attributes:
        predict_list = []
        ground_list = []
        for file in all_file:
            try:
                if(str(predict[file][attribute]) + str(ground_truth[file][attribute])):
                    p = str(predict[file][attribute])
                    q = str(ground_truth[file][attribute])
                    predict_list.append(p.lower())
                    ground_list.append(q.lower())
                    
            except :
                # print(f"{file} failed")
                pass
        if len(predict_list) ==0  or  len(ground_list) ==0:
            f1 = 0
            af1 = 0
        else:
            f1, af1 = text_f1(predict_list,ground_list)
        # print(attribute)
        # print(f1,af1)
        data["Attribute"].append(attribute)
        data["avarag_F1_Score"].append(f1)
        data["median_F1_Score"].append(af1)


    df = pd.DataFrame(data)

    # 反转表格
    df_transposed = df.set_index("Attribute").transpose()

    # 输出反转后的表格
    print(df_transposed)
    print("f1:")
    print(np.mean(data["avarag_F1_Score"]))
    return df_transposed

def text_f1(preds=[], golds=[]):
    """Compute average F1 of text spans.
    Taken from Squad without prob threshold for no answer.
    """
    total_f1 = 0
    total_recall = 0
    total_prec = 0
    f1s = []
    for pred, gold in zip(preds, golds):
        if isinstance(pred, list):
            pred = ' '.join(pred)  # Example way to convert list to string
        if isinstance(gold, list):
            gold = ' '.join(gold)  # Example way to convert list to string
        pred_toks = pred.split()
        gold_toks = gold.split()
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            total_f1 += int(gold_toks == pred_toks)
            f1s.append(int(gold_toks == pred_toks))
        elif num_same == 0:
            total_f1 += 0
            f1s.append(0)
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            total_f1 += f1
            total_recall += recall
            total_prec += precision
            f1s.append(f1)
    recal_avg =   total_recall/ len(golds) 
    prec_avg = total_prec / len(golds) 
    f1_avg = total_f1 / len(golds)
    f1_median = np.percentile(f1s, 50)     
    return f1_avg,f1_median,prec_avg,recal_avg

def get_f1_dataset_from_var(ground_truth,predict):
        

    all_file = list(predict.keys())
    
    ten_percent_index = max(1, int(len(all_file) * 1))  # 计算10%的元素数量，确保至少取一个元素
    all_file=  all_file[:ten_percent_index]  # 返回前10%的元素
    # print(all_file)
    attributes = list(predict[all_file[0]].keys())

    data = defaultdict(list)

    for attribute in attributes:
        predict_list = []
        ground_list = []
        for file in all_file:
            try:
                if(str(predict[file][attribute]) + str(ground_truth[file][attribute])):
                    p = str(predict[file][attribute])
                    q = str(ground_truth[file][attribute])
                    predict_list.append(p)
                    ground_list.append(q)
                    
            except Exception as e:
                # print(f"{file} failed,",e)
                pass
        if len(predict_list) ==0  or  len(ground_list) ==0:
            f1 = 0
            af1 = 0
            prec = 0
            recal = 0
        else:
            f1, af1,prec,recal = text_f1(predict_list,ground_list)
        # print(attribute)
        # print(f1,af1)
        data["Attribute"].append(attribute)
        data["avarag_F1_Score"].append(f1)
        data["median_F1_Score"].append(af1)
        data["prec"].append(prec)
        data["recal"].append(recal)


    df = pd.DataFrame(data)

    # 反转表格
    df_transposed = df.set_index("Attribute").transpose()

    # 输出反转后的表格
    print(df_transposed)
    print("f1:")
    print(np.mean(data["avarag_F1_Score"]))
    # return df_transposed
       




def get_f1_dataset_table(groundtruth_file,predict_file,rate):
    with open(groundtruth_file, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    with open(predict_file, "r", encoding="utf-8") as file:
        predict = json.load(file)
        

    all_file = list(predict.keys())
    
    ten_percent_index = max(1, int(len(all_file) * rate))  # 计算10%的元素数量，确保至少取一个元素
    all_file=  all_file[:ten_percent_index]  # 返回前10%的元素
    # print(all_file)
    attributes = list(predict[all_file[0]].keys())

    data = defaultdict(list)

    for attribute in attributes:
        predict_list = []
        ground_list = []
        for file in all_file:
            try:
                if(str(predict[file][attribute]) + str(ground_truth[file][attribute])):
                    p = str(predict[file][attribute])
                    q = str(ground_truth[file][attribute])
                    predict_list.append(p)
                    ground_list.append(q)
                    
            except Exception as e:
                # print(f"{file} failed,",e)
                pass
        if len(predict_list) ==0  or  len(ground_list) ==0:
            f1 = 0
            af1 = 0
            prec = 0
            recal = 0
        else:
            f1, af1,prec,recal = text_f1(predict_list,ground_list)
        # print(attribute)
        # print(f1,af1)
        data["Attribute"].append(attribute)
        data["avarag_F1_Score"].append(f1)
        data["median_F1_Score"].append(af1)
        data["prec"].append(prec)
        data["recal"].append(recal)


    df = pd.DataFrame(data)

    # 反转表格
    df_transposed = df.set_index("Attribute").transpose()

    # 输出反转后的表格
    print(df_transposed)
    print("f1:")
    print(np.mean(data["avarag_F1_Score"]))
    # return df_transposed



def get_f1_dataset_dic(groundtruth_file, predict_file, rate,input_token_dict):
    with open(groundtruth_file, "r", encoding="utf-8") as file:
        ground_truth = json.load(file)

    with open(predict_file, "r", encoding="utf-8") as file:
        predict = json.load(file)

    # 获取所有文件，并仅取前 rate% 的数据
    all_files = list(predict.keys())
    num_selected_files = max(1, int(len(all_files) * rate))  # 确保至少选1个
    selected_files = all_files[:num_selected_files]  

    # 获取属性列表（基于第一个文件）
    attributes = list(predict[selected_files[0]].keys())

    # 结果存储字典
    f1_results = {}

    for attribute in attributes:
        predict_list, ground_list = [], []

        for file in selected_files:
            try:
                pred_value = str(predict[file].get(attribute, ""))
                true_value = str(ground_truth[file].get(attribute, ""))
                
                if pred_value and true_value:  # 确保两者都不为空
                    predict_list.append(pred_value)
                    ground_list.append(true_value)
            except Exception as e:
                # print(f"处理文件 {file} 时出错: {e}")
                pass

        # 计算 F1 分数
        if predict_list and ground_list:
            f1, af1, prec, recal = text_f1(predict_list, ground_list)
        else:
            f1, af1, prec, recal = 0, 0, 0, 0  # 如果没有数据，F1 设为 0
        
        # 存储结果
        f1_results[attribute] = {
            "f1": f1,
            "precision": prec,
            "recall": recal,
            "token": input_token_dict[attribute]
        }

    return f1_results