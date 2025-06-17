

from collections import Counter, defaultdict
import csv
import json
from textwrap import dedent
from attr import asdict
import numpy as np
import openai
import pandas as pd
from dataclasses import dataclass
from pydantic import BaseModel,Field
from typing import List
import json

from embedding import get_embeddings

wikiartList = [
               "Name: Full name of the artist,or leave empty if not applicable", 
               "Birth_Date: Date in %Y/%-m/%-d format, e.g., 1839/1/9,or leave empty if not applicable",
               "Death_Date: Date in %Y/%-m/%-d format, e.g., 1836/1/1, or leave empty if not applicable",
               "Age: Age the age of the artist e.g.,87, or leave empty if not applicable",
               "Birth_Country: [Country where the artist was born] eg,.Netherlands,or leave empty if not applicable",
               "Death_Country: [Country where the artist died, or leave empty if not applicable] eg,. Switzerland,or leave empty if not applicable",
               "Birth_City: [City where the artist was born] ,or leave empty if not applicable",
               "Death_City: [City where the artist died, or leave empty if not applicable]",
               "Field: [Primary artistic field, e.g., Painting, Sculpture],or leave empty if not applicable",
               "Genre: [Primary genre of work, e.g., Portrait, Landscape],or leave empty if not applicable",
               "Marriage: [Marital status or details, or leave empty if not applicable,as succinct as possible]",
               "Art_Movement: [Art movement contributed to, e.g., Impressionism],or leave empty if not applicable"
               ]

nba_list = [
         "name: the name of the nba player", 
         "birth_date: the birthday of the nba player e.g., June 10, 1959",
         "nationality: the nation of the nba player.eg., American,or leave empty if not applicable",
         "age: the age of the nba player. eg., 34,this year is 2024 ",
         "team: the current team of the nba player.eg.,Lokomotiv Kuban,or leave empty if not applicable",
         "position: player’s position on the court. eg., foeword,or leave empty if not applicable",
         "draft_pick: the draft pick of the nba player eg.,24. ,or leave empty if not applicable",
         "draft_year: the the draft year of the nba player. eg.,2015,or leave empty if not applicable",
         "college: the college of the nba player. eg.,University of Notre Dame,or leave empty if not applicable",
         "NBA_championships:How many times has this NBA player won NBA championships. eg.,1,or 0 empty if not applicable",
         "mvp_awards: How many times has this NBA player won MVP award. eg.,2,or leave 0 if not applicable",
         "olympic_gold_medals: How many times has this NBA player won an Olympic gold medal. eg.,2,or leave 0 if not applicable",
         "FIBA_World_Cup: How many times has this NBA player won FIBA World Cup. eg,.1,or leave 0 if not applicable",
               ]

legal_list = ["judge_name  VARCHAR, judge_name is the name of the judge presiding over the case. e.g., arshall J.or leave empty if not applicable", 
               "plaintiff VARCHAR, plaintiff is the name of the person or organization initiating the case. e.g., [Australian Competition and Consumer Commission].or leave empty if not applicable",
               "defendant VARCHAR, defendant is the name of the person or entity being sued or accused. e.g., [Narnia Investments Pty Ltd].or leave empty if not applicable",
               "hearing_year DATE, hearing_year is the date when the hearing began (first day if multiple days).in %Y/%-m/%-d format. e.g., 2009/4/22.or leave empty if not applicable",
               "judgment_year DATE, judgment_year is the date when the judgment was delivered (first day if multiple days).in %Y/%-m/%-d format. e.g.,  2009/4/22.or leave empty if not applicable",
               "case_type ENUM('Criminal Case', 'Civil Case', 'Commercial Case', 'Administrative Case'), case_type identifies the type of the case. e.g., Criminal",
               "verdict ENUM('Guilty', 'Not Guilty', 'Others', 'Dismissed', 'Approved', 'Others'), verdict indicates the court's decision. e.g., Guilty",
               "counsel_for_applicant VARCHAR, counsel_for_applicant is the name of the applicant’s lawyer. e.g., Mr I Faulkner SC.or leave empty if not applicable",
               "counsel_for_respondent VARCHAR, counsel_for_respondent is the name of the respondent’s lawyer. e.g., Mr D Barclay.or leave empty if not applicable",
               "nationality_for_applicant VARCHAR, nationality_for_applicant is the applicant’s nationality. e.g., Australia.or leave empty if not applicable",
               "hearing_location VARCHAR, hearing_location is the location where the hearing occurred. e.g., Hobart.or leave empty if not applicable",
               "evidence ENUM('1', '0'), evidence specifies whether evidence was provided in the case.1 for yes, 0 for no. e.g., 1",
               "first_judge ENUM('1', '0'), first_judge indicates if it was the first judgment or a subsequent one. 1 for yes, 0 for no. e.g., 1"
               ]

fin_list = ['Company Name: The name of the company, add “Inc.” or “Company” as a suffix. or leave empty if not applicable',
               'Founding Year: The year the company was founded (e.g., “2003”). or leave empty if not applicable', 
               'Headquarters Location: The headquarters address, as it is written in the original text.or leave empty if not applicable', 
               'Industry Type: The industry category of the company (Categories include:Energy andResources; Industrial and Manufacturing; Real Estate and Construction; Retaiand Consumer;)Multiple categories can be selected.or leave empty if not applicable', 
               'Product/Service Type: The company’s products, which can be inferred from the opening of each paragraph.or leave empty if not applicable', 
               'Customer Type: The type of customers the company serves.or leave empty if not applicable',
               'Tax Policy: Whether the company has a special tax policy. If yes, fill 1; if no, fill 0.', 
               'Branches Office: Whether the company has branch offices. If yes, fill 1; if no, fill 0.',
               'Brand Number: The number of brands the company has, in numerical form.if no, fill 0.', 
               'Changed Name: Whether the company has changed its name. If yes, fill 1; if no, fill 0.', 
               'Segments Number: The number of departments or divisions the company has, in numerical form.if no, fill 0.']


datasets_attributes = {
"fin" :['Company Name', 
               'Founding Year', 
               'Headquarters Location',
               'Industry Type', 
               'Product/Service Type', 
               'Customer Type', 
               'Tax Policy',
               'Branches Office', 
               'Brand Number', 
               'Changed Name', 
               'Segments Number'],

"legal":["judge_name", 
            "plaintiff",
            "defendant",
            "hearing_year",
            "judgment_year",
            "case_type",
            "verdict",
            "counsel_for_applicant",
            "counsel_for_respondent",
            "nationality_for_applicant",
            "hearing_location",
            "evidence",
            "first_judge"
            ],

"nba":[
            "name", 
            "birth_date",
            "nationality",
            "age",
            "team",
            "position",
            "draft_pick",
            "draft_year",
            "college",
            "NBA_championships",
            "mvp_awards",
            "olympic_gold_medals",
            # "FIBA_World_Cup",
            ],

"wikiart":[
            "Name", 
            "Birth_Date",
            "Death_Date",
            "Age",
            "Birth_Country",
            "Death_Country",
            "Birth_City",
            "Death_City",
            "Field",
            "Genre",
            "Marriage",
            "Art_Movement"
            ]
}



info_dataset= {
        "fin":{
            "total_len":1500,
            "length":258,
             "given_cost_raw":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            "given_acc_raw":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
            # "given_cost_raw":[0.05,0.1,0.12,0.14,0.16,0.18,0.2],
             "datadir_small": "/data/lijiajun/dataset/fin/fin_small",
             "datadir": "/data/lijiajun/dataset/fin/data",
            #  "groundtruth": "/data/lijiajun/dataset/fin/table.json",
            "groundtruth":"/data/lijiajun/dataset/fin/table_all.json",
             "eva":"/data/lijiajun/eva/fin/output.json",
             "openie6":"/home/lijiajun/openie6_exp_data/fin/predict_new.json",
            # "debertv3":"/data/lijiajun/debertv3/debertv3/fin.json",
            "debertv3":"/home/lijiajun/DebertV3/debertv3/fin_predictions_new.json",
             "rag":"/home/lijiajun/RAG/fin",
             "array": fin_list,
             "topk": 9, # topk 是一个整数值
             "chunksize":0,
             "exist_model_dir":"/home/lijiajun/train_T5/save_model/fin/ep8",  
             "valadation":"/home/lijiajun/train_T5/unique_valiadtionset/val_and_trainset/fin_train.json",
              "valadation_dir":"/home/lijiajun/train_T5/unique_valiadtionset/validation/fin",
        },
        "legal":{
            "total_len":600,
            "length":5926,
            "given_acc_raw":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
            "given_cost_raw":[0.71112,0.8,1,1.5,2,2.5,3,3.5,4,4.5,5],
            "datadir_small": "/data/lijiajun/dataset/legal/legal_small",
            "datadir": "/data/lijiajun/dataset/legal/legal-tity",
            # "datadir_small": "/data/lijiajun/dataset/legal/legal_small",
            "groundtruth": "/data/lijiajun/dataset/legal/table.json",
            "eva":"/data/lijiajun/eva/legal/output.json",
            "openie6":"/home/lijiajun/openie6_exp_data/legal/predict_new.json",
            "debertv3":"/home/lijiajun/DebertV3/debertv3/legal_predictions_new.json",
            "rag":"/home/lijiajun/RAG/legal",
            "array": legal_list,
            "topk": 9,  # topk 是一个整数值
            "chunksize":0.7,
            "exist_model_dir":"/home/lijiajun/train_T5/save_model/legal/ep10",  
            "valadation":"/home/lijiajun/train_T5/unique_valiadtionset/val_and_trainset/legal_train.json",
            "valadation_dir":"/home/lijiajun/train_T5/unique_valiadtionset/validation/legal",
        },
        "wikiart":{
            "total_len":1000,
            "length":714,
            "given_acc_raw":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
            "given_cost_raw":[0.1428,0.2,0.25,0.38,0.5,0.75,1.0,1.25,1.5,1.75,2],
            "datadir_small": "/data/lijiajun/dataset/wikiart/wikiart_small",
            "datadir": "/data/lijiajun/dataset/wikiart/data_wikiart",
            "groundtruth": "/data/lijiajun/dataset/wikiart/table.json",
            "eva":"/data/lijiajun/eva/wikiart/output.json",
            "openie6":"/home/lijiajun/openie6_exp_data/wikiart/predict_new.json",
            "debertv3":"/home/lijiajun/DebertV3/debertv3/wikiart_predictions_new.json",
            "array": wikiartList,
            "topk": 9,  # topk 是一个整数值
            "chunksize":0.4 ,
            "exist_model_dir":"/home/lijiajun/train_T5/save_model/wikiart/ep8",    
            "valadation":"/home/lijiajun/train_T5/unique_valiadtionset/val_and_trainset/wikiart_train.json",
            "valadation_dir":"/home/lijiajun/train_T5/unique_valiadtionset/validation/wikiart",
        },
        "nba":{
            "total_len":100,
            "length":1645,
            "given_acc_raw":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
            "given_cost_raw":[0.0329,0.04,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4],
            # "given_cost_raw":[0.0329,0.04,0.05,0.075,0.1,0.15],
            "datadir_small": "/data/lijiajun/dataset/NBA/nba_small",
            "datadir": "/data/lijiajun/dataset/NBA/data",
            "groundtruth": "/data/lijiajun/dataset/NBA/table.json",
            "eva":"/data/lijiajun/eva/nba/output.json",
            "openie6":"/home/lijiajun/openie6_exp_data/nba/predict_new.json",
            "debertv3":"/home/lijiajun/DebertV3/debertv3/nba_predictions_new.json",   
            "rag":"/home/lijiajun/RAG/nba",     
            "array": nba_list,
            "topk": 9,  # topk 是一个整数值
            "chunksize":0.4,
            "exist_model_dir":"/home/lijiajun/train_T5/save_model/nba/ep8",
            "valadation_dir":"/home/lijiajun/train_T5/unique_valiadtionset/validation/nba",     
            "valadation":"/home/lijiajun/train_T5/unique_valiadtionset/val_and_trainset/nba_train.json",
        },  
    }


@dataclass
class Legal(BaseModel):
    judge_name:str
    plaintiff:str
    defendant:str
    hearing_year:str
    judgment_year:str
    case_type:str
    verdict:str
    counsel_for_applicant:str
    counsel_for_respondent:str
    nationality_for_applicant:str
    hearing_location:str
    evidence:str
    first_judge:str
 
       
selected_attribute_prompt = """
    You will be provided with content from  an  document.
    Your goal will be to get the values of the attibutes following the attibute provided.
    Here is a description of the parameters:
    {attribute_explain}
    """   



dataset_raw={
        "legal": {
            "datadir": "/data/lijiajun/dataset/legal/legal-tity",
            "groundtruth": "/data/lijiajun/dataset/legal/table.json",
            "array": legal_list,
            "topk": 9,  # topk 是一个整数值
            "chunksize":0.7
        },
        # wikiart Average chunks number: 15
        "wikiart": {
            "datadir": "/data/lijiajun/dataset/wikiart/data_wikiart",
            "groundtruth": "/data/lijiajun/dataset/wikiart/table.json",
            "array": wikiartList,
            "topk": 9,  # topk 是一个整数值
            "chunksize":0.4
        },
        # nba Average chunks number: 33
        "nba": {
            "datadir": "/data/lijiajun/dataset/NBA/data",
            "groundtruth": "/data/lijiajun/dataset/NBA/table.json",
            "array": nba_list,
            "topk": 9,  # topk 是一个整数值
            "chunksize":0.4
        },
        
        "fin": {
            "datadir": "/data/lijiajun/dataset/fin/data",
            "groundtruth": "/data/lijiajun/dataset/fin/table.json",
            "array": fin_list,
            "topk": 9, # topk 是一个整数值
            "chunksize":0
        }
}



def text_f1(preds=[], golds=[]):
    """Compute average F1 of text spans.
    Taken from Squad without prob threshold for no answer.
    """
    
    total_f1 = 0
    total_recall = 0
    total_prec = 0
    f1s = []
    if  not preds or not golds:
        return 0,0,0,0
    for pred, gold in zip(preds, golds):
        if isinstance(pred, list):
            pred = ' '.join(pred)  # Example way to convert list to string
        if isinstance(gold, list):
            gold = ' '.join(gold)  # Example way to convert list to string
        pred_toks = str(pred).split()
        gold_toks = str(gold).split()
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
    prec = total_prec / len(golds)
    recall =  total_recall/len(golds)
    f1_avg = total_f1 / len(golds)
    f1_median = np.percentile(f1s, 50)     
    return f1_avg,f1_median,prec,recall

# def detect_file_encoding(file_path):
#     with open(file_path, 'rb') as file:
#         raw_data = file.read(10000)  # 读取部分内容检测
#         result = chardet.detect(raw_data)
#     return result['encoding']



from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_accuracy_batch(a, b, threshold=0.8):
    """
    使用批处理计算实体列表 a 和 b 之间的匹配准确度，基于语义相似度。

    参数：
    a (list): 实体列表 a
    b (list): 实体列表 b
    threshold (float): 相似度阈值，默认为 0.8，表示如果相似度大于等于 0.8，就认为是相同的实体

    返回：
    float: 匹配的准确度，介于 0 和 1 之间
    """
    # 确保 a 和 b 列表长度相同
    min_len = min(len(a), len(b))
    
    # 获取每个实体的嵌入
    embeddings_a = get_embeddings(a[:min_len])  # 仅取较短列表的长度部分
    embeddings_b = get_embeddings(b[:min_len])
    
    # 批量计算余弦相似度矩阵
    similarities = cosine_similarity(embeddings_a.detach().numpy(), embeddings_b.detach().numpy())
    
    # 统计匹配的数量
    matching_count = np.sum(similarities >= threshold)
    
    # 计算准确度，匹配对数除以两个列表中较长的长度
    accuracy = matching_count / max(len(a), len(b))
    return accuracy


def get_EM_dic(ground_truth,predict):
    all_file = list(predict.keys())
    # print(all_file)
    attributes = list(predict[all_file[0]].keys())

    data = defaultdict(list)

    for attribute in attributes:
        predict_list = []
        ground_list = []
        for file in all_file:
            try:
                p = str(predict[file][attribute])
                q = str(ground_truth[file][attribute])
                predict_list.append(p.lower())
                ground_list.append(q.lower())
       
            except :
                print(f"{file} ,{attribute},predict {predict[file][attribute]},grundtruth,{ground_truth[file][attribute]} failed")
                pass
        if len(predict_list) ==0  or  len(ground_list) ==0:
            f1 = 0
            af1 = 0
            prec= 0
            recall = 0
        else:
            f1 = calculate_accuracy_batch(predict_list,ground_list)
        # print(attribute)
        # print(f1,af1)
        data["Attribute"].append(attribute)
        data["avarag_F1_Score"].append(f1)

def get_f1_dataset(ground_truth,predict):

    all_file = list(predict.keys())
    # print(all_file)
    attributes = list(predict[all_file[0]].keys())

    data = defaultdict(list)

    for attribute in attributes:
        predict_list = []
        ground_list = []
        for file in all_file:
            try:
                p = str(predict[file][attribute])
                q = str(ground_truth[file][attribute])
                predict_list.append(p.lower())
                ground_list.append(q.lower())
       
            except :
                print(f"{file} ,{attribute},predict {predict[file][attribute]},grundtruth,{ground_truth[file][attribute]} failed")
                pass
        if len(predict_list) ==0  or  len(ground_list) ==0:
            f1 = 0
            af1 = 0
            prec= 0
            recall = 0
        else:
            f1, af1,prec,recall = text_f1(predict_list,ground_list)
        # print(attribute)
        # print(f1,af1)
        data["Attribute"].append(attribute)
        data["avarag_F1_Score"].append(f1)
        data["median_F1_Score"].append(af1)
        data["prec"].append(prec)
        data["recall"].append(recall)


    df = pd.DataFrame(data)

    # 反转表格
    df_transposed = df.set_index("Attribute").transpose()

    # 输出反转后的表格
    print(df_transposed)
    return sum(data["avarag_F1_Score"])/len(data["avarag_F1_Score"])

def get_f1_dic(ground_truth,predict):

    all_file = list(predict.keys())
    # print(all_file)
    attributes = list(predict[all_file[0]].keys())

    data = defaultdict(lambda: defaultdict())

    for attribute in attributes:
        predict_list = []
        ground_list = []
        for file in all_file:
            try:
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
            prec= 0
            recall = 0
        else:
            f1, af1,prec,recall = text_f1(predict_list,ground_list)

        data[attribute] = f1
    return data
       


    df = pd.DataFrame(data)

    # 反转表格
    df_transposed = df.set_index("Attribute").transpose()

    # 输出反转后的表格
    # print(df_transposed)
    return sum(data["avarag_F1_Score"])/len(data["avarag_F1_Score"])

def match_ratio(list_a, list_b):
    min_len = min(len(list_a), len(list_b))
    matches = sum(a == b for a, b in zip(list_a, list_b))
    return matches / min_len if min_len > 0 else 0.0

def get_01_dic(ground_truth,predict):
    all_file = list(predict.keys())
    # print(all_file)
    attributes = list(predict[all_file[0]].keys())

    data = defaultdict(list)

    for attribute in attributes:
        predict_list = []
        ground_list = []
        for file in all_file:
            try:
                p = str(predict[file][attribute])
                q = str(ground_truth[file][attribute])
                predict_list.append(p.lower())
                ground_list.append(q.lower())
       
            except :
                print(f"{file} ,{attribute},predict {predict[file][attribute]},grundtruth,{ground_truth[file][attribute]} failed")
                pass
        if len(predict_list) ==0  or  len(ground_list) ==0:
            f1 = 0
            af1 = 0
            prec= 0
            recall = 0
        else:
            f1 = match_ratio(predict_list,ground_list)
        # print(attribute)
        # print(f1,af1)
        data["Attribute"].append(attribute)
        data["avarag_F1_Score"].append(f1)


    df = pd.DataFrame(data)

    return sum(data["avarag_F1_Score"])/len(data["avarag_F1_Score"])


def create_dynamic_class(attribute_list: List[str]) -> type:
    # 动态创建一个类，继承自 BaseModel
    class DynamicModel(BaseModel):
        # 动态添加字段类型注解
        __annotations__ = {attr: str for attr in attribute_list}

    # 给动态生成的类添加字段，使用 set_attr 添加 Field 到每个字段
    for attr in attribute_list:
        # 使用 setattr 来动态地将字段添加到类中，并使用 Field 来标记字段是必填项
        setattr(DynamicModel, attr, Field(..., description=f"{attr} field"))

    return DynamicModel



def extract_attributes_from_text(text, attributes_obj,llm_prompt,model):

    client = openai.OpenAI(base_url="https://api.gptsapi.net/v1",
        api_key=""
        )

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": dedent(llm_prompt)},
            {"role": "user", "content": text}
        ],
        response_format=attributes_obj,
    )

    answer =  response.choices[0].message.parsed

    return answer.dict()



# 读取策略文件
def load_strategy(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    


def save_dict_to_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4)  # indent=4 用于美化输出（添加缩进）
    


def csv_to_dict_without_file(csv_file_path):
    """
    将CSV文件转化为字典对象，key为file列的值，value为该行的字典形式，但移除file列的值
    """
    result_dict = {}
    
    with open(csv_file_path, mode='r', encoding='utf-8',errors='ignore') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            print(row)
            file_key = row['\ufeffID']  # 获取file列作为key
            row.pop('\ufeffID')  # 移除file列
            result_dict[file_key+".txt"] = row  # 将剩余的字ID典存入value中
    
    return result_dict


    
    
    
    
    







