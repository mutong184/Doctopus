import itertools
import os
import json
from collections import Counter, defaultdict
import pickle
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from openai import OpenAI
from nltk.tokenize import sent_tokenize
import tiktoken
from Utils.similiar_chunk import get_similiar_chunks

from embedding import get_embeddings
# nltk.download('punkt_tab')



def get_evidence(filepath):
    with open(filepath,"r") as file:
        content = file.read()

    # 将文本按行拆分
    lines = content.strip().split("\n")

    # 生成组合，每三行作为一组
    combinations = [lines[i:i+2] for i in range(0, len(lines), 2)]

    return combinations

def get_evidence_explain(filepath):
    with open(filepath,"r") as file:
        content = file.read()

    # 将文本按行拆分
    lines = content.strip().split("\n")

    # 生成组合，每三行作为一组
    combinations = [lines[i:i+3] for i in range(0, len(lines), 3)]

    return combinations



# api_key = "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"
# prompt = "Say ok"
# result = query_chatgpt(api_key, prompt)
# print(result)
def query_chatgpt(api_key, prompt):
    client = OpenAI(
        api_key=api_key
    )

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ]
    )

    return {
        "answer": completion.choices[0].message,
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens
    }






def file_to_sentences(file_path):
    """
    读取文件并将其内容拆分成单独的句子。
    参数:
    file_path (str): 要读取的文件路径。
    返回:
    list: 从文件中提取的句子列表。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 将内容分词成句子
        sentences = sent_tokenize(content)
        return sentences,content
    except FileNotFoundError:
        print("未找到文件。")
        return []
    except Exception as e:
        print(f"发生错误：{e}")
        return []



def Files2sentences(file_list,storepath):
    """
    读取文件并将其内容拆分成单独的句子 并存储在storepath文件中。
    参数:
    file_list: 要读取的文件完整路径列表
    返回:
    storepath: dict 文件对应的句子的列表存储在storepath 中
    """
    file2sentences = {}
    file2contents = {}


    if os.path.exists(f"{storepath}/file2sentences.pkl"):
        with open(f"{storepath}/file2sentences.pkl", "rb") as f:
            file2chunks = pickle.load(f)
            print(f"{storepath}/file2sentences.pkl 加载成功")
        with open(f"{storepath}/file2contents.pkl", "rb") as f:
            file2contents = pickle.load(f)
            print(f"{storepath}/file2contents.pkl 加载成功")
        return file2sentences,file2contents
    
    for filepath in file_list:
        sentences,content = file_to_sentences(filepath)
        file2sentences[filepath] = sentences
        file2contents[filepath] = content

    with open(f"{storepath}/file2sentences.pkl", "wb") as f:
        pickle.dump(file2sentences, f)
    with open(f"{storepath}/file2contents.pkl", "wb") as f:
        pickle.dump(file2contents, f)


    
    return file2sentences,file2chunks


def Files2chunks(file_list,storepath,chunk_size):
    """
    读取文件并将其内容拆分成单独的句子 并存储在storepath文件中。
    参数:
    file_list: 要读取的文件完整路径列表
    返回:
    storepath: dict 文件对应的句子的列表存储在storepath 中
    """

    if os.path.exists(f"{storepath}/size{len(file_list)}_chunkSize{chunk_size}_file2chunks.pkl"):
        with open(f"{storepath}/size{len(file_list)}_chunkSize{chunk_size}_file2chunks.pkl", "rb") as f:
            file2chunks = pickle.load(f)
            print(f"{storepath}/size{len(file_list)}_chunkSize{chunk_size}_file2chunks.pkl  load!")
        with open(f"{storepath}/size_{len(file_list)}_chunkSize{chunk_size}_file2contents.pkl", "rb") as f:
            file2contents = pickle.load(f)
            print(f"{storepath}/size_{len(file_list)}_chunkSize{chunk_size}_file2contents.pkl load !")
        return file2chunks, file2contents
    
    file2chunks = {}
    file2contents = {}
    for file in tqdm(file_list, total=len(file_list), desc="Chunking files"):
        content, chunks = chunk_file(
            file, 
            chunk_size=chunk_size, 
        )
        if not chunks:
            continue
        
        file2chunks[file] = chunks
        file2contents[file] = content
    # return file2chunks, file2contents
    
    if not os.path.exists(storepath):
        os.mkdir(storepath)
    with open(f"{storepath}/size{len(file_list)}_chunkSize{chunk_size}_file2chunks.pkl", "wb") as f:
        pickle.dump(file2chunks, f)
        print(f"{storepath}/size{len(file_list)}_chunkSize{chunk_size}_file2chunks.pkl 存储成功")
    with open(f"{storepath}/size_{len(file_list)}_chunkSize{chunk_size}_file2contents.pkl", "wb") as f:
        pickle.dump(file2contents, f)
        print(f"{storepath}/size_{len(file_list)}_chunkSize{chunk_size}_file2contents.pkl存储成功")
    return file2chunks,file2contents


def chunk_file(file, chunk_size=5000):
    content =  get_file_contents(file)
    if chunk_size > 1:
        content, chunks = get_txt_parse(content, chunk_size=chunk_size)
    else:
        chunks = get_similiar_chunks(file,chunk_size)

    return content, chunks

def get_file_contents(file):
    text = ''
    if file.endswith(".swp"):
        return text
    try:
        with open(file) as f:
            text = f.read()
    except:
        with open(file, "rb") as f:
            text = f.read().decode("utf-8", "ignore")
    return text


# GENERIC TXT --> CHUNKS
def get_txt_parse(content, chunk_size=5000):
    # convert to chunks
    chunks = content.split("\n")
    clean_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size:
            sub_chunks = chunk.split(". ")
            clean_chunks.extend(sub_chunks)
        else:
            clean_chunks.append(chunk)

    chunks = clean_chunks.copy()
    clean_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_size:
            sub_chunks = chunk.split(", ")
            clean_chunks.extend(sub_chunks)
        else:
            clean_chunks.append(chunk)

    final_chunks = []
    cur_chunk = []
    cur_chunk_size = 0
    for chunk in clean_chunks:
        if cur_chunk_size + len(chunk) > chunk_size:
            final_chunks.append("\n".join(cur_chunk))
            cur_chunk = []
            cur_chunk_size = 0
        cur_chunk.append(chunk)
        cur_chunk_size += len(chunk)
    if cur_chunk:
        final_chunks.append("\n".join(cur_chunk))

    return content, final_chunks




def list_all_file_paths(directory_path):
    """
    给定一个文件路径，列出该路径下所有的文件路径。

    参数:
    directory_path (str): 需要遍历的目录路径。

    返回:
    list: 包含所有文件路径的列表。
    """
    file_paths = []  # 用于存储所有文件路径的列表
    # 使用 os.walk() 遍历目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 将文件的完整路径添加到列表中
            file_paths.append(os.path.join(root, file))
    
    return file_paths


def get_most_similarity(target_sentence, sentences):
    target_embedding = get_embeddings([target_sentence])[0]
    embeddings = get_embeddings(sentences)
    max_similarity = torch.nn.functional.cosine_similarity(target_embedding, embeddings, dim = -1)
    most_similar_sentence = max_similarity.argmax()
    return sentences[most_similar_sentence]




def get_topk_similarity(target_sentence_emb, sentences_emb, topk,chunks):
    
    similarities = torch.nn.functional.cosine_similarity(target_sentence_emb.unsqueeze(0), sentences_emb, dim=-1)
    
    # 获取与目标向量最相似的 topk 索引
    if topk <= len(chunks):
        topk_values, topk_indices = torch.topk(similarities, topk)
    else:
        return chunks

    topk_indices = topk_indices.tolist()    
    # 根据索引返回对应的句子和相似度
    topk_sentences = [chunks[i] for i in topk_indices]
    # topk_results = list(zip(topk_sentences, topk_values))

    combined_results = []
    # # 两两组合的结果
    # for i in range(2,k+1):
    #     combinations = itertools.combinations(topk_sentences, i)
    #     for combo in combinations:
    #         combined_sentence = " ".join(combo)
    #         # 获取组合句子的嵌入并计算相似度
    #         combined_embedding = get_embeddings([combined_sentence])[0]
    #         combined_similarity = torch.nn.functional.cosine_similarity(target_embedding, combined_embedding, dim=-1).item()
    #         combined_results.append((combined_sentence, combined_similarity))

    # return topk_results + combined_results
    return topk_sentences




def get_LLM_extractions(zero_shot_cot,
    file2chunks, 
    sample_files, 
    attribute, 
    manifest_session
):
     file2results = {}
     for i, (file) in tqdm(enumerate(sample_files),
        total=len(sample_files),
        desc=f"Extracting attribute {attribute} using LM",
     ):
          chunks = file2chunks[file] 
          extractions = []
          for chunk_num, chunk in enumerate(chunks):
               if zero_shot_cot:
                    PROMPTS = PROMPT_EXTRACTION_WITH_LM_ZERO_SHOT[0]
               else:
                    PROMPTS = PROMPT_EXTRACTION_WITH_LM_CONTEXT[0]
                

               prompt = PROMPTS.format(attribute=attribute, chunk=chunk)
               try:
                    extraction, num_toks =  apply_prompt(
                    prompt, 
                    manifest=manifest_session, 
                    max_toks=1024, 
                )  
               except:
                    print(f"Failed to extract {attribute} for {file}")
                    continue
               
               
               pattern = r'\[(.*?)\]'
               match = re.search(pattern, extraction,flags=re.DOTALL)
    
               if match:
                    raw_ex = extraction = match.group(1)
               else:
                    print("error:,",extraction)


          file2results[file] = extractions
     return file2results



# get the token nums
def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    if model_name == "gpt-4o" or model_name == "gpt-4o-mini":
        encoding_name = "o200k_base"
    else:
        encoding_name = "cl100k_base"

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_acc_from_similiary(topk_scores,topk_sentences,baseline_sentence):
    #ToDo
    return topk_scores


# print(num_tokens_from_string("tiktoken is great!", "gpt-4o"))





# target_sentence = "I love programming."
# sentences = [
#     "Programming is fun.",
#     "I enjoy coding.",
#     "Debugging can be challenging.",
#     "Python is great."
# ]
# k = 3

# # 测试
# result = get_topk_similarity(target_sentence, sentences, k)
# for content, score in result:
#     print(f"Content: {content}, Similarity Score: {score:.4f}")


import json

def extract_keys_from_json(json_file_path):
    """
    从 JSON 文件中提取所有的属性名（键名）。
    
    Args:
        json_file_path (str): JSON 文件路径。
        
    Returns:
        list: 包含所有属性名的列表。
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 获取所有值的属性名（子字典的键）
    keys = set()  # 使用 set 防止重复
    for item in data.values():
        keys.update(item.keys())
    
    return list(keys)

# 示例调用
# json_file_path = 'path/to/your/json_file.json'  # 替换为实际的 JSON 文件路径
# keys_list = extract_keys_from_json(json_file_path)
# print(keys_list)

import json
import os
import re


import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# a = num_tokens_from_string("tiktoken is great!", "cl100k_base")
# print(a)

def calculate_avg_tokens(folder_path,filenum):
    total_tokens = 0
    file_count = 0

    # 遍历文件夹中的所有文件
    i = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if i >= filenum:
                continue
            file_path = os.path.join(root, file)
            if file.endswith(".txt"):  # 只处理文本文件
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()  # 读取文件内容
                    tokens = num_tokens_from_string(content,"cl100k_base")
                    total_tokens+= tokens
                except Exception as e:
                    print(f"无法处理文件: {file_path}, 错误: {e}")
            i = i +1

    # 计算平均 token 数
    return total_tokens

filepath = "/data/lijiajun/dataset/NBA/data"
print(calculate_avg_tokens(filepath,97))
