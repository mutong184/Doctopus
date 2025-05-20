import os
import pickle
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel


from text2emb import get_sentence_embdding


def process_files(tuple_data_dir, emb_data_dir):
    """
    处理数据集文件，将三元组转换为嵌入并保存
    """
    os.makedirs(emb_data_dir, exist_ok=True)

    for filename in tqdm(os.listdir(tuple_data_dir), desc="Processing files", unit="file"):
        if not filename.endswith(".txt"):
            continue
        # 读取文件
        source_file = os.path.join(tuple_data_dir, filename)
        with open(source_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f.readlines()]
        sentences = [sentence[6:] for sentence in sentences if len(sentence) > 6]
        # 获取嵌入
        embeddings = get_sentence_embdding(sentences)

        # 保存嵌入
        sentence_to_embedding_dic = {sentence: emb for sentence, emb in zip(sentences, embeddings)}

        target_file = os.path.join(emb_data_dir, filename)
        with open(target_file, "wb") as file:
            pickle.dump(sentence_to_embedding_dic, file)

if __name__ == "__main__":
    ## legal
    # stime = time.time()
    # tuple_dir = "/data/lijiajun/openie6/legal-tity/legal_tity_tuple"
    # emb_dir = "/data/lijiajun/openie6/wikiart/legal_tity_emb"
    # process_files(tuple_dir, emb_dir)
    # etime = time.time()
    # print(f"total time: {etime - stime}")
    
    
    ## fin
    stime = time.time()
    tuple_dir = "/data/lijiajun/openie6/fin/fin_tity_tuple"
    emb_dir = "/data/lijiajun/openie6/fin/fin_tity_emb"
    process_files(tuple_dir, emb_dir)
    etime = time.time()
    print(f"total time: {etime - stime}")


"""
import os
import pickle
from tqdm import tqdm
from cluster.text2emb import get_sentence_embdding
import argparse


def process_files(tuple_data_dir, emb_data_dir):

    os.makedirs(emb_data_dir, exist_ok=True)

    for filename in tqdm(os.listdir(tuple_data_dir), desc="Processing files", unit="file"):
        if not filename.endswith(".txt"):
            continue
        # 读取文件
        source_file = os.path.join(tuple_data_dir, filename)
        with open(source_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f.readlines()]

        # 获取嵌入
        embeddings = get_sentence_embdding(sentences)

        # 保存嵌入
        sentence_to_embedding_dic = {sentence: emb for sentence, emb in zip(sentences, embeddings)}

        target_file = os.path.join(emb_data_dir, filename)
        with open(target_file, "wb") as file:
            pickle.dump(sentence_to_embedding_dic, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset files to generate embeddings.")
    parser.add_argument("--tuple_dir", type=str, required=True, help="Path to the directory containing tuple files.")
    parser.add_argument("--emb_dir", type=str, required=True, help="Path to the directory to store embedding files.")
    parser.add_argument("--log_file", type=str, default=None, help="Path to the log file.")

    args = parser.parse_args()

    # 设置日志输出
    if args.log_file:
        import sys
        log_file = open(args.log_file, "a")
        sys.stdout = log_file
        sys.stderr = log_file

    print(f"Starting embedding generation.")
    print(f"Tuple directory: {args.tuple_dir}")
    print(f"Embedding directory: {args.emb_dir}")

    process_files(args.tuple_dir, args.emb_dir)

    print(f"Embedding generation completed.")

    if args.log_file:
        log_file.close()


"""


