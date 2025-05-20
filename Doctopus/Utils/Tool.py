import os
import pickle
from typing import Dict, List, Set

def save_dict_to_pickle(dictionary, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(dictionary, f)
        print(f"Dictionary saved to {file_path}")
    except Exception as e:
        print(f"Error saving dictionary to Pickle: {str(e)}")


def load_dict_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            dictionary = pickle.load(f)
        print(f"Dictionary loaded from {file_path}")
        return dictionary
    except Exception as e:
        print(f"Error loading dictionary from Pickle: {str(e)}")
        return None
    



def merge_chunks_to_set(
    chunk_dict: Dict[str, Dict[str, List[str]]]
) -> Dict[str, Set[str]]:
    """
    将每个文件的所有强关联属性字符串的 Top-K 分块合并为一个 set，返回文件路径到 set 的字典。
    参数:
        chunk_dict: 字典 {文件路径: {强关联属性字符串: [Top-K 分块列表]}}。
    返回:
        字典: {文件路径: 分块 set}。
    """
    result = {}
    
    # 遍历每个文件路径
    for file_path, assoc_chunks in chunk_dict.items():
        # 初始化该文件的分块 set
        chunk_set = set()
        
        # 遍历每个强关联属性字符串及其分块
        for assoc_attrs, chunks in assoc_chunks.items():
            if not assoc_attrs.strip() or not assoc_attrs.split("|"):
                print(f"警告: 文件 {file_path} 的强关联属性字符串 '{assoc_attrs}' 无效")
                continue
            # 添加非空分块到 set
            for chunk in chunks:
                if chunk.strip():  # 跳过空分块
                    chunk_set.add(chunk)
        
        # 存储结果
        if chunk_set:  # 仅当 set 非空时添加
            result[file_path] = chunk_set
    
    # 处理空输入
    if not result and chunk_dict:
        print("警告: 所有文件的分块 set 为空，返回空字典")
    elif not chunk_dict:
        print("警告: 输入字典为空，返回空字典")
    
    return result