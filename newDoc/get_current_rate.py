import sys
import spacy
import numpy as np
from itertools import combinations
import pandas as pd
from typing import List, Dict, Set, Tuple
import os
import glob
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from get_validation_ref import extract_attributes_from_folder, nba_list, prompt_template
from Doctopus.Utils.Tool import load_dict_from_pickle
from Doctopus.var import datasets_attributes

def get_Strong_attris(
    folder_path: str,
    attribute_snippets: Dict[str, Dict[str, Dict[str, List[str]]]],
    attributes: List[str],
    lang_model: str = "en_core_web_sm",
    jaccard_threshold: float = 0.5,
    k_clusters: int = 3,
    top_k: int = 3,
    bert_model: str = "all-MiniLM-L6-v2",
    max_tokens_per_chunk: int = 100
) -> Tuple[np.ndarray, str, Dict[str, int], int, List[Set[str]], List[List[Dict]], Dict[str, str]]:
    """
    分析 NBA 文档，提取强关联属性集合，计算共现矩阵，聚类 Supporting Fragments 并返回 Top-K 元素。
    Args:
        folder_path: 包含文档的文件夹路径
        attribute_snippets: 属性片段集合 {file_path: {attr: {"Value": value, "Supporting Fragments": [fragment, ...]}}}
        attributes: 属性列表
        lang_model: spaCy 语言模型
        jaccard_threshold: Jaccard 相似度阈值
        k_clusters: 聚类数量
        top_k: 每个属性集合返回的 Top-K 元素数量
        bert_model: BERT 模型名称
        max_tokens_per_chunk: 每个分块的最大 token 数量
    Returns:
        Tuple containing:
        - cooc_matrix: 共现矩阵（NumPy 数组）
        - markdown_table: 共现矩阵的 Markdown 表格
        - attr_counts: 属性出现次数
        - total_chunks: 总块数
        - associated_sets: 强关联属性集合
        - fragment_top_k: Top-K 片段列表
        - attri_refs: 属性集合与 Top-K 片段的映射
    """
    def load_documents_from_folder(folder_path: str) -> List[Dict]:
        """
        从文件夹读取所有 .txt 文件作为文档。
        """
        txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
        if not txt_files:
            raise FileNotFoundError(f"{folder_path} 中未找到 .txt 文件")
        documents = []
        for doc_id, file_path in enumerate(txt_files):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                documents.append({
                    "doc_id": doc_id,
                    "content": content,
                    "file_path": file_path
                })
        return documents

    def preprocess_documents(documents: List[Dict], lang_model: str, max_tokens: int) -> List[Dict]:
        """
        将文档集合按固定 token 数量分块，确保不截断句子。
        """
        try:
            nlp = spacy.load(lang_model)
        except OSError:
            raise ImportError(f"未找到 spaCy 模型 '{lang_model}'。请使用 'python -m spacy download {lang_model}' 安装")
        
        chunks = []
        for doc in documents:
            doc_id = doc["doc_id"]
            doc_text = doc["content"]
            doc_nlp = nlp(doc_text)
            
            current_chunk = []
            current_token_count = 0
            chunk_id = 0
            
            for sent in doc_nlp.sents:
                sent_tokens = len(sent)
                
                # If adding this sentence exceeds max_tokens, finalize the current chunk
                if current_token_count + sent_tokens > max_tokens and current_chunk:
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": " ".join(current_chunk).strip()
                    })
                    current_chunk = []
                    current_token_count = 0
                    chunk_id += 1
                
                # Add the sentence to the current chunk
                current_chunk.append(sent.text.strip())
                current_token_count += sent_tokens
            
            # Add the final chunk if it exists
            if current_chunk:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": " ".join(current_chunk).strip()
                })
        
        return chunks

    def match_attributes_in_chunks(
        chunks: List[Dict],
        attribute_snippets: Dict[str, Dict[str, Dict[str, List[str]]]],
        attributes: List[str]
    ) -> List[Dict]:
        """
        基于属性片段集合，检测每个块中的属性。
        """
        chunk_attributes = []
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            chunk_text = chunk["text"]
            found_attrs = set()
            
            file_path = next((doc["file_path"] for doc in documents if doc["doc_id"] == doc_id), None)
            
            if file_path in attribute_snippets:
                for attr in attributes:
                    if attr in attribute_snippets[file_path]:
                        fragments = attribute_snippets[file_path][attr]["Supporting Fragments"]
                        for fragment in fragments:
                            if fragment and fragment in chunk_text:
                                found_attrs.add(attr)
            
            if found_attrs:
                chunk_attributes.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk["chunk_id"],
                    "attributes": found_attrs
                })
        
        return chunk_attributes

    def build_cooccurrence_matrix(chunk_attributes: List[Dict], attributes: List[str]) -> np.ndarray:
        """
        构建共现矩阵。
        """
        N = len(attributes)
        cooc_matrix = np.zeros((N, N), dtype=int)
        attr_to_index = {attr: idx for idx, attr in enumerate(attributes)}
        
        for chunk in chunk_attributes:
            attrs = chunk["attributes"]
            for attr_i, attr_j in combinations(attrs, 2):
                i = attr_to_index[attr_i]
                j = attr_to_index[attr_j]
                cooc_matrix[i][j] += 1
                cooc_matrix[j][i] += 1
        
        return cooc_matrix

    def matrix_to_markdown(cooc_matrix: np.ndarray, attributes: List[str]) -> str:
        """
        将共现矩阵转换为 Markdown 表格。
        """
        df = pd.DataFrame(cooc_matrix, index=attributes, columns=attributes)
        markdown = df.to_markdown(index=True)
        return markdown

    def find_strongly_associated_attributes(
        cooc_matrix: np.ndarray,
        attributes: List[str],
        attr_counts: Dict[str, int],
        threshold: float
    ) -> List[Set[str]]:
        """
        从共现矩阵中提取关联性强的属性集合。
        """
        N = len(attributes)
        jaccard_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                cooc = cooc_matrix[i, j]
                union = attr_counts[attributes[i]] + attr_counts[attributes[j]] - cooc
                if union > 0:
                    jaccard = cooc / union
                else:
                    jaccard = 0
                jaccard_matrix[i, j] = jaccard
                jaccard_matrix[j, i] = jaccard
        
        G = nx.Graph()
        G.add_nodes_from(attributes)
        for i, attr_i in enumerate(attributes):
            for j, attr_j in enumerate(attributes[i + 1:], start=i + 1):
                if jaccard_matrix[i, j] >= threshold:
                    G.add_edge(attr_i, attr_j, weight=jaccard_matrix[i, j])
        
        connected_components = list(nx.connected_components(G))
        return [set(component) for component in connected_components]

    def cluster_fragments(
        attribute_snippets: Dict[str, Dict[str, Dict[str, List[str]]]],
        associated_sets: List[Set[str]],
        k: int,
        top_k: int,
        bert_model: str
    ) -> List[List[Dict]]:
        """
        对每个强关联属性集合的 Supporting Fragments（不去重）进行聚类（使用 BERT 嵌入），并返回 Top-K 个元素。
        """
        try:
            model = SentenceTransformer(bert_model)
        except Exception as e:
            raise ImportError(f"无法加载 BERT 模型 '{bert_model}'：{e}。请确保已安装 sentence-transformers 并下载模型")
        
        results = []
        for attr_set in associated_sets:
            fragments = []
            for file_path in attribute_snippets:
                for attr in attr_set:
                    if attr in attribute_snippets[file_path]:
                        frags = attribute_snippets[file_path][attr]["Supporting Fragments"]
                        if not frags:
                            continue
                        for frag in frags:
                            if frag and frag.strip():
                                fragments.append(frag)
            
            if not fragments:
                print(f"警告：属性集合 {attr_set} 未找到任何非空 Supporting Fragments")
                results.append([])
                continue
            
            k_actual = min(k, len(fragments))
            if k_actual < k:
                print(f"警告：属性集合 {attr_set} 的 Fragments 数量 {len(fragments)} 小于 k={k}，将 k 调整为 {k_actual}")
            
            try:
                fragment_embeddings = model.encode(fragments, convert_to_numpy=True, show_progress_bar=False)
            except Exception as e:
                print(f"属性集合 {attr_set} 的 BERT 嵌入生成失败：{e}")
                results.append([])
                continue
            
            kmeans = KMeans(n_clusters=k_actual, random_state=42)
            cluster_labels = kmeans.fit_predict(fragment_embeddings)
            centroids = kmeans.cluster_centers_
            
            fragment_similarities = []
            for idx, (frag, embedding, cluster_id) in enumerate(zip(fragments, fragment_embeddings, cluster_labels)):
                similarity = cosine_similarity([embedding], [centroids[cluster_id]])[0][0]
                fragment_similarities.append({
                    "fragment": frag,
                    "cluster_id": cluster_id,
                    "similarity": similarity
                })
            
            fragment_similarities.sort(key=lambda x: x["similarity"], reverse=True)
            top_k_results = fragment_similarities[:min(top_k, len(fragment_similarities))]
            
            results.append(top_k_results)
        
        return results

    # 主逻辑
    # 检查文件数量
    txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    if len(txt_files) < 3:
        raise ValueError(f"{folder_path} 中预期至少 3 个 .txt 文件，实际找到 {len(txt_files)} 个")
    
    # 加载文档
    documents = load_documents_from_folder(folder_path)
    
    # 分块
    chunks = preprocess_documents(documents, lang_model, max_tokens_per_chunk)
    
    # 匹配属性
    chunk_attributes = match_attributes_in_chunks(chunks, attribute_snippets, attributes)
    
    # 构建共现矩阵
    cooc_matrix = build_cooccurrence_matrix(chunk_attributes, attributes)
    
    # 转换为 Markdown 表格
    markdown_table = matrix_to_markdown(cooc_matrix, attributes)
    
    # 计算属性出现次数
    attr_counts = {attr: 0 for attr in attributes}
    for chunk in chunk_attributes:
        for attr in chunk["attributes"]:
            attr_counts[attr] += 1
    
    # 提取强关联属性集合
    associated_sets = find_strongly_associated_attributes(cooc_matrix, attributes, attr_counts, jaccard_threshold)
    
    # 聚类 Supporting Fragments
    fragment_top_k = cluster_fragments(attribute_snippets, associated_sets, k_clusters, top_k, bert_model)
    
    # 生成 attri_refs
    attri_refs = {}
    for attr_set, top_k_frags in zip(associated_sets, fragment_top_k):
        if not top_k_frags:
            continue
        ref = [frag['fragment'] for frag in top_k_frags]
        refs = ".".join(ref)
        attri_refs["|".join(list(attr_set))] = refs
    
    # 打印输出（与原代码一致）
    print("共现矩阵（NumPy 数组）：")
    print(cooc_matrix)
    print("\n共现矩阵（Markdown 表格）：")
    print(markdown_table)
    print("\n属性出现次数：")
    print(attr_counts)
    print(f"\n总块数：{len(chunks)}")
    print("\n关联性强的属性集合：")
    for i, attr_set in enumerate(associated_sets, 1):
        print(f"集合 {i}: {attr_set}")
        
    print("\n每个强关联属性集合的 Top-K Fragments（使用 BERT 嵌入）：")
    for i, (attr_set, top_k_frags) in enumerate(zip(associated_sets, fragment_top_k), 1):
        print(f"\n属性集合 {i}: {attr_set}")
        if not top_k_frags:
            print("  无 Top-K Fragments")
        for j, frag in enumerate(top_k_frags, 1):
            print(f"    Fragment: {frag['fragment']}")
    print(attri_refs)
    
    return cooc_matrix, markdown_table, attr_counts, len(chunks), associated_sets, fragment_top_k, attri_refs

if __name__ == "__main__":
    folder_path = "C:/Users/jjli74/Desktop/nba/nba_sample"
    attributes = datasets_attributes["nba"]
    # 假设 res 已定义为 attribute_snippets
    valipath = "C:/Users/jjli74/ljj/Doctopus-main/storedata/NBA.pkl"
    val = load_dict_from_pickle(valipath)

    res = val["validation"]
    attribute_snippets = res  # 请确保 res 已正确定义
    print(attribute_snippets)
    
    cooc_matrix, markdown_table, attr_counts, total_chunks, associated_sets, fragment_top_k, attri_refs = get_Strong_attris(
        folder_path=folder_path,
        attribute_snippets=attribute_snippets,
        attributes=attributes,
        lang_model="en_core_web_sm",
        jaccard_threshold=0.5,
        k_clusters=3,
        top_k=3,
        bert_model="all-MiniLM-L6-v2",
        max_tokens_per_chunk=64
    )