import os
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.utils.data import DataLoader


def get_dataset_embdding(storepath, file2content, batch_size=1):
    if os.path.exists(f"{storepath}/size{len(file2content)}_file2emb.pkl"):
        with open(f"{storepath}/size{len(file2content)}_file2emb.pkl", "rb") as f:
            file2content = pickle.load(f)
        return file2content

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = list(file2content.keys())
    contents = list(file2content.values())

    modepath = "/home/lijiajun/RAG/nvidia/NV-Embed-v2"
    model = AutoModel.from_pretrained(modepath, trust_remote_code=True, torch_dtype=torch.float16)

    model = model.to(device)  # 将模型迁移到 GPU

    # 分批处理内容以减少显存使用
    dataloader = DataLoader(contents, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    max_length = 32768
    for batch in dataloader:
        # 确保内容输入在设备上
        passage_embeddings = model.encode(
            batch,
            instruction="",
            max_length=10240,
            device=device,
        )
        # 归一化嵌入并迁回 CPU
        passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1).to("cpu")
        all_embeddings.append(passage_embeddings)

    # 合并所有批次的嵌入
    all_embeddings = torch.cat(all_embeddings, dim=0)


    # 将嵌入与文件名对应
    embedding_dict = {files[i]: all_embeddings[i] for i in range(len(files))}

    if not os.path.exists(storepath):
        os.mkdir(storepath)
    with open(f"{storepath}/size{len(file2content)}_file2emb.pkl", "wb") as f:
        pickle.dump(embedding_dict, f)

    return embedding_dict


# 示例文件内容字典
file_dict = {
    "file1.txt": "This is the content of the first file.",
    "file2.txt": "Python is a powerful programming language.",
    "file3.txt": "Data science and machine learning are exciting fields.",
    "file4.txt": "Artificial intelligence is transforming industries.",
    "file5.txt": "This is a test file with random content.",
    "file6.txt": "Deep learning is a subset of machine learning.",
    "file7.txt": "Natural language processing is an interesting area of AI.",
    "file8.txt": "Big data analytics help in making informed decisions.",
    "file9.txt": "Cloud computing provides scalable solutions for businesses.",
    "file10.txt": "Cybersecurity is critical in the digital era.",
    "file11.txt": "Blockchain technology ensures secure transactions.",
    "file12.txt": "The Internet of Things connects physical devices globally.",
    "file13.txt": "Quantum computing is a revolutionary technology.",
    "file14.txt": "Virtual reality is enhancing user experiences.",
    "file15.txt": "Augmented reality combines the real and virtual worlds.",
    "file16.txt": "Robotics is automating industrial processes.",
    "file17.txt": "5G networks are enabling faster communication.",
    "file18.txt": "Renewable energy is essential for a sustainable future.",
    "file19.txt": "Genetic engineering has significant medical applications.",
    "file20.txt": "Space exploration expands human understanding of the universe.",
}

# 调用函数并打印结果
# embedding_dict = get_dataset_embdding("/home/lijiajun/RAG/", file_dict, batch_size=4)



def get_sentence_embdding(file2content,max_length=1024,batch_size=8):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    contents = file2content

    modepath = "/home/lijiajun/RAG/nvidia/NV-Embed-v2"
    model = AutoModel.from_pretrained(modepath, trust_remote_code=True, torch_dtype=torch.float16)

    model = model.to(device)  # 将模型迁移到 GPU

    # 分批处理内容以减少显存使用
    dataloader = DataLoader(contents, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    for batch in dataloader:
        # 确保内容输入在设备上
        passage_embeddings = model.encode(
            batch,
            instruction="",
            max_length=max_length,
            device=device,
        )
        # 归一化嵌入并迁回 CPU
        passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1).to("cpu")
        all_embeddings.append(passage_embeddings)

    # 合并所有批次的嵌入
    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings



