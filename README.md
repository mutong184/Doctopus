<div align="center">
  <h1>Doctopus: Budget-aware Structural Table Extraction from
Unstructured Documents</h1>
  <h3></h3>
</div>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-structure">Structure</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-data">Data</a> •
  <a href="#-results">Results</a>
</p>

<div align="center">
  <img src="assets/architecture.png" width="750">
</div>

## 🌟 Overview
Doctopus combines LLMs with non-LLM strategies to achieve a good trade-off: 
1. *Muti-strategy combined**: a system designed for accurate attribute extraction from unstructured documents with a user-specified cost constraint through combinning LLMs with non-LLM strategies to achieve a good trade-off
2. **New Benchmark**:We have built a comprehensive benchmark including 4 document sets with various characteristics,
as well as the ground truth values that are manually labeled using 1000 human hours.
3. **Improve Quality**: Doctopus can improve the quality by 11% given the same cost constraint

---

## 📂 Structure <a name="-structure"></a>

```text
.
├── configs/             # 超参数配置
│   ├── base.yaml        # 基础配置
│   └── experiments/     # 实验专用配置
├── core/                # 核心算法
│   ├── engine.py        # 执行引擎
│   ├── optimizers/      # 优化算法实现
│   └── utils/           # 辅助工具
├── data/                # 数据管道
│   ├── loaders/         # 数据加载器
│   └── preprocess.py    # 预处理脚本
├── docs/                # 文档资源
├── experiments/         # 实验记录
├── outputs/             # 生成结果
├── requirements.txt     # 依赖项
└── run.py               # 主入口
