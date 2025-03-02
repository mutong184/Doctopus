<div align= "center">
    <h1>octopus: Budget-aware Structural Table Extraction from Unstructured Documents</h1>
</div>
<p align="center">
  <a href="#-struct">Folder Structure</a> •
  <a href="#-getstart">Getting Start</a> •
  <a href="#-datasets">Datasets</a> •
</p>


<br>
<div align="center">
<img src="imgs/framework.jpg" width="1000px">
<!-- <iframe src="imgs/framework.jpg" width="100%" height="600px"></iframe> -->
</div>
<br>
To fulfill the potential great value of unstructured documents, it is
critical to extract structural data (e.g., attributes) from them, which
can benefit various applications such as analytical SQL queries
and decision-making. Multiple strategies, such as pre-trained lan-
guage models (PLMs), can be employed for this task. However,
these methods often struggle to achieve high-quality results, partic-
ularly when dealing with attribute extraction that requires intricate
reasoning or semantic comprehension. Recently, large language
models (LLMs) have proven to be effective in extracting attributes
but incur substantial costs caused by token consumption, making
them impractical for large-scale document set.
To best trade off quality and cost, we present Doctopus, a sys-
tem designed for accurate attribute extraction from unstructured
documents with a user-specified cost constraint. Overall, Doctopus
combines LLMs with non-LLM strategies to achieve a good trade-
off. First, the system employs an index-based approach to efficiently
identify and process only relevant text chunks, thereby reducing the
LLM cost. Afterwards, it further estimates the quality of multiple
strategies for each attribute. Finally, based on the cost and esti-
mated quality, Doctopus dynamically selects the optimal strategies
through budget-aware optimization. We have built a comprehensive
benchmark including 4 document sets with various characteristics,
as well as the ground truth values that are manually labeled us-
ing 1000 human hours. Extensive experiments on the benchmark
have demonstrated that compared with state-of-the-art baselines,
Doctopus can improve the quality by 11% given the same cost
constraint.
<span id="-struct"></span>

## <img src="/Users/lijiajun/text-table/imgs/folder1.jpg"  width="42" height="42"> Folder Structure


```

.
├── imgs                      # Directory for storing images or visualizations
├── join                      # Directory for join algorithms
│   ├── mult_table_join.ipynb # Notebook for multi-table join
│   └── two_table_join.ipynb  # Notebook for two-table join
├── cal-sel.py                # Function to calculate selectivity of one filter based on sampled data
├── llm.py                    # Functions related to large language models (LLMs)
├── query_optimization.py     # Executes query optimization and runs the query for each document
├── util_order.py             # Utilities for parsing and sorting filters
├── segment_index.py          # Build segment index
├── document_index.py         # Build document index
├── main.py                   # Main script controlling the system's overall flow
├── README.md                 # Project overview and instructions
└── requirements.txt          # Python dependencies for the project
```

The full version of the technical report is [here](./Full_version.pdf). 

<br>
<span id="-getstart"></span>

##  <img src="imgs/run.jpg" width="42" height="42"> Getting Started

This is an example of how to set up DOCTOPUS locally. To get a local copy up, running follow these simple example steps.

### Prerequisites

To install the required packages, you can create a conda environment:

```sh
conda create --name DOCTOPUS python=3.9
```To fulfill the potential great value of unstructured documents, it is
critical to extract structural data (e.g., attributes) from them, which
can benefit various applications such as analytical SQL queries
and decision-making. Multiple strategies, such as pre-trained lan-
guage models (PLMs), can be employed for this task. However,
these methods often struggle to achieve high-quality results, partic-
ularly when dealing with attribute extraction that requires intricate
reasoning or semantic comprehension. Recently, large language
models (LLMs) have proven to be effective in extracting attributes
but incur substantial costs caused by token consumption, making
them impractical for large-scale document set.
To best trade off quality and cost, we present Doctopus, a sys-
tem designed for accurate attribute extraction from unstructured
documents with a user-specified cost constraint. Overall, Doctopus
combines LLMs with non-LLM strategies to achieve a good trade-
off. First, the system employs an index-based approach to efficiently
identify and process only relevant text chunks, thereby reducing the
LLM cost. Afterwards, it further estimates the quality of multiple
strategies for each attribute. Finally, based on the cost and esti-
mated quality, Doctopus dynamically selects the optimal strategies
through budget-aware optimization. We have built a comprehensive
benchmark including 4 document sets with various characteristics,
as well as the ground truth values that are manually labeled us-
ing 1000 human hours. Extensive experiments on the benchmark
have demonstrated that compared with state-of-the-art baselines,
Doctopus can improve the quality by 11% given the same cost
constraint.

then use pip to install -r requirements.txt

```sh
pip install -r requirements.txt
```

The following commands will quickly get you started with our optimization code.
```sh
python test.py
```


And we have provided a complete demo process to help you become familiar with our code.
Before that, please manually modify the BERT path in the get_embed function within the files document_index.py and segment_index.py, as shown in the picture.

<img src="imgs/robert.jpg" width="50px" width="20" > 

```sh
python demo.py
```

### <img src="imgs/run.png"  > Run DOCTOPUS

**Step1: fill up your information**

You first need to fill in the following custom variables in `main.py` and provide the path to your custom data lake documents.
```python
#main.py
OPENAI_KEY = 'Your OPENAI KEY'
sql_query = "Your query"
sql_query_description = "Your query description"
file_lake_dir = "Your directory of the data lake"      
result_dir = "Your directory of the result "            
file_candidate_dir = "Your directory of the candidate files"  # Your file_candidate_dir directory must contain two subfolders: /candi and /key. These subfolders can be empty.
```
```python
#example filling
OPENAI_KEY = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxx'
sql_query = "SELECT name, age, team FROM NBA_palyer WHERE age < 40"
sql_query_description = 
"""
name : name of NBA player
age  : NBA player's age
team : NBA player's team
"""
file_lake_dir = "./datalake/"            # the directory of the data lake
result_dir = "./result/"                 # the directory of the result
file_candidate_dir = "./candidate_file/" # the directory of the candidate files
```

**Step2: run main.py**

```sh
python main.py
```

**Step3: result**
The `output.csv` file is located in your custom folder.

## <img src="imgs/join.png" alt="Description" width="42" height="42"> JOIN Operation

Before using these two programs, you need to first run `main.py` to obtain the candidate files for the required queries. Then, you can modify the code blocks in these two .ipynb files according to your needs to get the desired version.
For example:
```python
OPENAI_KEY = 'Your API key'
sql_query = "Your SQL query"
datalake_dir = "Your datalake dir"
result_dir = "Your result dir"  
##############################################
# use main.py to generate the candidate files#
##############################################
candidate_file_dir_A ="Your candidate file dir"
candidate_file_dir_B ="Your candidate file dir"
if not os.path.exists(datalake_dir):
    os.makedirs(datalake_dir)
if not os.path.exists(candidate_file_dir_A):
    os.makedirs(candidate_file_dir_A)
if not os.path.exists(candidate_file_dir_B):
    os.makedirs(candidate_file_dir_B)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
```

<span id="-datasets"></span>
##  <img src="imgs/dataset.png" alt="Description" width="42" height="42">  Datasets
### Wikiart
Wikiart includes 1,000 documents describing the biographies of
various artists. Each document contains 714 tokens on average, and
8 attributes.
### Financial
Financial includes 1,500 documents that provide details about var-
ious companies. On average, each document comprises 258 tokens
and 8 attributes.
### Sports
Sports consists of 100 Wikipedia pages about NBA players. We
transform each webpage into a document that includes only text.
Each document typically includes 1,645 tokens and provides infor-
mation on 10 attributes.
### Law
Law includes 3,000 case reports from the Federal Court of Australia
covering the years 2006 to 2009, from which we sample 600 doc-
uments. On average, each document contains 5,926 tokens and 9
attributes.


### Dataset URL
The dataset can be found at the following URL : 

```
https://drive.google.com/*****************************************
```
Markdown All in One

## <img src="imgs/result.png" alt="Description" width="42" height="42"> Result
Our experimental results are as follows:
<br>
<div align="center">
<img src="imgs/main_f1.png" width="1000px">
</div>
<br>
<br>
<div align="center">
<img src="imgs/main_cost.png" width="1000px">
</div>
<br>
