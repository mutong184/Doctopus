<div align= "center">
    <h1>Doctopus: Budget-aware Structural Table Extraction from Unstructured Documents</h1>
</div>
<p align="center">
  <a href="#-struct">Folder Structure</a> •
  <a href="#-getstart">Getting Start</a> •
  <a href="#-datasets">Datasets</a> •
</p>


<br>
<div align="center">
<img src="imgs/framework.pdf" width="1000px">
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

## <img src="imgs/folder.png"  width="42" height="42"> Folder Structure


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

##  <img src="imgs/start.png" width="42" height="42"> Getting Started

This is an example of how to set up QUEST locally. To get a local copy up, running follow these simple example steps.

### Prerequisites

To install the required packages, you can create a conda environment:

```sh
conda create --name QUEST python=3.9
```

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
<img src="imgs/e5.png" width="500px" > 

```sh
python demo.py
```

### <img src="imgs/run.png"  width="42" height="42"> Run QUEST

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
##  <img src="imgs/datasets.png" alt="Description" width="42" height="42">  Datasets
### WikiText
We crawl 800 Wikipedia pages across 10 domains, such as directors, cities, NBA players, and companies, etc. The average tokens per document is 1,264.
### SWDE
SWDE is a broadly used dataset in the field of information extraction, including two major topics: Movies and Universities. It consists of 2,000 web pages, in average each web page containing 416 tokens per page. Despite the relatively short length of the documents, SWDE contains 16 attribute.
### LCR
LCR includes 3,000 case reports from the Federal Court of Australia from 2006 to 2009. Each documents contains 6,247 tokens in average. The documents have rich information such as the court where the case was heard, the judge, the judgment outcome, judicial interpretations, and relevant statutes, etc.


### Dataset URL
The dataset can be found at the following URL : 

```
https://drive.google.com/file/d/1dk6phBPbUcTPdpE0iNy7ORa01sNTsBGK/view?usp=sharing
```


## <img src="imgs/query.png" alt="Description" width="42" height="42"> Queries

### Single-table queries
We create queries for single tables and join tables. The single-table queries cover both range and equality filters. We first construct the filters in WHERE clause: 
(1) For each query, we randomly sample a certain number of attributes from the attribute set in the query to construct the filters; 
(2) For numerical attributes, we randomly create different types of filters, including =, < and >, while for categorical attributes, we only generate equality filters; 
(3) We then use these single filters to construct conjunctions, disjunctions, or the hybrid of both conjunction and disjunction. Each of these three categories has roughly the same number of queries. 
Next, we randomly sample a certain number of attributes to form the SELECT clause. Finally, we ask the graduate students to validate all queries and eliminate the unreasonable ones.

C1: The WHERE clause contains only one filter.
C2: The WHERE clause contains 2-3 filters.
C3: The WHERE clause contains 4 or more filters.

Some of the queries are as follows:
```SQL
C1
SELECT name, age, team FROM NBA_player WHERE age < 40
SELECT name, age, team FROM NBA_player WHERE team = 'Los Angeles Lakers'
SELECT name, age, team FROM NBA_player WHERE NBA_championships > 2
SELECT name, age, team FROM NBA_player WHERE nationality = USA
SELECT name, position, draft_year FROM NBA_player WHERE draft_pick <= 10
SELECT university_name,rank FROM University WHERE rank <= 3
SELECT university_name,rank FROM University WHERE year_founded < 1850
SELECT university_name,rank FROM University WHERE total_undergraduate_enrollment < 1890
SELECT university_name, setting, rank FROM University WHERE setting = 'rural'
SELECT university_name, academic_calendar, rank FROM University WHERE academic_calendar = semester
SELECT court_name, hearing_year, legal_basis FROM LCR WHERE hearing_year = 2009
SELECT court_name, judge_name, judgment_year FROM LCR WHERE judge_name = 'Justice Spender'
SELECT court_name, charges, judgment_year FROM LCR WHERE judgment_year = 2008
SELECT court_name, hearing_year, verdict FROM LCR WHERE verdict = 'dismissed'
SELECT court_name, hearing_year, charges FROM LCR WHERE charges = 'obtaining property by false pretences (8 counts)'
...
C2
SELECT name, age, team FROM NBA_player WHERE nationality = USA AND age < 40
SELECT name, age, mvp_awards FROM NBA_player WHERE mvp_awards >= 1 AND (age > 30 OR team = New York Knicks)
SELECT name, draft_pick, draft_year FROM NBA_player WHERE draft_year >= 2000 AND (draft_pick <= 15 OR team = Philadelphia 76ers)
SELECT name, team, NBA_championships FROM NBA_player WHERE NBA_championships >= 3 OR team = 'Los Angeles Lakers' OR team = 'Houston Rockets'
SELECT name, draft_pick, draft_year,NBA_championships FROM NBA_player WHERE  (draft_year >= 2010 OR draft_pick <= 15) AND NBA_championships > 0
SELECT university_name, year_founded, rank FROM University WHERE year_founded < 1900 AND rank < 20 AND rank > 10
SELECT university_name, setting, total_undergraduate_enrollment FROM University WHERE setting = urban AND total_undergraduate_enrollment < 2000
SELECT university_name, year_founded, setting FROM University WHERE year_founded >= 1900 AND setting = suburban
SELECT university_name, rank FROM University WHERE rank <= 10 AND rank >= 5
SELECT university_name, setting, year_founded FROM University WHERE setting = 'rural' AND year_founded >= 1950
SELECT court_name, hearing_year, legal_basis FROM LCR WHERE hearing_year = 2009 AND legal_basis = 'Extradition Act 1988 (Cth)'
SELECT court_name, judge_name, judgment_year FROM LCR WHERE judge_name = 'Magistrate GN Calder' AND judgment_year = 2009
SELECT court_name, charges, legal_basis FROM LCR WHERE hearing_year = 2008 AND charges = 'Contraventions of s 45 of the Competition Code'
SELECT court_name, hearing_year, judgment_year FROM LCR WHERE hearing_year = 2007 AND judgment_year = 2007
SELECT court_name, hearing_year, judgment_year FROM LCR WHERE hearing_year = 2009 AND judgment_year >= 2009
...
C3
SELECT name, position, draft_year FROM NBA_player WHERE draft_pick <= 10 AND draft_year > 2000 AND nationality = USA AND age > 20
SELECT name, age, college FROM NBA_player WHERE age > 20 AND age < 40 AND (team  = 'Los Angeles Lakers' OR college = 'University of Nebraska')
SELECT name, team, position FROM NBA_player WHERE (team = 'Los Angeles Lakers' AND position = 'forward') OR (position = 'guard' AND age < 50)
SELECT name, draft_pick, draft_year FROM NBA_player WHERE draft_year >= 2000 AND draft_year <= 2010 AND draft_pick <= 15 AND draft_pick >= 5
SELECT name,age, draft_pick,NBA_championships FROM NBA_player WHERE age < 40 AND draft_pick <= 15 AND NBA_championships > 2 AND age > 20
SELECT university_name, academic_calendar, campus_size FROM University WHERE academic_calendar = '4-1-4-based' AND (year_founded < 1900 OR total_undergraduate_enrollment < 2000)
SELECT university_name, academic_calendar, rank FROM University WHERE academic_calendar = semester AND (year_founded < 1900 OR total_undergraduate_enrollment < 2000) AND rank < 100
SELECT university_name, academic_calendar, rank FROM University WHERE academic_calendar = 4-1-4-based AND year_founded < 1900 OR total_undergraduate_enrollment < 1500 AND rank < 20
SELECT university_name, year_founded, rank FROM University WHERE year_founded > 1900 AND year_founded < 2000 AND rank < 100 AND rank > 10
SELECT university_name, year_founded, rank FROM University WHERE setting = urban AND year_founded < 2000 AND rank < 100 AND rank > 10
SELECT court_name, judge_name, hearing_year, judgment_year FROM LCR WHERE hearing_year = 2007 OR hearing_year = 2008 OR judgment_year = 2008 OR judgment_year = 2007
SELECT court_name, judge_name, charges, legal_basis FROM LCR WHERE hearing_year = 2009 OR charges = 'obtaining property by false pretences' OR judge_name = 'Justice Tamberlin' OR legal_basis = 'Federal Court of Australia Act 1976 (Cth)'
SELECT court_name, hearing_year, judgment_year, legal_basis FROM LCR WHERE hearing_year = 2008 AND judgment_year = 2009 AND legal_basis = 'Competition Code of New South Wales'
SELECT court_name, judge_name, charges, verdict FROM LCR WHERE hearing_year >= 2007 AND charges = 'Contraventions of s 45' AND verdict = 'dismissed' OR judge_name = 'Justice Spender'
SELECT court_name, hearing_year, judgment_year, legal_basis FROM LCR WHERE hearing_year = 2009 OR judgment_year >= 2009 OR legal_basis = 'Australian Patent 774224 A barrier'
...
```
We have provided some additional queries for reference. The files are located in the directory ./query/.


### Join operation queries
For our join experiment, we selected four domains from WikiText: nba\_player, nba\_team, city, and owner. The Player and Team tables join on the team\_name attribute. The Team and City tables join on the location attribute. The Team and Owner tables join on the owner\_name attribute. The construction of filters in the where clause can be directly transferred from the filter construction in previous single-table queries of WikiText.

We constructed a total of 90 queries for two-table joins and multi-table joins. We will showcase a portion of them next, as the rest follow the same construction as the ones presented.

We first present a complete SQL query along with its description as follows:

```SQL
SELECT NBA_players.name, NBA_players.team , NBA_teams.name , NBA_teams.founded_year 
FROM NBA_players,NBA_teams
WHERE NBA_players.age > 30 AND NBA_teams.founded_year > 1940 AND NBA_players.team = "Los Angeles Lakers" AND NBA_players.team = NBA_teams.name
<Description> :
NBA_players            : A table providing basic information about NBA players.
NBA_teams              : A table providing basic information about NBA teams.
NBA_players.name       : The full name of the player.
NBA_players.age        : The age of the player.
NBA_players.team       : Current team of the player.
NBA_teams.name         : The full name of the team.
NBA_teams.founded_year : The year in which the team was established.
```

```SQL
# Example 1
SELECT NBA_player.name, NBA_player.team , NBA_team.name , NBA_team.founded_year 
FROM NBA_player,NBA_team
WHERE NBA_player.age <= 40 AND NBA_team.founded_year > 1980 AND NBA_player.draft_pick < 10 AND NBA_player.team = NBA_team.name
# Example 2
SELECT NBA_player.name, NBA_player.team , NBA_team.name , NBA_team.founded_year 
FROM NBA_player,NBA_team
WHERE NBA_player.age <= 80 AND NBA_player.draft_pick > 10 AND NBA_team.location = "Los Angeles" AND NBA_player.team = NBA_team.name
# Example 3
SELECT 
    NBA_player.name, 
    NBA_player.team,
    NBA_team.founded_year,
    City.name,
    Owner.name
FROM
    NBA_player,NBA_team,City,Owner
WHERE 
    NBA_player.age <= 40 
    AND NBA_player.draft_pick < 2 
    AND NBA_team.location = "Los Angeles"
    AND City.population > 1000000
    AND Owner.own_year > 2000
    AND NBA_player.team = NBA_team.name
    AND NBA_team.location = City.name
    AND NBA_team.boss_name = Owner.name
```


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
