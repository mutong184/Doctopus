import json
import pickle
import time
import os
import argparse
from collections import defaultdict
from var import (
    create_dynamic_class,
    extract_attributes_from_text,
    get_01_dic,
    get_EM_dic,
    info_dataset,
    datasets_attributes,
    get_f1_dataset,
    save_dict_to_json,
    text_f1,
)
from utils import num_tokens_from_string
from var import legal_list, wikiartList, nba_list, fin_list, selected_attribute_prompt
from Utils.prompt import extract_attributes_from_chunk

# 读取策略文件
def load_strategy(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 解析策略并构造背包问题的 items
def prepare_knapsack_items(strategy_data_list, attribute_list, dataset_name, acc_cost_dic, allfiles, exist_abs, unexist_abs, null_rate, time_para_reduce):
    items = []
    for attr_idx, attribute in enumerate(attribute_list, start=1):  # 属性编号从1开始
        for file_idex, currentfile in enumerate(allfiles, start=1):
            for strategy_ele in strategy_data_list:
                if "rag.json" in strategy_ele:
                    strategy_data = load_strategy(strategy_ele)[dataset_name]
                    current_len = len(acc_cost_dic[dataset_name][currentfile][attribute])
                    for i in range(1, current_len + 1):
                        abstract_accuracy = strategy_data[f"{i}"][attribute]["f1"]
                        exist_accuract = acc_cost_dic[dataset_name][currentfile][attribute][:i]
                        try:
                            exist_accuract = max([float(acc) for acc, cost in exist_accuract])
                        except Exception as e:
                            print(e)
                            exist_accuract = 0

                        predict_acc = exist_accuract * exist_abs[attribute] * (1 - null_rate[dataset_name][attribute]) + (1 - exist_accuract) * unexist_abs[attribute] * null_rate[dataset_name][attribute]
                        cost = acc_cost_dic[dataset_name][currentfile][attribute][i - 1][1] // time_para_reduce
                        items.append((attr_idx * 100 + file_idex * 100000, cost, predict_acc))  # (组号, cost, 价值)
                    continue

                strategy_data = load_strategy(strategy_ele)[dataset_name]
                if attribute not in strategy_data:
                    print(f"{strategy_ele} has no {attribute}")
                    continue

                strategy = strategy_data[attribute]  # 该属性的12种策略
                if isinstance(strategy, float):
                    accuracy = strategy
                else:
                    accuracy = strategy["f1"]

                cost = 0
                items.append((attr_idx * 100 + file_idex * 100000, cost, accuracy))  # (组号, cost, 价值)

    return items

# 分组背包算法
def group_knap_sack(n, m, items):
    # 按组号排序
    items.sort(key=lambda x: x[0])

    # 重新组织数据，将同一组的物品放在一起
    groups = []
    current_group = None
    for item in items:
        g, c, w = item
        if g != current_group:
            groups.append([])
            current_group = g
        groups[-1].append((c, w, g))  # (cost, value, group)

    t = len(groups)  # 组的数量

    # 初始化 dp 数组
    dp = [[-float('inf')] * (m + 1) for _ in range(t + 1)]
    dp[0][0] = 0  # 初始状态，没有选择任何组时容量0价值0

    # 记录选择的物品和策略索引
    choice = [[[] for _ in range(m + 1)] for _ in range(t + 1)]
    strategy_choice = [[[] for _ in range(m + 1)] for _ in range(t + 1)]

    for k in range(1, t + 1):
        group = groups[k - 1]  # 当前组的物品列表
        for j in range(m + 1):
            max_val = -float('inf')
            best_prev_j = -1
            best_item = None
            best_strategy = None
            # 遍历当前组的所有物品，寻找最佳选择
            for idx, (c, w, g) in enumerate(group):
                if j >= c:
                    prev_j = j - c
                    if dp[k - 1][prev_j] + w > max_val:
                        max_val = dp[k - 1][prev_j] + w
                        best_prev_j = prev_j
                        best_item = (g, c, w)
                        best_strategy = (g, idx)
            # 更新dp和选择
            if max_val != -float('inf'):
                dp[k][j] = max_val
                choice[k][j] = choice[k - 1][best_prev_j].copy()
                choice[k][j].append(best_item)
                strategy_choice[k][j] = strategy_choice[k - 1][best_prev_j].copy()
                strategy_choice[k][j].append(best_strategy)
            else:
                dp[k][j] = -float('inf')  # 无可行选择

    # 获取最大价值
    max_value = max(dp[t])
    if max_value == -float('inf'):
        return 0, [], []  # 无解情况处理

    max_index = dp[t].index(max_value)
    selected_items = choice[t][max_index]
    selected_strategies = strategy_choice[t][max_index]

    return max_value, selected_items, selected_strategies

# 主函数
def main(dataset_name, model_index, time_para_reduce,ref_id,no_strategy,LLM_evaluate,metric):
    if LLM_evaluate:
        strategy_file_list = [
        "/home/lijiajun/RAG/strategy/openie6.json",
        "/home/lijiajun/RAG/strategy/debertv3.json",
        "/home/lijiajun/RAG/strategy/eva.json",
        "/home/lijiajun/RAG/strategy/rag.json",
    ]  # 假设策略存储在 JSON 文件中

    else:
        strategy_file_list = [
        "/home/lijiajun/RAG/stratery_llm/openie6.json",
         "/home/lijiajun/RAG/stratery_llm/debertv3.json",
        "/home/lijiajun/RAG/stratery_llm/eva.json",
        "/home/lijiajun/RAG/strategy/rag.json",
    ]  # 假设策略存储在 JSON 文件中
    #textf1,01match,entitymatch
    if metric == "01match":
         strategy_file_list = [
        "/home/lijiajun/RAG/stratery_01match/debertv3_01match.json",
         "/home/lijiajun/RAG/stratery_01match/eva_01match.json",
        "/home/lijiajun/RAG/stratery_01match/openie6_01match.json",
        "/home/lijiajun/RAG/strategy/rag.json",]  
    elif metric == "textf1":
        strategy_file_list = [
        "/home/lijiajun/RAG/strategy/openie6.json",
        "/home/lijiajun/RAG/strategy/debertv3.json",
        "/home/lijiajun/RAG/strategy/eva.json",
        "/home/lijiajun/RAG/strategy/rag.json",
    ]  # 假设策略存储在 JSON 文件中
    elif metric == "entitymatch":
        strategy_file_list = [
        "/home/lijiajun/RAG/strategy_entitymatch/debertv3_entitymatch.json",
        "/home/lijiajun/RAG/strategy_entitymatch/eva_entitymatch.json",
        "/home/lijiajun/RAG/strategy_entitymatch/openie6_entitymatch.json",
        "/home/lijiajun/RAG/strategy/rag.json",
    ]  # 假设策略存储在 JSON 文件中
        
        
    
    strategy_to_remove = no_strategy + ".json"

    # 使用列表推导式过滤掉包含 "eva" 的策略
    strategy_file_list = [strategy for strategy in strategy_file_list if strategy_to_remove not in strategy]
    
    print(f"strategy: {strategy_file_list}")

    

    null_rate = load_strategy("null_rate.json")  # 包含四个数据集的空值统计
    len_wikiart = 100
    model_choices = ["gpt-3.5-turbo-0125", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
    datapath = "datadir"
    
    
    if ref_id == 0:
        with open('/home/lijiajun/RAG/acc_cost_chunks/acc_cost_dic_no_val_llm.pkl', 'rb') as f:
            acc_cost_dic = pickle.load(f)

        with open('/home/lijiajun/RAG/acc_cost_chunks/main_chunks_no_val_llm.pkl', 'rb') as f:
            chunks_dataset = pickle.load(f)
    elif ref_id == 1:
        with open('/home/lijiajun/RAG/acc_cost_chunks/acc_cost_dic_llm.pkl', 'rb') as f:
            acc_cost_dic = pickle.load(f)

        with open('/home/lijiajun/RAG/acc_cost_chunks/main_chunks_llm.pkl', 'rb') as f:
            chunks_dataset = pickle.load(f)
    elif ref_id == 2:       
        with open('/home/lijiajun/RAG/acc_cost_chunks/acc_cost_dic_val.pkl', 'rb') as f:
                acc_cost_dic = pickle.load(f)

        with open('/home/lijiajun/RAG/acc_cost_chunks/main_chunks_val.pkl', 'rb') as f:
            chunks_dataset = pickle.load(f)
    elif ref_id == 3:
        with open('/home/lijiajun/RAG/acc_cost_chunks/acc_cost_dic_val_llm.pkl', 'rb') as f:
                acc_cost_dic = pickle.load(f)

        with open('/home/lijiajun/RAG/acc_cost_chunks/main_chunks_val_llm.pkl', 'rb') as f:
            chunks_dataset = pickle.load(f)
    else:
        print("ref id occur error!")
        
    exist_pro_file = f"/home/lijiajun/RAG/exist_un_pro/{dataset_name}/pro_exist.json"
    unexist_pro_file = f"/home/lijiajun/RAG/exist_un_pro/{dataset_name}/pro_unexist.json"

    total_len = info_dataset[dataset_name]["total_len"]
    length = info_dataset[dataset_name]["length"]
    given_cost_raw = info_dataset[dataset_name]["given_cost_raw"]
    given_cost_list = [int(1000000 * ele) for ele in given_cost_raw]

    f1 = []
    cost_for_dataset = []
    cost_for_real = []

    for idx, given_cost in enumerate(given_cost_list):
        if given_cost < length * total_len * 0.2:
            print(f"token not enough, must {length * total_len * 0.2} for valuation")
            continue
        cost_for_dataset.append(given_cost_raw[idx])
        max_cost = int((given_cost - length * total_len * 0.2) // total_len * len_wikiart // time_para_reduce)

        ## allfile
        allfiles = os.listdir(info_dataset[dataset_name][datapath])
        attribute_list = datasets_attributes[dataset_name]

        # 解析策略并转换成 (组号, cost, accuracy)
        exist_abs = load_strategy(exist_pro_file)
        unexist_abs = load_strategy(unexist_pro_file)
        items = prepare_knapsack_items(strategy_file_list, attribute_list, dataset_name, acc_cost_dic, allfiles, exist_abs, unexist_abs, null_rate, time_para_reduce)
        print("total cost", max_cost)
        print("group number,", len(items))
        stime = time.time()
        max_value, selected_items, selected_strategies = group_knap_sack(len(items), max_cost, items)
        etime = time.time()
        print("time consume:", etime - stime)

        predict_ours = defaultdict(lambda: defaultdict(list))
        for selected_strategy in selected_strategies:
            filename_attri, stratege_index = selected_strategy
            filename_index = int(filename_attri // 100000 - 1)
            attri_index = int((filename_attri % 100000) // 100 - 1)
            filename = allfiles[filename_index]
            attribute = attribute_list[attri_index]
            if stratege_index < len(strategy_file_list) - 1:
                pass
            else:
                strategey = "rag"
                topk_index = stratege_index - len(strategy_file_list) + 1 + 1
                topk_chunks = chunks_dataset[dataset_name][filename][attribute][:topk_index]
                predict_ours[filename][attribute] = topk_chunks

        # 创建一个新的字典来存储每个文件的总 topk_chunks
        merged_predict_ours = {}
        for filename, attributes in predict_ours.items():
            merged_chunks = set()
            for attribute, chunks in attributes.items():
                merged_chunks.update(chunks)  # 将当前属性的 chunks 添加到集合中
            merged_predict_ours[filename] = list(merged_chunks)

        total_cost = 0
        final_rag_res = defaultdict()
        for current_filename, attribute_list in predict_ours.items():
            explain_list = info_dataset[dataset_name]["array"]
            attribute_explanations = {
                attribute: "".join([item for item in explain_list if item.startswith(attribute)])
                for attribute in attribute_list.keys()
            }
            selected_attributes = list(attribute_explanations.keys())
            selected_explain_attributes = list(attribute_explanations.values())
            selected_class = create_dynamic_class(selected_attributes)
            selected_prompt = selected_attribute_prompt.format(attribute_explain="\n".join(selected_explain_attributes))

            total_cost += num_tokens_from_string("".join(merged_predict_ours[current_filename]), "cl100k_base")
            attribute_value_dic = extract_attributes_from_text("".join(merged_predict_ours[current_filename]), selected_class, selected_prompt, model_choices[model_index])
            final_rag_res[current_filename] = attribute_value_dic

        allfiles = os.listdir(info_dataset[dataset_name][datapath])
        attribute_list = datasets_attributes[dataset_name]
        predict_oursways = defaultdict(lambda: defaultdict())
        for selected_strategy in selected_strategies:
            filename_attri, stratege_index = selected_strategy
            filename_index = int(filename_attri // 100000 - 1)
            attri_index = int((filename_attri % 100000) // 100 - 1)
            filename = allfiles[filename_index]
            attribute = attribute_list[attri_index]
            if stratege_index < len(strategy_file_list) - 1:
                strategey = os.path.basename(strategy_file_list[stratege_index]).replace(".json", "").strip()
                try:
                    predict_oursways[filename][attribute] = load_strategy(info_dataset[dataset_name][strategey])[filename][attribute]
                except Exception as e:
                    print(e)
                    print(f"strategy occur problem {strategey}, filename: {filename}, attribute {attribute}")
                    predict_oursways[filename][attribute] = ""
            else:
                strategey = "rag"
                if isinstance(final_rag_res[filename][attribute], list):
                    value = " ".join(final_rag_res[filename][attribute])
                else:
                    value = final_rag_res[filename][attribute]
                predict_oursways[filename][attribute] = value
        print("every cost", total_cost)
        
        if metric =="textf1":
            f1_given_cost = get_f1_dataset(load_strategy(info_dataset[dataset_name]["groundtruth"]), predict_oursways)
        elif metric == "01match":
            f1_given_cost = get_01_dic(load_strategy(info_dataset[dataset_name]["groundtruth"]), predict_oursways)
        elif metric == "entitymatch":
            f1_given_cost = get_EM_dic(load_strategy(info_dataset[dataset_name]["groundtruth"]), predict_oursways)
            
            
        f1.append(f1_given_cost)
        print("f1_for_one,",f1_given_cost)
        cost_for_one = total_cost * total_len / len_wikiart / 1000000 + length * total_len * 0.2 / 1000000
        cost_for_real.append(cost_for_one)
        print("cost_for_one:",cost_for_one)
    print(dataset_name)
    print("f1 list :", f1)
    print("given cost:", cost_for_dataset)
    print("cost_for_real:", cost_for_real)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG strategy optimization.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., wikiart).")
    parser.add_argument("--model_index", type=int, default=1, help="Index of the model to use (default: 1).")
    parser.add_argument("--time_para_reduce", type=int, default=40, help="Time parameter reduction factor (default: 40).")
    parser.add_argument("--reference", type=int, default=3, help="0:no_red,1.LLM_ref 2.val_ref,3.val_LLm")
    parser.add_argument("--no_strategy", type=str, default="no", help="eva,openie6,debertv3")
    parser.add_argument("--LLM_evaluate", type=str, default=1, help="1.evaluate with LLM 0.evaluate with Human")
    parser.add_argument("--metric", type=str, default="textf1", help="textf1,01match,entitymatch")
    args = parser.parse_args()

    main(args.dataset, args.model_index, args.time_para_reduce,args.reference,args.no_strategy,args.LLM_evaluate,args.metric)
    
    
""""

nohup python -u main.py --dataset wikiart --model_index 1 --time_para_reduce 40  --reference 0  > wikiart_model_1_time_reduce40_ref0_nostra.log 2>&1 &

## 执行openie6 缺的结果

nohup python -u main.py --dataset wikiart --model_index 1 --time_para_reduce 40  --reference 3  --no_strategy openie6 > wikiart_model_1_time_reduce40_ref3_noOpenie6.log 2>&1 &

nohup python -u main.py --dataset nba --model_index 1 --time_para_reduce 40  --reference 0   > nba_model_1_time_reduce40_ref0.log 2>&1 &


# 执行 大模型评估的结果

nohup python -u main.py --dataset wikiart --model_index 1 --time_para_reduce 40  --reference 3 --LLM_evaluate 0 > wikiart_model_1_time_reduce40_ref3_LLM_evaluate0.log 2>&1 &


nohup python -u main.py --dataset nba --model_index 1 --time_para_reduce 40  --reference 3 --LLM_evaluate 0 > nba_model_1_time_reduce40_ref3_LLM_evaluate0.log 2>&1 &

nohup python -u main.py --dataset nba --model_index 1 --time_para_reduce 40  --reference 0  > nba_model_1_time_reduce40_ref3_LLM_evaluate0.log 2>&1 &

nohup python -u main.py --dataset nba --model_index 1 --time_para_reduce 40  --reference 2  > nba_model_1_time_reduce40_ref2_LLM_evaluate0.log 2>&1 &

nohup python -u main.py --dataset nba --model_index 1 --time_para_reduce 40  --reference 3 --LLM_evaluate 1 > nba_model_1_time_reduce40_ref3_LLM_evaluate1.log 2>&1 &

"""


""" 

nohup python -u main.py --dataset fin --model_index 2 --time_para_reduce 40  --reference 3  --no_strategy openie6 > wikiart_model_2_time_reduce40_ref3_noOpenie6.log 2>&1 &


nohup python -u main.py --dataset fin --model_index 2 --time_para_reduce 40  --reference 3  --no_strategy eva > wikiart_model_2_time_reduce40_ref3_noeva.log 2>&1 &



nohup python -u main.py --dataset fin --model_index 2 --time_para_reduce 40  --reference 3  --no_strategy eva > wikiart_model_2_time_reduce40_ref3_noclose.log 2>&1 &


nohup python -u main.py --dataset fin --model_index 2 --time_para_reduce 40  --reference 3  --no_strategy eva > wikiart_model_2_time_reduce40_ref3_noclose.log 2>&1 &


nohup python -u main.py --dataset fin --model_index 2 --time_para_reduce 40  --reference 1   > fin_model_2_time_reduce40_ref1.log 2>&1 &

nohup python -u main.py --dataset fin --model_index 2 --time_para_reduce 40  --reference 0   > fin_model_2_time_reduce40_ref0.log 2>&1 &

nohup python -u main.py --dataset fin --model_index 2 --time_para_reduce 40  --reference 2   > fin_model_2_time_reduce40_ref2.log 2>&1 &

nohup python -u main.py --dataset nba --model_index 2 --time_para_reduce 40  --reference 2  > nba_model_2_time_reduce40_ref2_LLM_evaluate0.log 2>&1 &


# 评估不通的指标

nohup python -u main.py --dataset wikiart --model_index 2 --time_para_reduce 40 --metric 01match > wikiart_model_2_time_reduce40_metric_01match.log 2>&1 &


nohup python -u main.py --dataset nba --model_index 2 --time_para_reduce 40 --metric 01match > nba_model_2_time_reduce40_metric_01match.log 2>&1 &

nohup python -u main.py --dataset fin --model_index 2 --time_para_reduce 40 --metric 01match > fin_model_2_time_reduce40_metric_01match.log 2>&1 &

nohup python -u main.py --dataset legal --model_index 2 --time_para_reduce 40 --metric 01match > legal_model_2_time_reduce40_metric_01match.log 2>&1 &

nohup python -u main.py --dataset wikiart --model_index 2 --time_para_reduce 40 --metric entitymatch > wikiart_model_2_time_reduce40_metric_entitymatch.log 2>&1 &

nohup python -u main.py --dataset nba --model_index 2 --time_para_reduce 40 --metric textf1 > nba_model_2_time_reduce40_metric.log 2>&1 &




"""