import json
import sys
import openai
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Doctopus.var import nba_list,fin_list,wikiartList,legal_list
from Doctopus.Utils.prompt import prompt_template_for_validation
from Doctopus.Utils.Tool import load_dict_from_pickle, save_dict_to_pickle


def clean_json_response(content):
    """
    清理模型返回的 JSON 响应，去掉多余的标记。
    
    Args:
        content (str): 从模型返回的字符串内容。
    
    Returns:
        dict: 如果解析成功，返回解析后的 JSON；否则返回错误消息。
    """
    try:
        # 去掉可能的 ```json 和 ``` 标记
        if content.startswith("```json"):
            content = content[7:]  # 去掉 ```json
        if content.endswith("```"):
            content = content[:-3]  # 去掉 ```
        
        # 去掉两端的多余空白
        content = content.strip()

        # 解析为 JSON 对象
        return json.loads(content)
    except json.JSONDecodeError as e:
        # 如果解析失败，返回错误信息
        print(content)
        return {"error": f"Failed to parse JSON response: {str(e)}"}

def extract_attributes_from_folder(folder_path, attributes, prompt_template, model="gpt-4o"):
    results = {}
    total_token = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 仅处理文本文件
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                # 构造 GPT 提示词
                prompt = prompt_template.format(
                    text=file_content,
                    attributes=", ".join(attributes)
                )

                # 创建 OpenAI 客户端
                client = openai.OpenAI(
                    base_url="https://api.gptsapi.net/v1",
                    api_key="sk-Hwv0998366e81e4a9a4cd0cd7f38890f3f7417d34134xAbD"
                )

                # 调用 OpenAI 接口
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an assistant for extracting information from documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                # 解析响应结果
                raw_content = response.choices[0].message.content.strip().replace("Supporting_Fragments","Supporting Fragments")
                FIN_content = clean_json_response(raw_content)

                # 获取输入 token 数量
                input_tokens = response.usage.prompt_tokens
                total_token += input_tokens
                print(f"SUCCESS: Processed {filename}, Input Tokens: {input_tokens}")
                # print(FIN_content)
                results[file_path] = FIN_content

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results[filename] = {"error": str(e)}

    return results, total_token

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Extract attributes from text files in a folder using OpenAI API.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing text files")
    parser.add_argument('--attributes', type=str, default=None, help="Comma-separated list of attributes to extract (e.g., 'name,age,city')")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output Pickle file")
    parser.add_argument('--model', type=str, default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument('--dataset', type=str, default="nba")

    args = parser.parse_args()

    # 处理 attributes 参数
    attri_list_dataset = {
        "nba": nba_list,
        "fin": fin_list,
        "wikiart": wikiartList,
        "legal": legal_list
    }

    if args.dataset:
        attributes = attri_list_dataset[args.dataset]
    else:
        attributes = nba_list  # 默认使用 nba_list

    # 使用导入的 prompt_template_for_validation
    prompt_template = prompt_template_for_validation

    # 调用提取函数
    results, total_token = extract_attributes_from_folder(
        folder_path=args.folder_path,
        attributes=attributes,
        prompt_template=prompt_template,
        model=args.model
    )

    # 存储结果
    store_file_dic = {"validation": results, "total_token": total_token}
    save_dict_to_pickle(store_file_dic, args.output_file)

    # 打印存储的字典
    # print("Stored dictionary:", store_file_dic)

if __name__ == "__main__":
    main()



"""

# python extract_attributes_from_folder.py --folder_path "C:/Users/jjli74/Desktop/nba/nba_sample" --output_file "../storedata/NBA.pkl"

#nba
& C:/Users/jjli74/AppData/Local/Continuum/anaconda3/envs/text2table/python.exe c:/Users/jjli74/ljj/Doctopus-main/preprocess/get_valadation_ref.py --folder_path "C:/Users/jjli74/Desktop/nba/nba_sample" --output_file "./storedata/NBA.pkl"


#fin
& C:/Users/jjli74/AppData/Local/Continuum/anaconda3/envs/text2table/python.exe c:/Users/jjli74/ljj/Doctopus-main/preprocess/get_valadation_ref.py --folder_path "C:/Users/jjli74/Desktop/fin/fin_sample" --output_file "./storedata/fin.pkl" --dataset fin



"""