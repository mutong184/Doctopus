import json
from typing import List
import openai
import os
from textwrap import dedent
from pydantic import BaseModel,Field
from typing import List
import json


prompt_chunk_attribute = """
Content to analyze:
{file_content}

Extract the exact value for: {attributes}

Output REQUIREMENTS:
1. ONLY the raw value 
2. No explanations/formatting
3. Strictly in one line

Value:
"""


prompt_template_for_extract = """
The content of the file is as follows:
{file_content}

Please extract the following attributes from the above content:
{attributes_list}

Return the result in JSON format, for example: {{"Attribute1": "Value1", "Attribute2": "Value2", ...}}.
If any attribute cannot be found, set its value to "".
    """

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
    
def extract_attributes_from_text(text,prompt_template,attribute_explanations,model="gpt-4o"):
    # if not text:
    #     text = "this is nothing"
    raw_content = ""
    try:
        
        prompt = prompt_template.format(
            file_content=text,
            attributes="\n".join(attribute_explanations)
        )

        client = openai.OpenAI(base_url="https://api.gptsapi.net/v1",
        api_key="sk-Hwv0998366e81e4a9a4cd0cd7f38890f3f7417d34134xAbD"
        )

        # Call OpenAI GPT model
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant for extracting information from documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Ensure consistency
        )

        # Parse GPT response
        raw_content = response.choices[0].message.content.strip()
        extracted_values = clean_json_response(raw_content)
        return extracted_values

    except Exception as e:
        print(raw_content)
        return {"error": str(e)}





    




