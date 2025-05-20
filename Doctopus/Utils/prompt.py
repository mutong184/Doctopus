import json
import openai
import os

# from utils import num_tokens_from_string





prompt_template_for_validation = """

You are an assistant proficient in text comprehension, tasked with extracting specified attribute values from the following text and indicating the supporting fragments for these values.
Task Description:
– Extract the value of each attribute from the text.
– For each attribute, return a dictionary containing:
 – "Value": The corresponding value of the attribute found in the text.
 – "Supporting Fragments": The text fragments that explicitly mention this value.
  there will be multiple fragments, return them as a list. Be sure to include all relevant fragments, and do not include any invalid characters or formats

Requirements:
– If an attribute is not explicitly mentioned in the text, return:
 – "Value": ""
 – "Supporting Fragments": [""]
–  an attribute value May corresponds to multiple fragments, so return them as a list.
– Do not add any explanations or comments, such as //, after "Supporting Fragments": [""]
Here is the input text and attribute list:
Text: {text}
Attribute List: {attributes}

Please extract the attribute values and return the results in JSON format.
"""


prompt_template_for_extract = """
The content of the file is as follows:
{file_content}

Please extract the following attributes from the above content:
{attributes_list}

Return the result in JSON format, for example: {{"Attribute1": "Value1", "Attribute2": "Value2", ...}}.
If any attribute cannot be found, set its value to "".
    """


prompt_template_for_ref_K = """
For a given attribute: {attribute}, do the following:

1. Generate {h} potential values for the attribute {attribute}.
   - For example, if the attribute is "name", generate {h} possible names like: John, Mike, etc.

2. For each potential attribute value generated in step 1, create {h} natural sentences (references) that contain the attribute value.
   - The references should mimic realistic usage of the attribute value by humans.
   - For instance, if the attribute value is "John", generate sentences like: "John is 11 years old", "My friend John likes to play basketball", etc.

3. Output the results in a JSON format where:
   - The keys are the generated attribute values
   - The corresponding values are lists containing the {h} generated references for each attribute value

Here's an example of the desired JSON output format for the "name" attribute:

{
   "John": [
      "John is 11 years old",
      "My friend John likes to play basketball",
      ...
   ],
   "Mike": [
      "I met Mike at the party last weekend",
      "Mike works as a software engineer at Google",
      ...
   ],
   ...
}

"""


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
        return {"error": f"Failed to parse JSON response: {str(e)}"}


def extract_attributes_from_files(file_list, attributes, prompt_template=prompt_template_for_extract, model="gpt-4o"):
    """
    Extract specified attributes from the content of given files.
    
    Args:
        file_list (list): List of file paths.
        attributes (list): List of attributes to extract.
        prompt_template (str): Template for constructing the GPT prompt.
        model (str): The GPT model to use, default is "gpt-4".

    Returns:
        dict: A dictionary where keys are filenames, and values are the extracted attributes and their values.
    """
    results = {}

    for file_path in file_list:
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Use template to construct the GPT prompt
            prompt = prompt_template.format(
                file_content=file_content,
                attributes_list=", ".join(attributes)
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
            results[file_path] = extracted_values

        except Exception as e:
            # Handle exceptions and log them
            results[os.path.basename(file_path)] = {"error": str(e)}

    return results


def extract_attributes_from_chunk(chunk, attribute, prompt_template=prompt_chunk_attribute, model="gpt-3.5-turbo"):
    """
    Extract specified attributes from the content of given files.
    
    Args:
        file_list (list): List of file paths.
        attributes (list): List of attributes to extract.
        prompt_template (str): Template for constructing the GPT prompt.
        model (str): The GPT model to use, default is "gpt-4".

    Returns:
        dict: A dictionary where keys are filenames, and values are the extracted attributes and their values.
    """

    
    try:
        prompt = prompt_template.format(
            file_content=chunk,
            attributes=attribute
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
        content = raw_content.strip('"')
        return content
    except Exception as e:
        # Handle exceptions and log them
        print("error",e)
        # print(raw_content)
        # return "",0,0
        return e

    
def extract_attributes_from_text(text,prompt_template,attribute_explanations,model="gpt-4o"):
    # if not text:
    try:
        
        prompt = prompt_template.format(
            file_content=text,
            attributes_list="\n".join(attribute_explanations)
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
        input_tokens = response.usage.prompt_tokens
        raw_content = response.choices[0].message.content.strip()
        extracted_values = clean_json_response(raw_content)
        return extracted_values,input_tokens

    except Exception as e:
        print(raw_content)
        return {"error": str(e)},0



