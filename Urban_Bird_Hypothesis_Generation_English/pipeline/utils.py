
import os,re
import json
import datetime
from typing import List,Dict,Any,Union,Tuple
import logging
import agentscope
from agentscope.agents import DialogAgent
from agentscope.message import Msg
import random,ast
import requests ,time
from sentence_transformers import SentenceTransformer, util
from agentscope.models import OpenAIChatWrapper
import functools
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np

# JSON Document Loading Function
def load_and_preprocess_json(filepath: str, include_conclusion: bool, include_intro: bool, logger: logging.Logger) -> \
tuple[List[str], Dict[str, str]]:
    if not os.path.exists(filepath):
        logger.error(f"Error: JSON file '{filepath}' not found.")
        return [], {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        all_document_texts = []
        document_map = {}

        for article in articles:
            doc_id = article.get('ID')
            if doc_id is None:
                logger.warning("Article without an ID found, skipping.")
                continue

            title = article.get('title', 'No Title')
            text_parts = [
                f"ID:{doc_id}",
                f"Title: {title}",
                f"Abstract: {article.get('abstract', '')}",
            ]
            if include_intro and article.get('Introduction'):
                text_parts.append(f"Introduction: {article.get('Introduction')}")
            if include_conclusion and article.get('conclusion'):
                text_parts.append(f"Conclusion: {article.get('conclusion')}")

            doc_text = "\n\n".join(text_parts)
            all_document_texts.append(doc_text)
            document_map[str(doc_id)] = doc_text

        logger.info(f"Successfully loaded {len(all_document_texts)} documents from '{filepath}'.")
        return all_document_texts, document_map

    except json.JSONDecodeError:
        logger.error(f"Error: The file '{filepath}' is not a valid JSON format.")
        return [], {}
    except Exception as e:
        logger.error(f"An unknown error occurred while reading or processing the JSON file: {e}")
        return [], {}

# JSON Knowledge Loading Function (For Grand Expert)
def load_CLL_json(filepath: str, logger: logging.Logger,) -> \
List[str]:
    if not os.path.exists(filepath):
        logger.error(f"Error: JSON file '{filepath}' not found.")
        return [], {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        all_document_texts = []

        for article in articles:
            doc_id = article.get('ID')
            if doc_id is None:
                logger.warning("Article without an ID found, skipping.")
                continue

            title = article.get('title', 'No Title')

            text_parts = [
                f"ID:{doc_id}",
                f"Title: {title}",
                f"Abstract: {article.get('abstract', 'N/A')}",
            ]

            doc_text = "\n\n".join(text_parts)
            all_document_texts.append(doc_text)

        logger.info(f"Successfully loaded the knowledge for GrandExpert from '{filepath}'.")
        return all_document_texts

    except json.JSONDecodeError:
        logger.error(f"Error: The file '{filepath}' is not a valid JSON format.")
        return []
    except Exception as e:
        logger.error(f"An unknown error occurred while reading or processing the JSON file: {e}")
        return []

def make_api_call_with_retry(
        agent_making_call: DialogAgent,
        message: Msg,
        all_agents_to_update: list,
        logger: logging.Logger,
        model_config: dict,
) -> Msg:
    """
    Encapsulates API calls.
    When a credit limit issue occurs, all agents are updated in-place by dynamically recreating model components to continue running.
    """

    while True:
        try:
            response = agent_making_call(message)
            return response
        except Exception as e:
            error_str = str(e).lower()

            logger.error(f"The API call failed, possibly because the credit limit is exhausted. Error:{e}")
            logger.info(f"Current API Key: ...{model_config['api_key'][-4:]}")

            new_key = model_config['api_key']
            #new_key = input("An API quota issue has been detected. Please enter a new API Key to continue, or press Enter to exit the program:").strip()

            if not new_key:
                logger.error("The new API Key is not provided, the program will terminate.")
                raise

            # 1. Modify the passed dictionary directly. This modification will be reflected in the original dictionary of the main file.
            model_config['api_key'] = new_key
            logger.info("A new API Key has been received.")

            # 2. Dynamically create and replace model components
            logger.info("Dynamically creating a new model engine and refreshing all Agent instances...")
            try:
                if not all_agents_to_update:
                    raise ValueError("The Agent list is empty and cannot be refreshed.")

                ModelWrapperClass = type(all_agents_to_update[0].model)
                logger.info(f"Dynamically identify the model class: {ModelWrapperClass.__name__}")

                # Create a new instance using the modified model_config
                new_model_wrapper = ModelWrapperClass(**model_config)
                logger.info("The model engine has been successfully created using the new key.")

                for agent_instance in all_agents_to_update:
                    agent_instance.model = new_model_wrapper
                    logger.info(f"Agent refreshed successfully {agent_instance.name}")

                logger.info("All Agent instances have been refreshed.")

            except Exception as refresh_e:
                logger.error(f"A serious error occurred while refreshing the Agent instance: {refresh_e}", exc_info=True)
                raise

            logger.info("Will retry the failed API call...")

            # else:
            #     logger.error(f"An unknown API call error occurred: {e}")
            #     raise

def get_relevant_literature(
        hypothesis: str,
        json_filepath: str,
        top_k: int = 5,
        threshold: float = 0.7,
        text_fields: List[str] = ['title']
) -> List[Dict[str, Union[int, str, float]]]:
    """
    Retrieves the most relevant documents from a JSON document repository based on a given hypothesis.

    This function calculates a semantic similarity score between the hypothesis and each document's text.
    It uses the 'all-MiniLM-L6-v2' model to generate text embeddings.

    Args:
        hypothesis (str): The scientific hypothesis string to search for.
        json_filepath (str): The path to the JSON file containing the document data.
                                file should be a list of objects, one for each article.
        top_k (int, optional): The maximum number of most similar documents to return. Defaults to 5.
        threshold (float, optional): The minimum similarity score that the results must meet.
                                    The score ranges from 0 to 1. Defaults to 0.7.
        text_fields (List[str], optional): A list of text fields in the JSON object to use for similarity
                                            calculation.


    Returns:
        List[Dict[str, Union[int, str, float]]]:
            A list of dictionaries, each containing an 'id' and a 'score',
            sorted by similarity score from highest to lowest.
            Only results with a score above the threshold are included.
            An empty list is returned if the file is not found or is malformed.
    """
    # --- 1. Load Sentence Transformer ---
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please ensure 'sentence-transformers' and 'torch' are installed.")
        return []

    # --- 2. Load and handle JSON ---
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"Error:Unable to find JSON, PATH: {json_filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file, please check the file format: {json_filepath}")
        return []

    corpus_texts = []
    corpus_ids = []
    print(f"Preparing text corpus from {len(articles)} articles...")
    for article in articles:
        if 'ID' not in article:
            continue
        combined_text = ". ".join([article.get(field, '') for field in text_fields])
        if combined_text.strip():
            corpus_texts.append(combined_text)
            corpus_ids.append(article['ID'])

    if not corpus_texts:
        print("Warning: Could not find valid bibliographic text in the JSON file.")
        return []

    # --- 3. Embedding ---
    print("Generating embedding vectors for hypothesis and literature corpus...")
    hypothesis_embedding = model.encode(hypothesis, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

    # --- 4. Calculate cosine similarity ---
    print("Calculating similarity score...")
    cosine_scores = util.cos_sim(hypothesis_embedding, corpus_embeddings)

    # --- 5. Sort, filter, and extract results ---
    score_pairs = list(zip(corpus_ids, cosine_scores[0].cpu().numpy()))
    score_pairs_sorted = sorted(score_pairs, key=lambda x: x[1], reverse=True)

    # Filter results by threshold
    filtered_pairs = [pair for pair in score_pairs_sorted if pair[1] >= threshold]

    # Extract the top_k results
    top_results = filtered_pairs[:top_k]

    # Formatted output, including ID and score
    formatted_results = [{'ID': doc_id, 'score': float(score)} for doc_id, score in top_results]

    print("\nSearch complete.")
    return formatted_results

def Corpus_Assign_New_Documents_to_Experts(expert_agents,literature_filepath,current_hypothesis,logger):

    hypothesis_text = current_hypothesis.get("Initial Hypothesis", "")

    # --- Step 1: Retrieve new relevant literature using the initial hypothesis ---
    newly_retrieved_docs = get_relevant_literature(
        hypothesis=hypothesis_text,
        json_filepath=literature_filepath,
        # top_k=len(experts) * docs_per_expert
        top_k=12,
        threshold=0.3
    )

    # --- Step 2: Randomly assign new literature to each Expert ---
    random.shuffle(newly_retrieved_docs)

    # Assign in turn and add each paper ID directly to the corresponding expert's attribute list.
    logger.info(f"INFO: 正在为每位专家分配新的阅读文献，共{len(newly_retrieved_docs)}篇文献待分配...")
    for i, doc in enumerate(newly_retrieved_docs):
        expert_index = i % len(expert_agents)
        target_expert = expert_agents[expert_index]
        # Get the paper ID.
        doc_id = doc.get('ID')
        if doc_id is not None:
            # Add the ID directly to the target expert's new_documents_ids list.
            target_expert.new_documents_ids.append(doc_id)
    logger.info("INFO:分配完毕")
    return expert_agents

def fix_json_string(json_str):
    """
    Attempt to repair common JSON-format errors.
    """
    if not json_str:
        return json_str

    # Remove excess whitespace.
    json_str = json_str.strip()

    # Repair common JSON-format issues.
    fixes = [
        # Replace single quotes with double quotes.
        (r"'([^']*)':", r'"\1":'),
        (r":\s*'([^']*)'", r': "\1"'),

        # Add missing quotation marks around property names.
        (r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":'),

        # Convert True, False, and None to lowercase JSON values.
        (r'\bTrue\b', 'true'),
        (r'\bFalse\b', 'false'),
        (r'\bNone\b', 'null'),

        # Remove trailing commas.
        (r',(\s*[}\]])', r'\1'),

        # Repair simple cases of missing commas.
        (r'"\s*\n\s*"', '",\n"'),
        (r'}\s*\n\s*{', '},\n{'),
        (r']\s*\n\s*\[', '],\n['),
    ]

    for pattern, replacement in fixes:
        json_str = re.sub(pattern, replacement, json_str, flags=re.MULTILINE)

    return json_str


def parse_json_with_fallback(json_str):
    """
    Parse JSON through multiple strategies with automatic error correction.
    """
    if not json_str:
        return None

    # First attempt: parse directly.
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Second attempt: repair common errors before parsing.
    try:
        fixed_json = fix_json_string(json_str)
        return json.loads(fixed_json)
    except json.JSONDecodeError:
        pass

    # Third attempt: use ast.literal_eval for Python-dictionary syntax.
    try:
        return ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        pass

    # Fourth attempt: remove possible prefix and suffix content.
    try:
        # Find content from the first { or [ through the last } or ].
        start_match = re.search(r'[{\[]', json_str)
        if start_match:
            start_pos = start_match.start()
            # Find the matching end position.
            bracket_count = 0
            end_pos = len(json_str)
            start_char = json_str[start_pos]
            end_char = '}' if start_char == '{' else ']'

            for i in range(start_pos, len(json_str)):
                if json_str[i] == start_char:
                    bracket_count += 1
                elif json_str[i] == end_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i + 1
                        break

            cleaned_json = json_str[start_pos:end_pos]
            fixed_json = fix_json_string(cleaned_json)
            return json.loads(fixed_json)
    except (json.JSONDecodeError, IndexError):
        pass

    # Fifth attempt: use a regular expression to extract a possible JSON fragment.
    try:
        # Attempt to extract a JSON-like structure.
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        matches = re.findall(json_pattern, json_str, re.DOTALL)

        for match in matches:
            try:
                fixed_match = fix_json_string(match)
                return json.loads(fixed_match)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return None


def ConvertJSON_list(info):
    """
    Convert to a list with automatic error correction.
    """
    try:
        # Extract JSON content.
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|([\s\S]*)', info.content)
        if not json_match:
            print("Error: No JSON content found")
            return []

        json_str = json_match.group(1) or json_match.group(2)
        if not json_str or not json_str.strip():
            print("Error: Empty JSON content")
            return []

        # Attempt to parse JSON.
        parsed_json = parse_json_with_fallback(json_str.strip())

        if parsed_json is None:
            print(f"Error: Failed to parse JSON content:\n{json_str[:200]}...")
            return []

        if isinstance(parsed_json, list):
            return parsed_json
        else:
            print("Warning: Parsed content is not a list, converting to list")
            return [parsed_json]

    except Exception as e:
        print(f"Error in ConvertJSON_list: {str(e)}")
        return []


def ConvertJSON_dict(info):
    """
    Convert to a dictionary with automatic error correction.
    """
    try:
        # Extract JSON content.
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|([\s\S]*)', info.content)
        if not json_match:
            print("Error: No JSON content found")
            return {}

        json_str = json_match.group(1) or json_match.group(2)
        if not json_str or not json_str.strip():
            print("Error: Empty JSON content")
            return {}

        # Attempt to parse JSON.
        parsed_json = parse_json_with_fallback(json_str.strip())

        if parsed_json is None:
            print(f"Error: Failed to parse JSON content:\n{json_str[:200]}...")
            return {}

        if isinstance(parsed_json, dict):
            return parsed_json
        else:
            print("Warning: Parsed content is not a dict, returning empty dict")
            return {}

    except Exception as e:
        print(f"Error in ConvertJSON_dict: {str(e)}")
        return {}


import requests
import time
import logging

# Set up a simple logger so this code can run independently.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def LiFuAI_API_NewRelevant_Documents(query: str, num_papers: int, logger, delay: float = 2.0,
                              max_retries: int = 5):
    """
    Retrieve papers by keyword and return key information for each paper until the requested
    number of papers with abstracts has been obtained. This function uses the lifuai.com API.

    Args:
        query (str): Search keyword or phrase.
        num_papers (int): Total number of papers with abstracts to return.
        logger: Logger instance.
        api_key (str, optional): lifuai.com API key. Defaults to "".
        delay (float, optional): Seconds to wait after each API call. Defaults to 2.0.
        max_retries (int, optional): Maximum retries after rate limiting. Defaults to 5.

    Returns:
        list: Retrieved paper information, or None after maximum retries or when no results exist.
    """
    logger.info(f"\n开始使用关键词 '{query}' 检索 {num_papers} 篇包含摘要的论文 (通过 lifuai.com API)...")
    api_key = os.getenv("LIFUAI_API_KEY", "")
    headers = {'Authorization': f'Bearer {api_key}'}
    url = "https://lifuai.com/api/v1/graph/v1/paper/search"

    fields = [
        'title', 'authors', 'abstract', 'externalIds', 'year', 'venue', 'citationCount', 'influentialCitationCount'
    ]

    papers_list = []
    retries = 0
    offset = 0

    while len(papers_list) < num_papers and retries < max_retries:
        limit = 100  # The default and maximum limit for the lifuai.com API.
        params = {
            'query': query,
            'limit': limit,
            'offset': offset,
            'fields': ','.join(fields)
        }

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            search_results = data.get('data', [])

            if not search_results:
                logger.info("API 未返回更多与关键词匹配的论文，检索结束。")
                break

            found_with_abstract = 0
            for paper in search_results:
                if paper.get('abstract') and len(papers_list) < num_papers:
                    paper_info = {
                        'title': paper.get('title', 'N/A'),
                        'year': paper.get('year', 'N/A'),
                        'abstract': paper.get('abstract'),
                        'citationCount': paper.get('citationCount', 0),
                        'doi': paper.get('externalIds', {}).get('DOI', 'N/A')
                    }
                    papers_list.append(paper_info)
                    found_with_abstract += 1

            logger.info(
                f"本次请求获取 {len(search_results)} 篇，其中 {found_with_abstract} 篇包含摘要。当前总数: {len(papers_list)}/{num_papers}。")

            offset += limit

            # Check for another page because of the lifuai.com API pagination structure.
            if len(search_results) < limit:
                logger.info("已获取所有可用的论文结果。")
                break

            retries = 0
            time.sleep(delay)

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                retries += 1
                backoff_time = delay * (2 ** retries)
                logger.warning(f"请求失败: 429 Too Many Requests。第 {retries} 次重试，等待 {backoff_time:.2f} 秒...")
                time.sleep(backoff_time)
            else:
                logger.error(f"发生 HTTP 错误: {http_err}")
                break
        except requests.exceptions.RequestException as req_err:
            logger.error(f"请求失败: {req_err}")
            break

    if len(papers_list) >= num_papers:
        logger.info(f"\n已成功获取 {len(papers_list)} 篇包含摘要的论文。任务完成。")
    elif retries >= max_retries:
        logger.warning(f"\n已达到最大重试次数 ({max_retries})，未能完成检索。")
    else:
        logger.info(f"\n检索完成，共找到 {len(papers_list)} 篇包含摘要的论文，未能达到指定的 {num_papers} 篇。")

    return papers_list if papers_list else None




def assign_document_ids(keyword_documents: dict) -> dict:
    """
    Add a unique ID beginning with "E1" to every document in keyword_documents.

    Args:
        keyword_documents (dict): Dictionary of keywords and related document lists.
                                 Every document in each list is a dictionary.

    Returns:
        dict: Copy of the original dictionary with a "document_id" key added to each document.
    """
    if not isinstance(keyword_documents, dict):
        print("输入必须是一个字典。")
        return {}

    modified_documents = {}
    doc_counter = 1

    for keyword, documents_list in keyword_documents.items():
        if not isinstance(documents_list, list):
            print(f"键 '{keyword}' 对应的值不是一个列表，跳过。")
            continue

        updated_list = []
        for doc_json in documents_list:
            if not isinstance(doc_json, dict):
                print("列表中的元素不是字典，无法添加ID。")
                continue

            # Create a document ID such as "E1" or "E2".
            doc_id = f"E{doc_counter}"
            # Add a new key to the document dictionary.
            doc_json['document_id'] = doc_id
            updated_list.append(doc_json)
            doc_counter += 1

        modified_documents[keyword] = updated_list

    return modified_documents

def assign_extra_document_ids(keyword_documents: dict) -> dict:
    """
    Add a unique ID beginning with "E1" to every document in keyword_documents.

    Args:
        keyword_documents (dict): Dictionary of keywords and related document lists.
                                 Every document in each list is a dictionary.

    Returns:
        dict: Copy of the original dictionary with a "document_id" key added to each document.
    """
    if not isinstance(keyword_documents, dict):
        print("输入必须是一个字典。")
        return {}

    modified_documents = {}
    doc_counter = 1

    for keyword, documents_list in keyword_documents.items():
        if not isinstance(documents_list, list):
            print(f"键 '{keyword}' 对应的值不是一个列表，跳过。")
            continue

        updated_list = []
        for doc_json in documents_list:
            if not isinstance(doc_json, dict):
                print("列表中的元素不是字典，无法添加ID。")
                continue

            # Create a document ID such as "E1" or "E2".
            doc_id = f"E{doc_counter}"
            # Add a new key to the document dictionary.
            doc_json['ID'] = doc_id
            updated_list.append(doc_json)
            doc_counter += 1

        modified_documents[keyword] = updated_list

    return modified_documents


def transform_inspirations(input_filepath: str, output_filepath: str):
    """
    Read a JSON file containing multiple "window" entries, extract and transform each "content"
    field, and save the result as a newly formatted JSON file.

    Args:
        input_filepath (str): Path to the input JSON file.
        output_filepath (str): Path for the transformed output JSON file.
    """
    try:
        # 1. Read and parse the input JSON file.
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_transformed_inspirations = []
        inspiration_counter = 1

        # 2. Iterate over each "window" object in the file.
        for window in data:
            # 3. Extract and clean the "content" string.
            # Remove Markdown code-block markers.
            content_str = window.get('content', '[]')
            cleaned_content_str = content_str.strip().replace('```json', '').replace('```', '').strip()

            if not cleaned_content_str:
                continue

            # 4. Parse the cleaned string as a Python list.
            try:
                inspirations_list = json.loads(cleaned_content_str)
            except json.JSONDecodeError as e:
                print(f"警告: 在窗口 {window.get('window_number')} 中解析JSON时出错: {e}")
                continue

            # 5. Iterate over and transform each inspiration object in the list.
            for insp_obj in inspirations_list:
                inspiration_key = None
                # Locate the "Inspiration X" key dynamically.
                for key in insp_obj.keys():
                    if key.lower().startswith("inspiration"):  # Use .lower() for greater compatibility.
                        inspiration_key = key
                        break

                if not inspiration_key:
                    print(f"警告: 在对象中未找到 'Inspiration' 键: {insp_obj}")
                    continue

                # 6. Extract the ID, content, source, and reason.

                inspiration_id = str(inspiration_counter)

                # 7. Build the new dictionary structure.
                transformed_insp = {
                    "inspiration_id": inspiration_id,
                    "source": insp_obj.get("Source", ""),
                    "content": insp_obj.get(inspiration_key, ""),
                    "reasoning": insp_obj.get("Reasoning", "")
                }

                # 8. Add the transformed object to the final list.
                all_transformed_inspirations.append(transformed_insp)

                inspiration_counter += 1
        # 9. Write the final list to a new JSON file.
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_transformed_inspirations, f, indent=4, ensure_ascii=False)

        print(f"文件转换成功！结果已保存至: {output_filepath}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_filepath}")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")


def reindex_inspiration_ids(input_file_path, output_file_path):
    """
    Read a JSON file, renumber its "inspiration_id" fields beginning at 1,
    and save the result to a new JSON file.

    Args:
    input_file_path (str): Path to the input JSON file.
    output_file_path (str): Path to the output JSON file.
    """
    try:
        # 1. Read and parse the original JSON file.
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure that the data is a list.
        if not isinstance(data, list):
            print("错误：JSON文件的根结构不是一个列表。")
            return

        # 2. Iterate over the list and renumber "inspiration_id".
        # Use enumerate with start=1.
        for index, item in enumerate(data, start=1):
            # Check whether the dictionary contains an "inspiration_id" key.
            if 'inspiration_id' in item:
                item['inspiration_id'] = str(index) # Convert the new ID to a string to preserve the original format.
            else:
                print(f"警告：在第 {index} 个条目中未找到 'inspiration_id' 键，已跳过。")

        # 3. Write the modified data to a new JSON file.
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # indent=4 formats the JSON for readability.
            # ensure_ascii=False preserves non-ASCII characters for display.
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"处理完成！")
        print(f"总共 {len(data)} 条记录的 'inspiration_id' 已被重新排序。")
        print(f"结果已保存到: {output_file_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file_path}'。请检查文件名和路径是否正确。")
    except json.JSONDecodeError:
        print(f"错误：文件 '{input_file_path}' 不是有效的JSON格式。")
    except Exception as e:
        print(f"发生未知错误: {e}")


def collect_paper_ids_from_hypothesis(
        hypothesis: Dict[str, Any],
        inspiration_pool: List[Dict[str, Any]],
        logger
) -> List[str]:
    """
    Collect all relevant paper IDs from a hypothesis's Source fields.

    Args:
        hypothesis: Hypothesis containing Source_from_InspirationPool and Source_from_Papers.
        inspiration_pool: Complete inspiration pool.
        logger: Logger instance.

    Returns:
        List[str]: Deduplicated list of paper IDs.
    """
    paper_ids = []

    # 1. Process Source_from_Papers by adding paper IDs directly.
    source_from_papers = hypothesis.get("Source_from_Papers", "")
    if source_from_papers:
        if isinstance(source_from_papers, str):
            direct_paper_ids = [id.strip() for id in source_from_papers.split(',') if id.strip()]
        elif isinstance(source_from_papers, list):
            direct_paper_ids = [str(id).strip() for id in source_from_papers]
        else:
            direct_paper_ids = [str(source_from_papers).strip()]

        paper_ids.extend(direct_paper_ids)
        logger.info(f"从Source_from_Papers收集到 {len(direct_paper_ids)} 个论文ID: {direct_paper_ids}")

    # 2. Process Source_from_InspirationPool by finding paper IDs for the corresponding inspirations.
    source_from_inspiration = hypothesis.get("Source_from_InspirationPool", "")
    if source_from_inspiration:
        # Parse inspiration IDs.
        if isinstance(source_from_inspiration, str):
            inspiration_ids = [id.strip() for id in source_from_inspiration.split(',') if id.strip()]
        elif isinstance(source_from_inspiration, list):
            inspiration_ids = [str(id).strip() for id in source_from_inspiration]
        else:
            inspiration_ids = [str(source_from_inspiration).strip()]

        logger.info(f"需要从灵感库中查找 {len(inspiration_ids)} 个灵感ID: {inspiration_ids}")

        # Find the corresponding paper IDs in the inspiration pool.
        for insp_id in inspiration_ids:
            for inspiration in inspiration_pool:
                if inspiration.get("inspiration_id") == insp_id:
                    # Get the inspiration's source field.
                    insp_source = inspiration.get("source", "")
                    if insp_source:
                        if isinstance(insp_source, str):
                            insp_paper_ids = [id.strip() for id in insp_source.split(',') if id.strip()]
                        elif isinstance(insp_source, list):
                            insp_paper_ids = [str(id).strip() for id in insp_source]
                        else:
                            insp_paper_ids = [str(insp_source).strip()]

                        paper_ids.extend(insp_paper_ids)
                        logger.info(f"灵感 {insp_id} 对应的论文ID: {insp_paper_ids}")
                    break

    # Deduplicate while preserving order.
    unique_paper_ids = []
    seen = set()
    for pid in paper_ids:
        if pid not in seen and pid:  # Exclude empty strings.
            seen.add(pid)
            unique_paper_ids.append(pid)

    logger.info(f"总共收集到 {len(unique_paper_ids)} 个唯一的论文ID")
    return unique_paper_ids


# Assume ConvertJSON_list, make_api_call_with_retry, Msg, LLM_COP_CONFIG, logger, all_agents, and Expert_MODEL_CONFIG are defined in context.

def get_json_reconfirmation_prompt(original_response_content: str) -> str:
    """
    Generate a system prompt for secondary LLM validation.
    """
    prompt = f"""
    Your task is to fix a malformed JSON string.
    The original JSON string was generated by another model, but it cannot be parsed by a standard JSON parser.
    Please carefully read the following original response and extract the valid JSON content.
    The error may be due to improper formatting, or unnecessary comments added outside the JSON or LIST. We expect to obtain a standard JSON or LIST.

    ---START OF ORIGINAL RESPONSE---
    {original_response_content}
    ---END OF ORIGINAL RESPONSE---

    Do not include any additional explanations, Markdown tags (such as ```json```) or text besides the JSON or LIST content.
    """
    return prompt


def reconfirm_json_with_llm(
        expert_response: Msg,
        expert_agent: Any,  # Agent instance that calls the API and supports logging.
        all_agents: List[Any],
        logger: logging.Logger,
        LLM_CONFIG: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    When ConvertJSON_list fails, call another LLM to attempt to repair the JSON format.

    Args:
        expert_response (Msg): Response object from the original LLM.
        expert_agent (Any): Agent instance that initiated the call.
        all_agents (List[Any]): List of all agents.
        logger (logging.Logger): Logger instance.
        LLM_CONFIG (Dict[str, Any]): Configuration for the validation LLM, such as Expert_MODEL_CONFIG.
        expected_format_description (str): Description of the expected JSON format.

    Returns:
        List[Dict[str, Any]]: Repaired JSON list, or an empty list if repair fails.
    """
    logger.warning("ConvertJSON_list 初次解析失败，启动 LLM 二次校验...")

    # 1. Create the secondary-validation prompt.
    reconfirmation_content = get_json_reconfirmation_prompt(
        original_response_content=expert_response.content
    )

    reconfirmation_prompt = Msg(
        name="system",
        content=reconfirmation_content,
        role="user"
    )

    # 2. Call the secondary-validation LLM.
    # Reuse make_api_call_with_retry while ensuring that the LLM knows it is repairing JSON.
    # A stronger model, or at least the Expert model configuration, may be preferable for this repair.
    try:
        # Reuse expert_agent for the API call with the new prompt.
        reconfirm_response = make_api_call_with_retry(
            expert_agent,
            reconfirmation_prompt,
            all_agents,
            logger,
            LLM_CONFIG  # Continue using the Expert configuration for repair.
        )

        # 3. Attempt to parse the secondary-validation result.
        # Although the validation LLM should return raw JSON, ConvertJSON_list still handles possible Markdown wrappers.
        reconfirmed_content = ConvertJSON_list(reconfirm_response)

        if reconfirmed_content:
            logger.info("LLM 二次校验成功修复 JSON 格式。")
            return reconfirmed_content
        else:
            logger.error("LLM 二次校验失败，无法修复 JSON 格式。")
            return []

    except Exception as e:
        logger.error(f"LLM 二次校验过程中发生异常: {str(e)}")
        return []

# Helper: retrieve corresponding inspirations from the pool by Source.
def retrieve_inspirations_by_source(source_data: str, inspiration_pool: List[Dict[str, Any]], logger: logging.Logger) -> \
        List[Dict[str, Any]]:
    """
    Retrieve corresponding inspirations from the pool using a hypothesis's Source field.

    Args:
        source_data (str): Hypothesis Source field, such as "scr_merged_001, scr_merged_005, scr_unique_006".
        inspiration_pool (List[Dict[str, Any]]): Complete inspiration pool.
        logger (logging.Logger): Logger instance.

    Returns:
        List[Dict[str, Any]]: Retrieved inspirations.
    """

    # Handle different source_data types.
    source_ids = []

    if isinstance(source_data, str):
        # Split strings on commas.
        source_ids = [id.strip() for id in source_data.split(',')]
    elif isinstance(source_data, list):
        # Use an existing list directly while still stripping whitespace.
        source_ids = [str(id).strip() for id in source_data]
    else:
        # Otherwise, attempt to convert the value to a string before processing.
        logger.warning(f"Source字段类型未知: {type(source_data)}，尝试转换为字符串")
        source_ids = [str(source_data).strip()]

    # Retrieve the corresponding inspirations from the inspiration pool.
    retrieved_inspirations = []
    for inspiration in inspiration_pool:
        if inspiration.get("inspiration_id") in source_ids:
            retrieved_inspirations.append(inspiration)

    logger.info(f"从Source '{source_data}' 中检索到 {len(retrieved_inspirations)} 条灵感")
    return retrieved_inspirations


def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)
    for model, ranked_list in rankings.items():
        for rank_position, hypothesis_id in enumerate(ranked_list):
            scores[str(hypothesis_id)] += 1 / (k + rank_position + 1)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def enable_token_tracking(token_stats_dict: Dict, verbose: bool = False):
    """
    Enable global token tracking.

    Args:
        token_stats_dict: Dictionary used to store token statistics.
        verbose: Whether to print token usage for each call.
    """
    original_call = OpenAIChatWrapper.__call__

    @functools.wraps(original_call)
    def wrapped_call(self, messages: List[dict], **kwargs: Any) -> Any:
        """Intercept model calls and track tokens."""
        response = original_call(self, messages, **kwargs)

        try:
            if hasattr(response, 'raw') and isinstance(response.raw, dict):
                usage = response.raw.get('usage')

                if usage and isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)

                    token_stats_dict['input_tokens'] += prompt_tokens
                    token_stats_dict['output_tokens'] += completion_tokens
                    token_stats_dict['total_tokens'] += total_tokens

                    if verbose:
                        print(f"  ✓ Token: +{prompt_tokens} input, "
                              f"+{completion_tokens} output, "
                              f"+{total_tokens} total")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Token 统计失败: {e}")

        return response

    OpenAIChatWrapper.__call__ = wrapped_call
    if verbose:
        print("✓ 已启用全局 Token 统计\n")


def evaluate_clustering_quality(
        embeddings_array: np.ndarray,
        labels: np.ndarray,
        silicon_jury_survivors: List[str],
        hypothesis_map: Dict[str, Dict],
        rrf_results: List[Tuple],
        save_dir: str,
        logger: logging.Logger
):
    """
    Evaluate Silicon Jury clustering quality.

    Args:
        embeddings_array: Embedding array with shape (n_samples, n_features).
        labels: Cluster labels with shape (n_samples,).
        silicon_jury_survivors: IDs of the 30 hypotheses selected by Silicon Jury.
        hypothesis_map: Mapping from hypothesis IDs to complete hypothesis objects.
        rrf_results: RRF ranking results as [(hyp_id, score), ...].
        save_dir: Output directory.
        logger: Logger instance.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
    import scipy.stats as stats

    logger.info(f"\n{'=' * 60}")
    logger.info("CLUSTERING QUALITY EVALUATION")
    logger.info(f"{'=' * 60}\n")

    evaluation_report = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": "Silicon Jury Clustering Quality Assessment",
        "metrics": {}
    }

    # ========== (A) Silhouette Score ==========
    try:
        silhouette = silhouette_score(embeddings_array, labels, metric="cosine")
        evaluation_report["metrics"]["silhouette_score"] = float(silhouette)

        if silhouette < 0.1:
            quality = "POOR - Clustering failed"
        elif silhouette < 0.3:
            quality = "FAIR - Barely acceptable"
        elif silhouette < 0.5:
            quality = "GOOD"
        elif silhouette < 0.7:
            quality = "VERY GOOD"
        else:
            quality = "EXCELLENT"

        logger.info(f"(A) Silhouette Score: {silhouette:.4f} - {quality}")
        evaluation_report["metrics"]["silhouette_interpretation"] = quality
    except Exception as e:
        logger.error(f"Failed to calculate Silhouette Score: {e}")
        evaluation_report["metrics"]["silhouette_score"] = None

    # ========== (B) Mean Intra-Cluster Distance ==========
    try:
        intra_dists = []
        cluster_details = []

        for cid in set(labels):
            members = embeddings_array[labels == cid]
            if len(members) > 1:
                dist = cosine_distances(members).mean()
                intra_dists.append(dist)
                cluster_details.append({
                    "cluster_id": int(cid),
                    "size": int(len(members)),
                    "intra_distance": float(dist)
                })

        mean_intra = np.mean(intra_dists) if intra_dists else 0.0
        evaluation_report["metrics"]["mean_intra_cluster_distance"] = float(mean_intra)
        evaluation_report["metrics"]["cluster_cohesion_details"] = cluster_details

        intra_quality = "Tight" if mean_intra < 0.35 else "Loose"
        logger.info(f"(B) Mean Intra-Cluster Distance: {mean_intra:.4f} - {intra_quality}")
        logger.info(f"    Recommendation: < 0.35 is good (lower = tighter clusters)")
    except Exception as e:
        logger.error(f"Failed to calculate Intra-Cluster Distance: {e}")
        evaluation_report["metrics"]["mean_intra_cluster_distance"] = None

    # ========== (C) Inter-Cluster Distance ==========
    try:
        # Obtain cluster centers.
        centers = []
        for cid in set(labels):
            members = embeddings_array[labels == cid]
            center = members.mean(axis=0)
            centers.append(center)

        centers_array = np.array(centers)
        inter_dist = cosine_distances(centers_array).mean()
        evaluation_report["metrics"]["mean_inter_cluster_distance"] = float(inter_dist)

        inter_quality = "Well-separated" if inter_dist > 0.45 else "Too close"
        logger.info(f"(C) Mean Inter-Cluster Distance: {inter_dist:.4f} - {inter_quality}")
        logger.info(f"    Recommendation: > 0.45 is good (higher = better separation)")
    except Exception as e:
        logger.error(f"Failed to calculate Inter-Cluster Distance: {e}")
        evaluation_report["metrics"]["mean_inter_cluster_distance"] = None

    # ========== (D) Cluster Size Distribution ==========
    try:
        cluster_sizes = np.bincount(labels)
        size_entropy = stats.entropy(cluster_sizes + 1e-10)  # Avoid log(0).

        evaluation_report["metrics"]["cluster_size_entropy"] = float(size_entropy)
        evaluation_report["metrics"]["cluster_sizes"] = cluster_sizes.tolist()
        evaluation_report["metrics"]["cluster_size_statistics"] = {
            "min": int(cluster_sizes.min()),
            "max": int(cluster_sizes.max()),
            "mean": float(cluster_sizes.mean()),
            "std": float(cluster_sizes.std())
        }

        logger.info(f"(D) Cluster Size Entropy: {size_entropy:.4f}")
        logger.info(f"    Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
                    f"mean={cluster_sizes.mean():.1f}, std={cluster_sizes.std():.1f}")

        # Check for mode collapse.
        max_cluster_ratio = cluster_sizes.max() / len(labels)
        if max_cluster_ratio > 0.5:
            logger.warning(
                f"WARNING: Largest cluster contains {max_cluster_ratio * 100:.1f}% of items (Mode Collapse)")
            evaluation_report["warnings"] = evaluation_report.get("warnings", [])
            evaluation_report["warnings"].append(
                f"Mode collapse detected: {max_cluster_ratio * 100:.1f}% in one cluster")

        singleton_count = (cluster_sizes == 1).sum()
        logger.info(f"    Singleton clusters: {singleton_count}/{len(cluster_sizes)}")
        evaluation_report["metrics"]["singleton_clusters"] = int(singleton_count)
    except Exception as e:
        logger.error(f"Failed to calculate Cluster Size Distribution: {e}")
        evaluation_report["metrics"]["cluster_size_entropy"] = None

    # ========== (E) Semantic Diversity of Survivors ==========
    try:
        # Get indices of the selected hypotheses.
        hypothesis_id_to_idx = {str(hyp_id): idx for idx, (hyp_id, _) in enumerate(rrf_results)}
        selected_indices = [hypothesis_id_to_idx[str(sid)] for sid in silicon_jury_survivors
                            if str(sid) in hypothesis_id_to_idx]

        selected_embeddings = embeddings_array[selected_indices]
        survivor_sim_matrix = cosine_similarity(selected_embeddings)

        # Use only the upper triangle, excluding the diagonal.
        triu_indices = np.triu_indices_from(survivor_sim_matrix, k=1)
        mean_survivor_similarity = survivor_sim_matrix[triu_indices].mean()

        evaluation_report["metrics"]["mean_survivor_similarity"] = float(mean_survivor_similarity)

        if mean_survivor_similarity > 0.8:
            diversity_quality = "Too repetitive"
        elif mean_survivor_similarity > 0.6:
            diversity_quality = "Moderate diversity"
        elif mean_survivor_similarity > 0.4:
            diversity_quality = "Good diversity"
        else:
            diversity_quality = "Excellent diversity"

        logger.info(f"(E) Mean Survivor Similarity: {mean_survivor_similarity:.4f} - {diversity_quality}")
        logger.info(f"    Recommendation: < 0.6 is good (lower = more diverse)")
    except Exception as e:
        logger.error(f"Failed to calculate Survivor Diversity: {e}")
        evaluation_report["metrics"]["mean_survivor_similarity"] = None

    # ========== (F) Redundancy Reduction Rate ==========
    try:
        # Calculate similarity for the RRF Top 30 without clustering.
        top30_rrf_indices = list(range(min(30, len(rrf_results))))
        top30_rrf_embeddings = embeddings_array[top30_rrf_indices]
        rrf_sim_matrix = cosine_similarity(top30_rrf_embeddings)
        mean_rrf_similarity = rrf_sim_matrix[np.triu_indices_from(rrf_sim_matrix, k=1)].mean()

        # Calculate the redundancy-reduction rate.
        redundancy_reduction = 1 - (mean_survivor_similarity / mean_rrf_similarity) if mean_rrf_similarity > 0 else 0

        evaluation_report["metrics"]["rrf_only_top30_similarity"] = float(mean_rrf_similarity)
        evaluation_report["metrics"]["redundancy_reduction_rate"] = float(redundancy_reduction)

        logger.info(f"(F) Redundancy Reduction Rate: {redundancy_reduction * 100:.2f}%")
        logger.info(f"    RRF-only Top-30 similarity: {mean_rrf_similarity:.4f}")
        logger.info(f"Silicon Jury similarity: {mean_survivor_similarity:.4f}")

        if redundancy_reduction > 0:
            logger.info(f"Silicon Jury reduced semantic redundancy by {redundancy_reduction * 100:.2f}%")
        else:
            logger.warning(f"No redundancy reduction achieved")
    except Exception as e:
        logger.error(f"Failed to calculate Redundancy Reduction: {e}")
        evaluation_report["metrics"]["redundancy_reduction_rate"] = None

    # ========== Overall Assessment ==========
    logger.info(f"\n{'=' * 60}")
    logger.info("OVERALL ASSESSMENT")
    logger.info(f"{'=' * 60}")

    overall_quality = []

    if evaluation_report["metrics"].get("silhouette_score", 0) > 0.35:
        overall_quality.append("Good clustering structure")
    else:
        overall_quality.append("Weak clustering structure")

    if evaluation_report["metrics"].get("mean_survivor_similarity", 1.0) < 0.6:
        overall_quality.append("High diversity achieved")
    else:
        overall_quality.append("Limited diversity")

    if evaluation_report["metrics"].get("redundancy_reduction_rate", 0) > 0.1:
        overall_quality.append("Effective redundancy reduction")
    else:
        overall_quality.append("Limited redundancy reduction")

    evaluation_report["overall_assessment"] = overall_quality

    for item in overall_quality:
        logger.info(f"  {item}")

    # ========== Save Report ==========
    report_file = os.path.join(save_dir, "clustering_quality_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, indent=4, ensure_ascii=False)

    logger.info(f"\nClustering quality report saved to: {report_file}")
    logger.info(f"{'=' * 60}\n")

    return evaluation_report
