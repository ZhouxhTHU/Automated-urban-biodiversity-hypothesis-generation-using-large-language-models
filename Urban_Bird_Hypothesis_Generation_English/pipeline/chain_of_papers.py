
import requests
import time
import logging
import os
import json
import random
from typing import List, Dict, Optional, Callable
import config
research_topic = config.CONFIG["research"]["RESEARCH_TOPIC"]
research_background = config.CONFIG["research"]["RESEARCH_BACKGROUND"]

# --- Keep the SentenceTransformer-related import unchanged ---
from sentence_transformers import SentenceTransformer, util

# --- Global model cache ---
_global_model = None
_model_cache_path = None


chain_of_papers_token_stats = {
    'input_tokens': 0,
    'output_tokens': 0,
    'total_tokens': 0
}


def call_llm_for_queries(base_hypothesis: str, num_queries: int, config: Dict, avoid_queries: List[str] = None) -> list:
    """
    Use requests to call the LLM API and generate multiple search queries related to the research topic.
    Each query represents a different research perspective.
    """
    logger = logging.getLogger()
    logger.info(f"调用LLM ({config['MODEL_NAME']}) 生成 {num_queries} 个 '{base_hypothesis}' 的不同检索视角...")

    if config['API_KEY'] == "YOUR_LLM_API_KEY":
        logger.error("LLM API 密钥未配置。回退到默认查询。")
        return [base_hypothesis]

    url = f"{config['BASE_URL']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['API_KEY']}",
        "Content-Type": "application/json"
    }

    # --- Build instructions that avoid previous failed approaches ---
    avoid_instruction = ""
    if avoid_queries and len(avoid_queries) > 0:
        avoid_str = ", ".join([f'"{q}"' for q in avoid_queries])
        avoid_instruction = f"""
        ## Previous attempts
        The following queries have already been tried. You MUST generate NOVEL and DIFFERENT queries from these:
        [{avoid_str}]
        """

    # Construct the prompt following the paper's
    # "Prompt used to convert a topic into a search query for literature retrieval" (Table 7).
    prompt = f"""
    ## Role and Tasks
    You are a master of literature searching, and an expert in **urban biodiversity, specialized in urban bird biodiversity**.
    Your task is to generate highly relevant queries for retrieving literature from Semantic Scholar Database, with the goal of refining a scientific hypothesis.
    The research topic is {research_topic}, and the research background is {research_background}.
    The hypothesis to be refined is:
    {base_hypothesis}
    
    Please generate **{num_queries} queries**. 
    
    ## IMPORTANT REQUIREMENTS
    1. Queries should aim to retrieve literature relevant to the hypothesis.
    2. Each query must target the **causal link or coupling process** between the IF and THEN components of the hypothesis.
    3. Each query should be concise, typically ** 4-6 words**.
    4. Prefer queries that reflect specific processes that plausibly connect the cause and outcome in the hypothesis.
    5. Avoid overly broad or generic queries that only describe a topic without indicating a meaningful process or relationship.
    6. Queries should remain grounded in **urban bird biodiversity** and avoid drifting into unrelated taxa or domains unless clearly relevant.
    {avoid_instruction}
    
    ## Output Format
    Return your response strictly in the following JSON format, without any additional text:
    
    {{
      "queries": [
        "query 1",
        "query 2",
        "query 3"
      ]
    }}
    """

    data = {
        "model": config['MODEL_NAME'],
        "messages": [{"role": "user", "content": prompt}],
        # "temperature": 0.7,
        # Ask the model for JSON; models without support may require prompt or post-processing changes.
        "response_format": {"type": "json_object"},
        "timeout": config["timeout"]
    }

    # Catch JSONDecodeError in case the API does not support response_format.

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        # Assume an OpenAI-compatible response structure.
        response_data = response.json()

        # Extract and track token usage.
        if 'usage' in response_data:
            usage = response_data['usage']
            chain_of_papers_token_stats['input_tokens'] += usage.get('prompt_tokens', 0)
            chain_of_papers_token_stats['output_tokens'] += usage.get('completion_tokens', 0)
            chain_of_papers_token_stats['total_tokens'] += usage.get('total_tokens', 0)
            logger.info(f"✓ Token统计: +{usage.get('prompt_tokens', 0)} input, "
                        f"+{usage.get('completion_tokens', 0)} output")

        if response_data.get('choices'):
            message_content = response_data['choices'][0]['message']['content'].strip()

            # Attempt to parse JSON.
            try:
                json_content = json.loads(message_content)
                queries = json_content.get("queries", [])
            except json.JSONDecodeError:
                logger.warning("LLM 响应不是标准的 JSON 格式。尝试文本后处理。")
                # Fallback to text processing if JSON parsing fails
                # This simple text processing might not work for all LLM outputs
                queries = [q.strip() for q in message_content.split("\n") if q.strip()]

            final_queries = [q.strip(' "') for q in queries if q.strip()]

            logger.info(f"LLM 生成的检索查询 ({len(final_queries)} 个): {final_queries}")

            if not final_queries:
                logger.warning("LLM 生成的查询为空。回退到默认查询。")
                return [base_hypothesis]

            return final_queries

        else:
            logger.error(f"LLM 响应中没有 'choices': {response_data}")
            return [base_hypothesis]

    except requests.exceptions.RequestException as e:
        logger.error(f"调用 LLM API 失败: {e}")
        return [base_hypothesis]

def _rerank_by_llm(papers: List[Dict], query: str, top_n: int = 10, config: Dict = None) -> List[Dict]:
    """
    Use an LLM to rank papers by similarity.
    The LLM analyzes each paper's relevance to the query and returns a ranking.
    """
    logger = logging.getLogger()
    logger.info(f"--- 使用LLM重排序 {len(papers)} 篇论文 ---")

    if not papers:
        return []

    if config['API_KEY'] == "YOUR_LLM_API_KEY":
        logger.error("LLM API 密钥未配置。回退到原始排序")
        return papers[:top_n]

    # Prepare paper information for LLM analysis.
    papers_info = []
    valid_papers = []
    for idx, paper in enumerate(papers):
        title = paper.get('title', '') or ''
        abstract = paper.get('abstract', '') or ''
        if title or abstract:
            papers_info.append({
                "index": idx,
                "title": title,
                "abstract": abstract
            })
            valid_papers.append(paper)

    if not papers_info:
        logger.warning("没有有效的论文信息可供LLM排序")
        return []

    # Build the prompt.
    papers_text = "\n\n".join([
        f"Paper {p['index']}:\nTitle: {p['title']}\nAbstract: {p['abstract']}"
        for p in papers_info[:30]  # Limit to 30 papers to avoid excessive token usage.
    ])

    prompt = f"""
        ## Role and Tasks
        You are an **academic literature reviewer in urban bird biodiversity, good at recognizing semantic relevance**. 
        Your task is to evaluate relevance between a set of candidate academic papers and the hypothesis.
        The research topic is {research_topic} and the research background is {research_background}
        **Hypothesis:** {query}
        Below is a list of candidate academic papers, each identified by a unique index.
        {papers_text}
        **CORE TASK: Relevance Ranking**
        - You MUST prioritize papers clearly operating within **urban bird biodiversity** and closely related to the research topic and the research background. 
        - Check all candidate papers and EXCLUDE those that are not closely related to the research topic and background.
        - Analyze each paper's **Title** and **Abstract** to determine its relevance to the hypothesis. ONLY return the relevance ranking for papers that are strictly relevant.
        - Choose the paper that **could truly help refine the hypothesis.**
        
        ## Output Format
        Your final response MUST be a single, valid JSON list (`[]`) containing several JSON objects.
        DO NOT include any introductory text, explanations, or markdown formatting like ```json before or after the JSON list.
        Ensure all strings within the JSON are properly escaped.
        The list must be ranked by relevance, **from highest (most relevant) to lowest (least relevant)**.
        Each JSON object in the list must strictly adhere to the following structure:
        {{
            "index": "", 
        }}
        Example output (rank by relevance, most relevant first):
        [{{"index": "0"}}, {{"index": "2"}}, {{"index": "1"}}]
        """

    url = f"{config['BASE_URL']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['API_KEY']}",
        "Content-Type": "application/json"
    }

    data = {
        "model": config['MODEL_NAME'],
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }

    # Review possible changes here.
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        response_data = response.json()

        # Extract and track token usage.
        if 'usage' in response_data:
            usage = response_data['usage']
            chain_of_papers_token_stats['input_tokens'] += usage.get('prompt_tokens', 0)
            chain_of_papers_token_stats['output_tokens'] += usage.get('completion_tokens', 0)
            chain_of_papers_token_stats['total_tokens'] += usage.get('total_tokens', 0)
            logger.info(f"✓ Token统计: +{usage.get('prompt_tokens', 0)} input, "
                        f"+{usage.get('completion_tokens', 0)} output")

        if response_data.get('choices'):
            message_content = response_data['choices'][0]['message']['content'].strip()

            try:
                # Attempt to parse JSON.
                rankings = json.loads(message_content)

                # Compatibility check: some models wrap a list under a key in json_object mode.
                # If the response has the form {'ranking': [...]}, attempt to extract the list.
                if isinstance(rankings, dict):
                    # Look for common list keys; otherwise assume rankings is not the required list.
                    for key in ['rankings', 'papers', 'result', 'list']:
                        if key in rankings and isinstance(rankings[key], list):
                            rankings = rankings[key]
                            break

                if not isinstance(rankings, list):
                    logger.warning(
                        f"LLM output format error: Expected list, got {type(rankings)}. Content: {message_content[:100]}...")
                    return valid_papers[:top_n]

                # 1. Extract the indices ranked by the LLM.
                ranked_indices = []
                for item in rankings:
                    # Handle items that may be strings or dictionaries.
                    if isinstance(item, str):
                        try:
                            item = json.loads(item)
                        except:
                            pass  # Ignore strings that cannot be parsed.

                    if isinstance(item, dict) and 'index' in item:
                        try:
                            # Convert to int to match indices in valid_papers.
                            idx = int(item['index'])
                            ranked_indices.append(idx)
                        except (ValueError, TypeError):
                            continue

                if not ranked_indices:
                    logger.warning("LLM returned no valid indices. Falling back to original order.")
                    return valid_papers[:top_n]

                # 2. Reassemble the paper list according to the indices.
                reordered_papers = []
                seen_indices = set()

                # A. Add papers ranked by the LLM.
                for idx in ranked_indices:
                    # Ensure the index is valid and has not already been added.
                    if 0 <= idx < len(valid_papers) and idx not in seen_indices:
                        reordered_papers.append(valid_papers[idx])
                        seen_indices.add(idx)

                # B. As a fallback, append papers omitted by the LLM.
                # This prevents truncated LLM output from dropping candidate papers.
                for i, paper in enumerate(valid_papers):
                    if i not in seen_indices:
                        reordered_papers.append(paper)

                logger.info(f"LLM 成功重排序。前5篇原索引: {ranked_indices[:5]}")

                return reordered_papers[:top_n]

            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON. Content: {message_content}")
                return valid_papers[:top_n]
        else:
            logger.error(f"LLM response did not contain 'choices': {response_data}")
            return valid_papers[:top_n]

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API call failed: {e}")
        return valid_papers[:top_n]


def SemanticScholar_API_Search_Documents(query: str, num_papers: int, logger, headers: dict = None,
                                         delay: float = 2.0, max_retries: int = 5):
    """
    Retrieve papers by query text using the Semantic Scholar API.
    """

    logger.info(f"\n开始使用查询文本检索 {num_papers} 篇论文 (通过 Semantic Scholar API)...")
    logger.info(f"查询文本: '{query}...'")
    if headers is None: headers = {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = ['title', 'abstract', 'year']

    papers_list = []
    retries = 0
    offset = 0

    while len(papers_list) < num_papers and retries < max_retries:
        limit = 100
        params = {'query': query, 'limit': limit, 'offset': offset, 'fields': ','.join(fields)}

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            search_results = data.get('data', [])

            if not search_results:
                logger.info("API 未返回更多论文,检索结束。")
                break

            found_with_abstract = 0
            for paper in search_results:
                if paper.get('abstract') and len(papers_list) < num_papers:
                    papers_list.append(paper)
                    found_with_abstract += 1

            logger.info(
                f"本次请求获取 {len(search_results)} 篇,其中 {found_with_abstract} 篇包含摘要。当前总数: {len(papers_list)}/{num_papers}。")
            offset += limit
            if 'next' not in data or not data.get('next'):
                logger.info("已获取所有可用的论文结果。")
                break

            retries = 0
            time.sleep(delay)

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                retries += 1
                backoff_time = delay * (2 ** retries)
                logger.warning(f"请求失败: 429 Too Many Requests。第 {retries} 次重试,等待 {backoff_time:.2f} 秒...")
                time.sleep(backoff_time)
            else:
                logger.error(f"发生 HTTP 错误: {http_err}")
                break
        except requests.exceptions.RequestException as req_err:
            logger.error(f"请求失败: {req_err}")
            break

    if len(papers_list) >= num_papers:
        logger.info(f"\n已成功获取 {len(papers_list)} 篇论文。")
    else:
        logger.warning(f"\n检索完成,共找到 {len(papers_list)} 篇论文,未能达到指定的 {num_papers} 篇。")

    return papers_list if papers_list else None


def _make_api_request(url: str, params: dict, headers: dict, logger,
                      delay: float, max_retries: int, request_type: str) -> List[Dict]:
    """Handle API requests through a shared implementation."""
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data or not isinstance(data, dict):
                logger.warning(f"从API为 '{request_type}' 收到了无效或空的响应。")
                return []

            results = data.get('data', [])
            if not results:
                logger.info(f"API响应中未找到 '{request_type}' 的有效数据。")
                return []

            papers = []

            if request_type == 'search':
                for item in results:
                    if item and item.get('abstract'):
                        papers.append(item)
            else:
                paper_key = 'citedPaper' if request_type == "references" else 'citingPaper'
                for item in results:
                    paper = item.get(paper_key)
                    if paper and paper.get('abstract'):
                        papers.append(paper)


            logger.info(f"API返回 {len(results)} 篇论文,其中 {len(papers)} 篇包含摘要")
            return papers

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 403:
                logger.error(f"发生 HTTP 403 禁止访问错误: {http_err}. 您的API密钥可能无效或缺失。")
                return []
            elif http_err.response.status_code == 429:
                retries += 1
                backoff_time = delay * (2 ** retries)
                logger.warning(f"请求失败: 429 Too Many Requests. 第 {retries} 次重试,等待 {backoff_time:.2f} 秒...")
                time.sleep(backoff_time)
            else:
                logger.error(f"发生 HTTP 错误: {http_err}")
                return []
        except requests.exceptions.RequestException as req_err:
            logger.error(f"请求失败: {req_err}")
            return []

    logger.error(f"达到最大重试次数,无法获取 {request_type}")
    return []


def _get_paper_references(paper_id: str, headers: dict, base_url: str, fields: list,
                          logger, delay: float, max_retries: int) -> List[Dict]:
    """Retrieve a paper's references."""
    url = f"{base_url}/paper/{paper_id}/references"
    params = {'fields': ','.join(fields), 'limit': 30}
    return _make_api_request(url, params, headers, logger, delay, max_retries, "references")


def _get_paper_citations(paper_id: str, headers: dict, base_url: str, fields: list,
                         logger, delay: float, max_retries: int) -> List[Dict]:
    """Retrieve papers that cite the given paper."""
    url = f"{base_url}/paper/{paper_id}/citations"
    params = {'fields': ','.join(fields), 'limit': 30}
    return _make_api_request(url, params, headers, logger, delay, max_retries, "citations")


def _find_anchor_paper(query: str, hypothesis: str, search_func: Callable,
                       logger, delay: float, max_retries: int, seen_paper_ids,
                       min_references: int = 1, min_citations: int = 1,
                       headers: dict = None, base_url: str = None,
                       fields: list = None,  LLM_CONFIG: Optional[Dict] = None) -> Optional[Dict]:
    search_results = search_func(query, 30, logger, headers=headers, delay=delay, max_retries=max_retries)

    if not search_results:
        logger.warning("初始搜索中未找到相关论文。")
        return None

    papers_with_abstract = [p for p in search_results if p.get('abstract')]
    if not papers_with_abstract:
        logger.warning("搜索结果不包含任何带摘要的论文。")
        return None

    logger.info(f"筛选到 {len(papers_with_abstract)} 篇带摘要的论文进行重排序。")
    logger.info("基于假设的语义相关性对搜索结果进行重排序...")

    # Obtain the top-ten candidate paper pool using the selected ranking method.
    logger.info("使用 LLM 进行相似度排序...")
    reranked_papers = _rerank_by_llm(papers_with_abstract, hypothesis, top_n=10, config=LLM_CONFIG)

    if not reranked_papers:
        logger.warning("重排序未产生任何候选论文。")
        return None

    logger.info(f"验证前 {len(reranked_papers)} 个候选论文的连接性...")
    logger.info(f"(注意: 'referenceCount' 字段不可用,使用实际API调用)")

    # Validate connectivity with an actual API call.
    if headers is None or base_url is None or fields is None:
        logger.warning("无法验证连接性 - 缺少API参数。使用第一个候选论文。")
        return reranked_papers[0]

    for idx, paper in enumerate(reranked_papers):
        paper_id = paper.get('paperId')
        if not paper_id:
            logger.info(f"候选论文 {idx + 1}: 跳过 (无paperId)")
            continue
        if paper_id in seen_paper_ids:
            logger.info(f"候选论文 {idx + 1}: 跳过 (paperId已使用)")
            continue

        title_preview = paper.get('title', 'N/A')[:60]
        logger.info(f"候选论文 {idx + 1}: '{title_preview}...' - 检查连接性...")

        try:
            actual_refs = _get_paper_references(
                paper_id, headers, base_url, fields, logger,
                delay, max_retries
            )

            actual_cites = _get_paper_citations(
                paper_id, headers, base_url, fields, logger,
                delay, max_retries
            )

            actual_ref_count = len(actual_refs) if actual_refs else 0
            actual_cite_count = len(actual_cites) if actual_cites else 0

            logger.info(f"→ 找到 {actual_ref_count} 个可访问的参考文献, "
                        f"{actual_cite_count} 个可访问的引用")

            if actual_ref_count >= min_references and actual_cite_count >= min_citations:
                seen_paper_ids.add(paper_id)
                logger.info(f"✓ 选择为锚点论文 ")
                logger.info(f"  标题: '{paper.get('title')}'")
                logger.info(f"  已验证: {actual_ref_count} 个参考文献, {actual_cite_count} 个引用")
                return paper
            else:
                logger.info(f"  ✗ 低于阈值 (需要 ≥{min_references} 个参考文献, ≥{min_citations} 个引用)")

            time.sleep(delay)

        except Exception as e:
            logger.warning(f"  ! 检查连接性时出错: {e}")
            continue

    # Fallback: return the most semantically relevant paper.
    logger.warning(f"没有候选论文满足连接性阈值 "
                   f"(≥{min_references} 个参考文献, ≥{min_citations} 个引用)")
    logger.warning("回退到语义最相关的论文")
    best_paper = reranked_papers[0]
    logger.info(f"回退锚点: '{best_paper.get('title', 'N/A')[:60]}...'")
    return best_paper


def _get_reference_chain(paper_id: str, count: int, headers: dict, base_url: str, seen_paper_ids,
                         fields: list, query: str, hypothesis: str, logger, delay: float, max_retries: int,
                         LLM_CONFIG=None) -> List[Dict]:
    """Build a reference chain containing earlier papers."""
    chain, current_id = [], paper_id
    for i in range(count):
        logger.info(f"向前追溯: 获取第 {i + 1} 层参考文献...")
        references = _get_paper_references(current_id, headers, base_url, fields, logger, delay, max_retries)
        if not references:
            logger.info(f"第 {i + 1} 层无参考文献,停止向前扩展")
            break

        # In the CoI paper, backward expansion appears to select the most relevant reference via an LLM prompt.
        # This implementation simplifies that step to semantic reranking; the paper's logic would be more precise.
        reranked_refs = _rerank_by_llm(references, hypothesis, top_n=10, config=LLM_CONFIG)

        # Find the first ranked candidate that has references.
        best_ref = None

        # At the final level, use the paper with the highest similarity directly.
        is_last_layer = (i == count - 1)

        for candidate in reranked_refs:
            candidate_id = candidate.get('paperId')
            if not candidate.get('paperId'):
                continue

            if candidate_id in seen_paper_ids:
                logger.info(f"  跳过候选论文（paperId已使用）: '{candidate.get('title', 'N/A')[:50]}...'")
                continue

            # Final level: use the candidate with the highest similarity directly.
            if is_last_layer:
                best_ref = candidate
                logger.info(f"最后一层无需检查,直接选择相似度最高的论文: '{candidate.get('title', 'N/A')[:50]}...'")
                break

            logger.info(f"  检查候选论文 '{candidate.get('title', 'N/A')[:50]}...' 是否有参考文献...")

            # Check whether the candidate paper has references.
            candidate_references = _get_paper_references(
                candidate_id, headers, base_url, fields, logger,
                delay * 0.5, max_retries
            )

            if candidate_references and len(candidate_references) > 0:
                best_ref = candidate
                break
            else:
                print(f"  ✗ 该论文无参考文献，跳过")

            time.sleep(delay * 0.3)

        if best_ref and best_ref.get('paperId'):
            seen_paper_ids.add(best_ref['paperId'])
            chain.insert(0, best_ref)
            current_id = best_ref['paperId']
            logger.info(f"  第 {i + 1} 层已添加: '{best_ref.get('title', 'N/A')[:60]}...'")
        else:
            logger.warning(f"第 {i + 1} 层所有候选论文都无参考文献,停止向前扩展")
            break

        time.sleep(delay)

    return chain


def _get_citation_chain(paper_id: str, count: int, headers: dict, base_url: str, seen_paper_ids,
                        fields: list, query: str, hypothesis: str, logger, delay: float, max_retries: int,
                        LLM_CONFIG=None) -> List[Dict]:
    """Build a citation chain containing later papers."""
    chain, current_id = [], paper_id
    for i in range(count):
        logger.info(f"向后追溯: 获取第 {i + 1} 层引用文献...")
        citations = _get_paper_citations(current_id, headers, base_url, fields, logger, delay, max_retries)
        if not citations:
            logger.info(f"第 {i + 1} 层无引用文献,停止向后扩展")
            break

        # Forward expansion in the CoI paper uses semantic similarity between the topic and anchor abstract.
        # Here, the hypothesis, which is also the query, is used for semantic reranking.
        reranked_citations = _rerank_by_llm(citations, hypothesis, top_n=10, config=LLM_CONFIG)

        # Find the first ranked candidate that has citing papers.
        best_citation = None

        # At the final level, use the paper with the highest similarity directly.
        is_last_layer = (i == count - 1)

        for candidate in reranked_citations:
            candidate_id = candidate.get('paperId')

            if not candidate.get('paperId'):
                continue

            if candidate_id in seen_paper_ids:
                logger.info(f"  跳过候选论文（paperId已使用）: '{candidate.get('title', 'N/A')[:50]}...'")
                continue

            # Final level: use the candidate with the highest similarity directly.
            if is_last_layer:
                best_citation = candidate
                logger.info(f"  最后一层,直接选择相似度最高的论文: '{candidate.get('title', 'N/A')[:50]}...'")
                break

            logger.info(f"  检查候选论文 '{candidate.get('title', 'N/A')[:50]}...' 是否有引用文献...")

            # Check whether the candidate paper has citing papers.
            candidate_citations = _get_paper_citations(
                candidate_id, headers, base_url, fields, logger,
                delay * 0.5, max_retries
            )

            if candidate_citations and len(candidate_citations) > 0:
                best_citation = candidate
                break
            else:
                print(f"  ✗ 该论文无引用文献，跳过")

            time.sleep(delay * 0.3)

        if best_citation and best_citation.get('paperId'):
            seen_paper_ids.add(best_citation['paperId'])
            chain.append(best_citation)
            current_id = best_citation['paperId']
            logger.info(f"  第 {i + 1} 层已添加: '{best_citation.get('title', 'N/A')[:60]}...'")
        else:
            logger.warning(f"第 {i + 1} 层所有候选论文都无引用文献,停止向后扩展")
            break

        time.sleep(delay)

    return chain


def _build_literature_chain(anchor_paper: Dict, chain_length: int, headers: dict, seen_paper_ids,
                            base_url: str, fields: list, query: str, hypothesis: str,
                            logger, delay: float, max_retries: int,
                            LLM_CONFIG=None) -> tuple[
    List[Dict], int, int]:
    """Build a paper chain centered on the anchor paper."""
    chain = [anchor_paper]
    anchor_id = anchor_paper.get('paperId')
    if not anchor_id:
        logger.error("锚点论文缺少paperId")
        return [], 0, 0

    # Attempt balanced forward and backward expansion.
    forward_count = (chain_length - 1) // 2
    backward_count = chain_length - 1 - forward_count

    forward_chain = _get_reference_chain(anchor_id, forward_count, headers, base_url, seen_paper_ids, fields, query,
                                         hypothesis,
                                         logger, delay, max_retries, LLM_CONFIG=LLM_CONFIG)
    backward_chain = _get_citation_chain(anchor_id, backward_count, headers, base_url, seen_paper_ids, fields, query,
                                         hypothesis,
                                         logger, delay, max_retries, LLM_CONFIG=LLM_CONFIG)

    complete_chain = forward_chain + chain + backward_chain
    logger.info(f"文献链构建完成: 向前{len(forward_chain)}篇 + 锚点1篇 + 向后{len(backward_chain)}篇")
    total_length = len(complete_chain)

    # Log a warning if the chain is too short.
    if total_length < chain_length:
        shortage = chain_length - total_length
        logger.warning(f"警告: 文献链长度不足! 目标{chain_length}篇，实际{total_length}篇，缺少{shortage}篇")

    else:
        logger.info(f"✓ 成功构建完整文献链，达到目标长度")

    return complete_chain, len(forward_chain), len(backward_chain)


def Assign_Chains_to_Experts_unique(expert_agents, query_documents, chain_per_expert, logger):
    """
    Assign paper chains to expert agents so that each expert receives different chains.

    Args:
    - expert_agents: List of expert agents.
    - query_documents: Dictionary mapping keywords to corresponding paper lists (paper chains).
    - chain_per_expert: Number of paper chains assigned to each expert.
    - logger: Logger instance.

    Returns:
    - expert_agents: Updated list of expert agents.
    - all_assigned_chains: Dictionary of all assigned paper chains.
    """

    assigned_chain_keywords = set()

    # Get all available paper chains, represented by the keyword list.
    available_chains = list(query_documents.keys())

    # Filter out empty paper chains.
    valid_chains = []
    for keyword in available_chains:
        if query_documents[keyword]:  # Ensure the paper list is neither empty nor None.
            valid_chains.append(keyword)

    if not valid_chains:
        logger.warning("WARNING: 没有可用的文献链进行分配")
        return expert_agents, {}

    logger.info(f"INFO: 共有{len(valid_chains)}条文献链可分配，每个专家将获得{chain_per_expert}条链")

    # Check whether enough paper chains are available.
    total_chains_needed = len(expert_agents) * chain_per_expert
    if len(valid_chains) < total_chains_needed:
        logger.warning(
            f"WARNING: 文献链数量不足！需要{total_chains_needed}条，但只有{len(valid_chains)}条可用"
        )

    # Maintain the remaining available paper chains globally.
    remaining_chains = valid_chains.copy()

    # Assign paper chains to each expert.
    for expert_idx, expert in enumerate(expert_agents):
        assigned_chains = 0
        expert_chain_info = []

        # Assign the specified number of paper chains to the current expert.
        while assigned_chains < chain_per_expert and remaining_chains:
            # Randomly select a paper chain.
            current_keyword = random.choice(remaining_chains)
            # Remove it from the globally remaining chains.
            remaining_chains.remove(current_keyword)

            assigned_chain_keywords.add(current_keyword)

            documents_in_chain = query_documents[current_keyword]

            # Add the paper-chain title.
            chain_header = f"**Chain of papers related to '{current_keyword}'**\n"
            expert.RLL.append(chain_header)

            # Add every document in the paper chain to the expert's document list.
            for doc in documents_in_chain:
                if isinstance(doc, dict):
                    # For dictionaries, extract only the title and abstract fields.
                    title = doc.get('title', 'No title available')
                    abstract = doc.get('abstract', 'No abstract available')
                    # Format the document content.
                    doc_content = f"Title: {title}\nAbstract: {abstract}\n"
                    expert.RLL.append(doc_content)
                else:
                    # Add strings directly.
                    expert.RLL.append(doc)

            expert_chain_info.append(f"关键词'{current_keyword}': {len(documents_in_chain)}篇文献")
            assigned_chains += 1

        # Log a warning if too few chains remain available.
        if assigned_chains < chain_per_expert:
            logger.warning(
                f"WARNING: 专家{expert_idx + 1}仅分配到{assigned_chains}条文献链（需求{chain_per_expert}条），可用链条不足"
            )

        logger.info(f"INFO: 专家{expert_idx + 1}分配到{assigned_chains}条文献链: {'; '.join(expert_chain_info)}")

    # Build a dictionary containing all assigned paper chains.
    all_assigned_chains = {
        keyword: query_documents[keyword] for keyword in assigned_chain_keywords
    }

    logger.info(f"INFO: 总共分配了 {len(all_assigned_chains)} 条不重复的文献链。")

    return expert_agents, all_assigned_chains


def API_Assign_New_Documents_to_Experts(expert_agents, hypothesis_text, log_dir, logger,
                                        num_chains_to_build=5,
                                        chain_length=6,
                                        chain_per_expert=1,
                                        LLM_CONFIG: Optional[Dict] = None):

    seen_paper_ids = set()
    query_documents = {}  # Store successful queries and their corresponding paper chains.
    tried_queries = set()  # Track attempted queries to prevent duplicates.
    failed_chain_queries = [] # Track failed queries.

    logger.info("========== Starting Multi-Branch Literature Chain Construction (Robust Mode) ==========")
    logger.info(f"Base Hypothesis: {hypothesis_text}...")
    logger.info(f"Target to build {num_chains_to_build} successful branches.")

    # API configuration parameters.
    search_func = SemanticScholar_API_Search_Documents
    base_url = "https://api.semanticscholar.org/graph/v1"
    detail_fields = ['title', 'abstract', 'year']

    academic_headers = {"x-api-key": "XXX"}

    delay = 3.0
    max_retries = 5

    # --- Loop-control variables ---
    max_generation_rounds = 5  # Maximum number of rounds that call the LLM.
    current_round = 0

    while len(query_documents) < num_chains_to_build and current_round < max_generation_rounds:
        current_round += 1
        num_needed = num_chains_to_build - len(query_documents)

        logger.info(f"\n>>> [Round {current_round}/{max_generation_rounds}] 还需要 {num_needed} 条文献链。正在调用 LLM...")

        # 1. Call the LLM to generate or supplement queries.
        # Use the original call_llm_for_queries function.
        new_queries = call_llm_for_queries(hypothesis_text, num_queries=num_needed, config=LLM_CONFIG, avoid_queries=list(tried_queries))

        # 2. Filter out queries that have already been processed.
        valid_new_queries = []
        for q in new_queries:
            q_clean = q.strip()
            if q_clean not in tried_queries:
                valid_new_queries.append(q_clean)
                tried_queries.add(q_clean)
            else:
                logger.info(f"跳过重复 Query: '{q_clean}'")

        if not valid_new_queries:
            logger.warning("本轮未生成新的有效 Query。")
            if current_round >= max_generation_rounds:
                break
            continue

        # 3. Iterate over newly generated queries to build chains.
        for idx, q in enumerate(valid_new_queries):
            # Check whether the requested number has been reached.
            if len(query_documents) >= num_chains_to_build:
                break

            logger.info(f"\n--- Processing Query: '{q}' ---")
            documents = []

            # Step 2a: Find the anchor paper.
            anchor = _find_anchor_paper(
                query=q,
                hypothesis=hypothesis_text,
                search_func=search_func,
                logger=logger,
                delay=delay,
                max_retries=max_retries,
                seen_paper_ids=seen_paper_ids,
                headers=academic_headers,
                base_url=base_url,
                fields=detail_fields,
                LLM_CONFIG=LLM_CONFIG
            )

            # Step 2b: Build the chain.
            if anchor:
                chain, fw, bw = _build_literature_chain(
                    anchor_paper=anchor,
                    chain_length=chain_length,
                    headers=academic_headers,
                    seen_paper_ids=seen_paper_ids,
                    base_url=base_url,
                    fields=detail_fields,
                    query=q,
                    hypothesis=hypothesis_text,
                    logger=logger,
                    delay=delay,
                    max_retries=max_retries,
                    LLM_CONFIG=LLM_CONFIG
                )
                documents = chain
                if len(documents) < chain_length:
                    logger.warning(f"链条长度不足 (实际{len(documents)} < 目标{chain_length})，放弃该 Query。")
                    documents = []  # Clear it so that it is not counted as successful.
            else:
                logger.warning(f"Query '{q}' 无法找到合适的 Anchor paper。")

            # 4. Determine success and save the result.
            if documents and len(documents) >= chain_length:
                query_documents[q] = documents
                logger.info(
                    f"✓ Query '{q}' 成功构建链条! (长度: {len(documents)}, 当前进度: {len(query_documents)}/{num_chains_to_build})")
            else:
                logger.warning(f"✗ Query '{q}' 构建失败，将在后续轮次尝试生成新 Query。失败的Query将被记录。")
                failed_chain_queries.append(q)

    # --- End of loop ---

    # === Perform final supplementation using failed queries ===
    if len(query_documents) < num_chains_to_build:
        logger.warning(f"警告: 复杂文献链构建不足 (现有 {len(query_documents)}/{num_chains_to_build})。")
        logger.info(f"启用补充策略: 复用之前构建链条失败的 {len(failed_chain_queries)} 个 Query 进行直接搜索。")

        # Iterate over previously recorded failed queries.
        for q in failed_chain_queries:
            # Stop once the requested number has been reached.
            if len(query_documents) >= num_chains_to_build:
                break

            logger.info(f"--- Fallback Direct Search (Reusing Query): '{q}' ---")

            # Search for 30 papers.
            pool_size = 30
            papers_pool = search_func(q, pool_size, logger, headers=academic_headers, delay=delay, max_retries=max_retries)

            if papers_pool and len(papers_pool) > 0:
                # 2. Use the LLM to rerank and select the chain_length papers most relevant to hypothesis_text.
                logger.info(f"正在对 {len(papers_pool)} 篇候选论文进行相关性排序...")
                selected_papers = _rerank_by_llm(
                    papers_pool,
                    hypothesis_text,  # Judge relevance with the core hypothesis; this usually works better than the query.
                    top_n=chain_length,
                    config=LLM_CONFIG
                )

                query_documents[q] = selected_papers
                logger.info(f"✓ 补充成功: '{q}' (从 {len(papers_pool)} 篇中优选了 {len(selected_papers)} 篇)")
            else:
                logger.warning(f"✗ 补充搜索也未返回结果: '{q}'")

    if len(query_documents) < num_chains_to_build:
        logger.warning(f"最终警告: 包含补充策略在内，仍只获取了 {len(query_documents)}/{num_chains_to_build} 组文献。")
    else:
        logger.info(f"最终统计: 共获取 {len(query_documents)} 组文献 (包含构建链和补充搜索)。")
    # =======================================

    # 3. Save results while preserving the original logic.
    logger.info("Saving collected documents...")
    chain_folder = os.path.join(log_dir, "chains_of_papers")
    os.makedirs(chain_folder, exist_ok=True)

    def sanitize_filename(name):
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)

    safe_name = sanitize_filename(hypothesis_text[:30])
    chain_docs_path = os.path.join(chain_folder, f"{safe_name}.json")

    try:
        with open(chain_docs_path, 'w', encoding='utf-8') as f:
            json.dump(query_documents, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved to {chain_docs_path}")
    except IOError as e:
        logger.error(f"Save failed: {e}")

    # 4. Assign results to experts while preserving the original logic.
    expert_agents, all_assigned_chains = Assign_Chains_to_Experts_unique(
        expert_agents, query_documents, chain_per_expert, logger
    )

    # Clean the structure by retaining only title and abstract.
    all_assigned_chains = {
        keyword: [
            {
                'title': paper.get('title', 'No Title Available'),
                'abstract': paper.get('abstract', 'No Abstract Available')
            }
            for paper in papers_list
        ]
        for keyword, papers_list in all_assigned_chains.items()
    }

    return expert_agents, all_assigned_chains

