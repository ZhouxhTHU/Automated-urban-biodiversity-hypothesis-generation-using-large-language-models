import json
import random
import glob
import sys, os, re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from agentscope.message import Msg
# from RLL import *
from chain_of_papers import *
from config import CONFIG

from utils import make_api_call_with_retry, ConvertJSON_list, ConvertJSON_dict, enable_token_tracking,\
    collect_paper_ids_from_hypothesis, reconfirm_json_with_llm, load_CLL_json, retrieve_inspirations_by_source

from agent import *
from prompt import *
from log import save_to_txt, save_config_summary, setup_logger
# <<< MODIFIED: Import logging
import logging

# Configure logging.
file_name = "phase2"

# Example of chain of papers
LLM_COP_CONFIG = {
    "API_KEY": "YOUR_LLM_API_KEY",
    "BASE_URL": "https://api..com/v1",
    "MODEL_NAME": "gpt-4o",
    "timeout": 600
}
LLM_COP_CONFIG["API_KEY"] = os.getenv("LLM_COP_API_KEY", "")
LLM_COP_CONFIG["BASE_URL"] = "https://svip-ip.xty.app/v1"
LLM_COP_CONFIG["MODEL_NAME"] = "gemini-3.1-pro-preview-thinking"
LLM_COP_CONFIG["timeout"] = 600

include_reasoning = CONFIG["docs"]["INCLUDE_REASONING"]
token_stats_phase2_name = "token_stats_phase2.json"
token_stats_phase2 = {
    'input_tokens': 0,
    'output_tokens': 0,
    'total_tokens': 0
}

# ==============================================================================
# Inspiration discussion.
# ==============================================================================

def inspiration_discussion(
        experts: List[BiodiversityExpert],
        grand_expert: GrandExpert,
        screener: InspirationScreener,
        current_hypothesis: Dict[str, Any],
        current_inspiration_pool: List[Dict[str, Any]],
        max_rounds: int,
        Curated_Literature_Library: str,
        all_assigned_chains: Dict[str, List],
        all_agents: List,
        logger: logging.Logger,
        Phase_2_config: Dict[str, Any],
        USE_API_FOR_NEW_PAPERS: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Organize one round of expert discussion to deepen inspirations and prepare for hypothesis refinement.
    """
    logger.info("\n" + "=" * 20 + " 开始假设深化与灵感讨论循环 " + "=" * 20)
    # Initialize variables.
    discussion_history = []
    # Clear expert discussion history to avoid input-token limits, while retaining leading-expert summaries.
    grand_expert_summary_history = []

    # --- Conduct multiple rounds of expert discussion ---
    for round_num in range(1, max_rounds + 1):
        logger.info(f"\n--- 第 {round_num}/{max_rounds} 轮讨论开始 ---\n")
        current_round_inspirations = []

        # Each Expert speaks in sequence.
        for expert in experts:
            # The expert can see all earlier messages from the current round.
            # Initial documents: hypothesis-related inspirations assigned to the expert.
            basic_documents_text_list = [expert.document_map[str(doc_id)] for doc_id in expert.assigned_doc_ids if
                                         str(doc_id) in expert.document_map]
            basic_documents_text = "---START OF DOCUMENTS---\n" + "\n---\n".join(
                basic_documents_text_list) + "\n---END OF DOCUMENTS---\n\n"

            # Used for the API-backed library.
            if USE_API_FOR_NEW_PAPERS:
                Retrieved_Literature_Library = expert.RLL
            # Used for the fixed library.
            else:
                Retrieved_Literature_Library = [expert.document_map[str(doc_id)] for doc_id in expert.new_documents_ids if
                                           str(doc_id) in expert.document_map]

            expert_prompt = Msg(
                name="system",
                # We reuse this prompt,
                # treating the current inspiration library as the initial inspiration,
                # and do not discuss it for the time being.
                content=get_Phase_two_Expert_Inspiration_prompt(current_inspiration_pool=current_inspiration_pool,
                                                                current_hypothesis=current_hypothesis,
                                                                curated_literature_library=basic_documents_text,
                                                                curated_literature_library_inspiration_pool=Curated_literature_library_inspiration_pool,
                                                                new_documents_text=Retrieved_Literature_Library,
                                                                discussion_history=discussion_history,
                                                                min_num_inspirations=Phase_2_config[
                                                                    "min_num_inspirations_discussion"],
                                                                max_num_inspirations=Phase_2_config[
                                                                    "max_num_inspirations_discussion"]),
                role="user"
            )

            expert_response = make_api_call_with_retry(expert, expert_prompt, all_agents, logger,
                                                       Expert_MODEL_CONFIG)

            parsed_content = ConvertJSON_list(expert_response)

            # --- Secondary JSON validation when parsing fails ---
            if not parsed_content:
                logger.warning(f"初次解析 {expert.name} 的响应失败。尝试 LLM 二次校验...")
                # Call the secondary validation function.
                parsed_content = reconfirm_json_with_llm(
                    expert_response=expert_response,
                    expert_agent=expert,
                    all_agents=all_agents,
                    logger=logger,
                    LLM_CONFIG=Expert_MODEL_CONFIG  # Use the expert model configuration for secondary validation.
                )

                if not parsed_content:
                    logger.error(f"LLM 二次校验后，{expert.name} 的响应仍无法解析为有效的列表，本轮将跳过其新灵感。")
            # --- End of secondary JSON validation ---

            # Record the message and new inspirations.
            discussion_history.append(parsed_content)
            logger.info(f"{expert.name}发言如下：\n {parsed_content}\n")
            # Collect all inspirations produced in this round.
            # Iterate over the parsed list.
            for item in parsed_content:
                # Iterate over all keys in the dictionary.
                for key in item.keys():
                    # Select keys whose names contain "inspiration".
                    if 'inspiration' in key.lower():
                        new_insp = item[key]
                        current_round_inspirations.append(new_insp)
            logger.info(f"INFO: 目前共计产生了 {len(current_round_inspirations)} 条新灵感。\n")

        # GrandExpert provides a summary and guidance.
        grand_expert_prompt = Msg(
            name="system",
            content=get_Phase_two_GrandExpert_Summary_prompt(curated_literature_library=Curated_Literature_Library,
                                                             curated_literature_library_inspiration_pool=Curated_literature_library_inspiration_pool,
                                                             retrieved_literature_library=all_assigned_chains,
                                                             current_hypothesis=current_hypothesis,
                                                             discussion_history=discussion_history),
            role="user"
        )

        # # Save the prompt here to inspect token usage.
        # with open('Phase_two_GrandExpert_Summary.txt', 'w', encoding='utf-8') as f:
        #     f.write(get_Phase_two_GrandExpert_Summary_prompt(curated_literature_library=Curated_Literature_Library,
        #                                                      curated_literature_library_inspiration_pool=Curated_literature_library_inspiration_pool,
        #                                                      retrieved_literature_library=all_assigned_chains,
        #                                                      current_hypothesis=current_hypothesis,
        #                                                      discussion_history=discussion_history))

        grand_expert_response = make_api_call_with_retry(grand_expert, grand_expert_prompt, all_agents, logger,
                                                         GrandExpert_MODEL_CONFIG)

        ## ===== Parse here: discard earlier guidance but retain the previous discussion history =====
        # 1. Parse the new round's summary and guidance.
        grand_expert_guidance = ConvertJSON_dict(grand_expert_response)

        # 2. Process the final history item before adding a new one.
        # Check that history is not empty, meaning this is not the first round.
        if grand_expert_summary_history:
            # Get the final list item, which is the previous round's summary.
            last_summary = grand_expert_summary_history[-1]

            # Remove "Guidance" because this item contains both guidance and summary keys.
            # .pop() returns the default value None instead of raising an error if the key is absent.
            if isinstance(last_summary, dict):
                last_summary.pop("Guidance", None)

        # 3. Add the new round's complete summary, including "Guidance", to history.
        grand_expert_summary_history.append(grand_expert_guidance)

        # The leading expert has already summarized this round's discussion.
        discussion_history.clear()
        discussion_history = grand_expert_summary_history

        logger.info(f"大专家总结如下：\n{grand_expert_response.content}\n")
        logger.info(f"INFO: {grand_expert.name} 已完成本轮总结。")

        # The Screener integrates new inspirations from this round.
        if current_round_inspirations:
            # If the papers yield inspirations, first add this round's inspirations to the pool.
            for new_inspiration in current_round_inspirations:
                current_inspiration_pool.append(new_inspiration)
            current_hypothesis_text = f"Based on the hypothesis:**{current_hypothesis}**, "
            screener_prompt = Msg(
                name="system",
                content=get_screener_update_prompt(current_inspiration_pool=current_inspiration_pool,
                                                   current_hypothesis=current_hypothesis_text,
                                                   min_pool_num=Phase_2_config["min_PoolSize_discussion"],
                                                   max_pool_num=Phase_2_config["max_PoolSize_discussion"]),
                role="user"
            )

            # The Screener output should be the updated inspiration pool.
            logger.info(f"INFO: {screener.name} 正在整合 {len(current_round_inspirations)} 条新灵感... \n")

            screener_response = make_api_call_with_retry(screener, screener_prompt, all_agents, logger,
                                                         Screener_MODEL_CONFIG)
            current_inspiration_pool = ConvertJSON_list(screener_response)
            logger.info(f"新的灵感库如下:\n {current_inspiration_pool} \n")
            logger.info(f"INFO: 灵感库已更新，当前包含 {len(current_inspiration_pool)} 条灵感。")
        else:
            logger.info(f"INFO: 本轮没有产生新的灵感，灵感库未更新。 \n")

    logger.info("\n" + "=" * 20 + " 灵感讨论环节结束 " + "=" * 20)
    return current_inspiration_pool, discussion_history

# ==============================================================================
# Complete Phase 2 workflow.
def main_phase_two(config_data, phase1_hypotheses_path, phase1_pool_path, checkpoint_inspiration_path=None,
                   custom_checkpoint_path=None):
    """
    Entry point for running the Phase 2 experiment.

    Args:
        config_data (dict): The configuration dictionary for this run.

        phase1_hypotheses_path (str): Path to the ranked_hypotheses.json from phase 1.
        phase1_pool_path (str): Path to the inspirations_pool.json from phase 1.
    """
    enable_token_tracking(token_stats_phase2, verbose=False)
    phase1_parent_dir = os.path.dirname(phase1_hypotheses_path)
    # 1. Configure the logger.
    model_name_for_log = config_data["Expert_MODEL_CONFIG"].get("model_name", "unknown_model").replace("/", "_")
    logger, log_dir = setup_logger(
        file_name=file_name,  # "phase2"
        model_name=model_name_for_log,
        parent_dir=phase1_parent_dir
    )

    logger.info(f"--- Starting Phase 2 ---")
    logger.info(f"Logging to directory: {log_dir}")

    # 2. Load the Phase 2 configuration.
    Phase_2_config = config_data["Phase_2_config"]
    # Save the configuration.
    save_config_summary(log_dir, Phase_2_config, mode='a')

    # --- Configuration parameters ---
    USE_API_FOR_NEW_PAPERS = Phase_2_config["USE_API_FOR_NEW_PAPERS"]
    max_refinement_loops = Phase_2_config["max_refinement_loops"]
    keyWords_num_papers = Phase_2_config["keyWords_num_papers"]
    iteration_GrandExpert_Critic = Phase_2_config["iteration_GrandExpert_Critic"]
    max_rounds_discussion = Phase_2_config["max_rounds_discussion"]

    # --- Load files from Phase 1 ---
    logger.info("正在加载假设和灵感库文件...")
    try:
        with open(phase1_hypotheses_path, 'r', encoding='utf-8') as f:
            all_hypotheses = json.load(f)

        with open(phase1_pool_path, 'r', encoding='utf-8') as f:
            all_inspiration_pool = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"无法加载阶段1的文件: {e}")
        raise

    logger.info(f"成功加载 {len(all_hypotheses)} 个假设和 {len(all_inspiration_pool)} 条灵感\n")

    # --- Process each hypothesis in turn ---
    final_results = []

    # Set the number of hypotheses to refine here.
    for hypothesis_idx, initial_hypothesis in enumerate(all_hypotheses[0:15], 1):
        # Initialize the retrieved literature library.
        all_assigned_chains = []
        # --- Initialize agents ---
        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始处理第 {hypothesis_idx}/{len(all_hypotheses)} 个假设 ")
        logger.info(f"{'=' * 60}\n")

        # Reinitialize agents each time to avoid duplicate entries in the CLL.
        expert_agents, inspiration_screener, grand_expert, critic, all_agents = create_all_agents(
            config_data,
            logger,
            num_of_experts=7,
            phase="phase2")

        # Collect paper IDs related to the hypothesis.
        collected_paper_ids = collect_paper_ids_from_hypothesis(
            initial_hypothesis,
            all_inspiration_pool,
            logger
        )

        # Assign these paper IDs to each expert, starting with papers related to the hypothesis.
        logger.info(f"为 {len(expert_agents)} 个专家分配CLL论文...")
        for expert in expert_agents:
            expert.assigned_doc_ids = collected_paper_ids.copy()
            logger.info(f"{expert.name} 被分配了 {len(expert.assigned_doc_ids)} 篇论文")

        # Clear each expert's new-document cache.
        for expert in expert_agents:
            expert.RLL.clear()
            expert.new_documents_ids.clear()

        # Prepare the result output location.
        save_config_summary(log_dir, config_data)
        current_prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.py')
        save_to_txt("prompt.txt", log_dir, current_prompt_path, logger)

        CLL_PATH = grand_expert.CLL_path
        Curated_Literature_Library = load_CLL_json(CLL_PATH, logger)

        logger.info(f"当前假设:\n{json.dumps(initial_hypothesis, indent=2, ensure_ascii=False)}\n")

        # Retrieve the corresponding inspirations by Source.
        source_data = initial_hypothesis.get("Source_from_InspirationPool", "")
        initial_inspiration_pool = retrieve_inspirations_by_source(source_data, all_inspiration_pool, logger)

        if not initial_inspiration_pool:
            logger.warning(f"警告: 未能为假设 {hypothesis_idx} 检索到任何灵感，将使用空灵感池继续处理")
            initial_inspiration_pool = []

        logger.info(f"检索到的灵感库:\n{json.dumps(initial_inspiration_pool, indent=2, ensure_ascii=False)}\n")

        current_inspiration_pool = initial_inspiration_pool

        # Assign new literature for the current hypothesis.
        hypothesis_text = initial_hypothesis.get("Initial_Hypothesis") or initial_hypothesis.get(
            "hypothesis") or initial_hypothesis.get("Hypothesis") or initial_hypothesis.get("Initial Hypothesis")

        custom_checkpoint = False

        if custom_checkpoint_path:
            checkpoint_pattern = f"checkpoint_inspiration_hypo{hypothesis_idx}_loop*.json"
            checkpoint_files = glob.glob(os.path.join(custom_checkpoint_path, checkpoint_pattern))

            if checkpoint_files:
                # If a checkpoint exists, load it directly and skip paper-chain construction.
                custom_checkpoint = True
                logger.info(f"检测到已有灵感池checkpoint，跳过文献链构建")

        if custom_checkpoint == True:
            custom_checkpoint = False

        else:
            chain_length = Phase_2_config["chain_length"]
            logger.info(f"构建长度为{chain_length}的文献链...")

            # Build paper chains.
            expert_agents, all_assigned_chains = API_Assign_New_Documents_to_Experts(expert_agents, hypothesis_text,
                                                                                     log_dir, logger,
                                                                                     num_chains_to_build=
                                                                                     Phase_2_config[
                                                                                         "num_chains_to_build"],
                                                                                     chain_length=Phase_2_config[
                                                                                         "chain_length"],
                                                                                     chain_per_expert=
                                                                                     Phase_2_config[
                                                                                         "chain_per_expert"],
                                                                                     LLM_CONFIG=LLM_COP_CONFIG
                                                                                     )

            # expert_agents, all_assigned_chains = API_Assign_New_Documents_to_Experts(expert_agents, hypothesis_text,
            #                                                                          log_dir, logger,
            #                                                                          LLM_CONFIG=LLM_COP_CONFIG
            #                                                                          )

        # Reset loop variables.
        current_critic_loop = 0
        current_hypothesis = {
            "hypothesis_id": initial_hypothesis.get("hypothesis_id"),
            "hypothesis": initial_hypothesis.get("Initial_Hypothesis") or initial_hypothesis.get(
                "hypothesis") or initial_hypothesis.get("Hypothesis") or initial_hypothesis.get(
                "Initial Hypothesis"),
            "Reasoning": initial_hypothesis.get("reasoning") or initial_hypothesis.get("Reasoning", "")
        }

        # --- Start the full workflow loop ---
        while current_critic_loop < max_refinement_loops:
            current_critic_loop += 1
            logger.info(
                f"\n{'#' * 30} 假设 {hypothesis_idx} - 第 {current_critic_loop}/{max_refinement_loops} 轮优化 {'#' * 30}")

            checkpoint_inspiration = False

            if checkpoint_inspiration_path and os.path.exists(checkpoint_inspiration_path):
                filename = os.path.basename(checkpoint_inspiration_path)
                match = re.search(r'checkpoint_inspiration_hypo(\d+)_loop(\d+)\.json', filename)
                if match:
                    file_hypo_idx = int(match.group(1))
                    file_loop_idx = int(match.group(2))
                    # Check for a match with the current hypothesis_idx and current_critic_loop.
                    if file_hypo_idx == hypothesis_idx and file_loop_idx == current_critic_loop:
                        checkpoint_inspiration = True

            # Check whether an inspiration-pool checkpoint exists.
            if checkpoint_inspiration:
                logger.info(f"检测到已有灵感池checkpoint，直接加载：{checkpoint_inspiration_path}")

                with open(checkpoint_inspiration_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)

                # Load the inspiration pool.
                refined_pool = checkpoint_data.get("inspiration_pool", [])
                logger.info(f"已从checkpoint加载 {len(refined_pool)} 条灵感")

                # Load all_assigned_chains.
                all_assigned_chains = checkpoint_data.get("all_assigned_chains", {})
                logger.info(f"已加载 all_assigned_chains: {len(all_assigned_chains)} 条链")

                # Load and restore each expert's paper chains in order.
                experts_chains_data = checkpoint_data.get("experts_chains", [])
                for i, expert in enumerate(expert_agents):
                    if i < len(experts_chains_data):
                        expert_data = experts_chains_data[i]
                        expert.RLL = expert_data.get("Retrieved_Literature_Library", [])
                        logger.info(f"已为 {expert.name} 恢复文献链")
                    else:
                        logger.warning(f"Checkpoint中没有足够的expert数据，{expert.name} 未恢复文献链")

            else:
                logger.info("未检测到灵感池checkpoint，开始运行灵感讨论...")

                refined_pool, history = inspiration_discussion(
                    experts=expert_agents,
                    grand_expert=grand_expert,
                    screener=inspiration_screener,
                    current_hypothesis=current_hypothesis,
                    current_inspiration_pool=current_inspiration_pool,
                    max_rounds=max_rounds_discussion,
                    Curated_Literature_Library=Curated_Literature_Library,
                    all_assigned_chains=all_assigned_chains,
                    all_agents=all_agents,
                    logger=logger,
                    Phase_2_config=Phase_2_config,
                    USE_API_FOR_NEW_PAPERS=USE_API_FOR_NEW_PAPERS
                )

                logger.info(f"阶段1（灵感讨论）完成")

                current_inspiration_pool = refined_pool

                # Save a checkpoint containing the inspiration pool, paper chains, and expert assignments.
                checkpoint_inspiration_path = os.path.join(
                    log_dir,
                    f"checkpoint_inspiration_hypo{hypothesis_idx}_loop{current_critic_loop}.json"
                )

                # Collect paper-chain information for each expert.
                experts_chains_data = []
                for expert in expert_agents:
                    expert_data = {
                        "Retrieved_Literature_Library": expert.RLL
                    }
                    experts_chains_data.append(expert_data)

                # Build the complete checkpoint data.
                checkpoint_data = {
                    "inspiration_pool": refined_pool,
                    "all_assigned_chains": all_assigned_chains,
                    "experts_chains": experts_chains_data,
                    "hypothesis_idx": hypothesis_idx,
                    "current_critic_loop": current_critic_loop
                }

                with open(checkpoint_inspiration_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                logger.info(f"灵感池checkpoint已保存至：{checkpoint_inspiration_path}")
                logger.info(f"已保存 {len(experts_chains_data)} 个expert的文献链信息")

                # ################################################

            # === Stage 2: GrandExpert updates the hypothesis ===
            logger.info("\n--- GrandExpert 正在根据新的灵感库更新假设... ---")
            critic_info = " "
            grand_expert_prompt = Msg(
                name="system",
                content=get_Phase_two_GrandExpert_Refine_Hypothsis_multiple_branch_prompt(curated_literature_library=Curated_Literature_Library,
                                                                                          curated_literature_library_inspiration_pool=Curated_literature_library_inspiration_pool,
                                                                                          retrieved_literature_library=all_assigned_chains,
                                                                                          current_hypothesis=current_hypothesis,
                                                                                          current_inspiration_pool=refined_pool,
                                                                                          critic_info=critic_info),
                role="user"
            )

            with open('GrandExpert_Refine_Hypothsis_multiple300.txt', 'w', encoding='utf-8') as f:
                f.write(get_Phase_two_GrandExpert_Refine_Hypothsis_multiple_branch_prompt(curated_literature_library=Curated_Literature_Library,
                                                                                          curated_literature_library_inspiration_pool=Curated_literature_library_inspiration_pool,
                                                                                          retrieved_literature_library=all_assigned_chains,
                                                                                          current_hypothesis=current_hypothesis,
                                                                                          current_inspiration_pool=refined_pool,
                                                                                          critic_info=critic_info))

            grand_expert_response = make_api_call_with_retry(grand_expert, grand_expert_prompt, all_agents, logger,
                                                             GrandExpert_MODEL_CONFIG)
            alternative_hypothesis = ConvertJSON_list(grand_expert_response)

            logger.info(f"INFO: 更新的多个假设: \n'{alternative_hypothesis}'\n")

            all_refined_hypotheses = []

            # Iterate over each alternative hypothesis.
            for hypo_idx, single_hypothesis in enumerate(alternative_hypothesis, 1):
                logger.info(f"\n处理第 {hypo_idx}/{len(alternative_hypothesis)} 个假设分支")
                current_hypothesis = single_hypothesis  # Assign the hypothesis being processed to current_hypothesis.

                # === Stage 3: Critic evaluation ===
                logger.info("\n--- Critic 评价 ---\n")

                for debate_turn in range(iteration_GrandExpert_Critic):
                    critic_prompt = Msg(
                        name="system",
                        content=get_Phase_two_Critic_prompt(hypothesis=current_hypothesis),
                        role="user"
                    )

                    critic_expert_response = make_api_call_with_retry(critic, critic_prompt, all_agents,
                                                                      logger, Critic_MODEL_CONFIG)
                    critic_info = ConvertJSON_dict(critic_expert_response)

                    logger.info(
                        f"评论家评价:\n{json.dumps(critic_info, indent=2, ensure_ascii=False)}\n")  # Use json.dumps for more readable output.

                    grand_expert_prompt = Msg(
                        name="system",
                        content=get_Phase_two_GrandExpert_Refine_Hypothsis_prompt(curated_literature_library=Curated_Literature_Library,
                                                                                  curated_literature_library_inspiration_pool=Curated_literature_library_inspiration_pool,
                                                                                  retrieved_literature_library=all_assigned_chains,
                                                                                  current_hypothesis=current_hypothesis,
                                                                                  current_inspiration_pool=refined_pool,
                                                                                  critic_info=critic_info),
                        role="user"
                    )


                    grand_expert_response = make_api_call_with_retry(grand_expert, grand_expert_prompt, all_agents,
                                                                     logger,
                                                                     GrandExpert_MODEL_CONFIG)
                    refined_hypothesis = ConvertJSON_dict(grand_expert_response)

                    logger.info(f"大专家更新假设\n")
                    logger.info(f"INFO: 更新前的假设: \n'{current_hypothesis}'\n")
                    current_hypothesis = refined_hypothesis
                    logger.info(f"INFO: 更新后的假设: \n'{current_hypothesis}'\n")

                all_refined_hypotheses.append({
                    "branch_index": hypo_idx,
                    "hypothesis": current_hypothesis,
                })

            # Adjust this result location as needed, either inside or outside the loop.
            # --- Save the final result for the current hypothesis ---
            result = {
                "initial_hypothesis": initial_hypothesis,
                "all_hypothesis_branches": all_refined_hypotheses,
                "final_inspiration_pool": refined_pool,
                "iterations_used": current_critic_loop
            }
            final_results.append(result)

        # --- Save all results ---
        output_path = os.path.join(log_dir, "final_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Import and merge token statistics from chain_of_papers.
    from chain_of_papers import chain_of_papers_token_stats

    token_stats_phase2['input_tokens'] += chain_of_papers_token_stats['input_tokens']
    token_stats_phase2['output_tokens'] += chain_of_papers_token_stats['output_tokens']
    token_stats_phase2['total_tokens'] += chain_of_papers_token_stats['total_tokens']

    # Save token counts.
    token_stats_phase2_file = os.path.join(log_dir, token_stats_phase2_name)

    with open(token_stats_phase2_file, 'w', encoding='utf-8') as f:
        json.dump(token_stats_phase2, f, indent=4, ensure_ascii=False)

    logger.info(f"\n所有假设处理完成！结果已保存至: {output_path}")
    logger.info(f"共处理 {len(final_results)} 个假设")
# ==============================================================================

if __name__ == '__main__':
    HYPO_PATH = "45Hypo/gemini-3.1-pro-preview-thinking_20260419_161728_window15_screener100/Checkpoint_gemini-3.1-pro-preview-thinking_20260419_161728_window15_screener100_stage2_20260420_230016/initial_hypothesis.json"
    POOL_PATH = "45Hypo/gemini-3.1-pro-preview-thinking_20260419_161728_window15_screener100/inspirations_pool.json"

    # Read input files.
    with open(POOL_PATH, 'r', encoding='utf-8') as f:
        Curated_literature_library_inspiration_pool = json.load(f)

    # if include_reasoning:
    #     keys_to_keep = ["inspiration_id", "source", "content", "reasoning"]
    # else:
    #     keys_to_keep = ["inspiration_id", "source", "content"]
    #
    # Curated_literature_library_inspiration_pool = [
    #     {key: item[key] for key in keys_to_keep}
    #     for item in Curated_literature_library_inspiration_pool
    # ]

    # Set the output directory location.
    phase1_dir = os.path.dirname(HYPO_PATH)
    phase2_base_dir = os.path.join(phase1_dir, file_name)

    # Define checkpoint_inspiration_path; both checkpoint paths must be provided together.
    checkpoint_inspiration_path = None
    custom_checkpoint_path = None

    print("Running Phase2.py...")

    if not os.path.exists(HYPO_PATH) or not os.path.exists(POOL_PATH):
        print(f"ERROR: Cannot run test. Required input files not found:")
        print(f"- {HYPO_PATH}")
        print(f"- {POOL_PATH}")
        print("Please update HYPO_PATH and POOL_PATH in Novelty_Check.py to point to valid files.")
    else:
        # 3. Call the new main function
        main_phase_two(
            CONFIG,  # Use the default config
            HYPO_PATH,
            POOL_PATH,
            checkpoint_inspiration_path=checkpoint_inspiration_path,
            custom_checkpoint_path=custom_checkpoint_path
        )
