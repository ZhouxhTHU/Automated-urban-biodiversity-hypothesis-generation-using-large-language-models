import logging
import datetime
from typing import Tuple
from sklearn_extra.cluster import KMedoids
from collections import defaultdict
import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
from agent import *
# Import required libraries and modules.
# Ensure that AgentScope and its dependencies are correctly installed and configured before running.
from agentscope.message import Msg
from utils import make_api_call_with_retry, load_CLL_json, ConvertJSON_list, ConvertJSON_dict, \
    transform_inspirations, reindex_inspiration_ids, enable_token_tracking, reciprocal_rank_fusion, evaluate_clustering_quality
from log import setup_logger, save_config_summary, save_to_txt
from prompt import *

# --- Basic Logger ---
""" This is used to distinguish which py file is running in which folder """
file_name = "generate_pre_hypo"
model_name_for_log = Expert_MODEL_CONFIG.get("model_name", "unknown_model").replace("/", "_")
logger, log_dir = setup_logger(file_name=file_name, model_name=model_name_for_log)

# ------------------------------------ Configuration and Initialization ------------------------------------
Results_PATH = '45Hypo'
all_windows_inspirations_name = 'all_windows_inspirations.json'
all_inspirations_name = 'all_inspirations.json'
inspirations_pool_name = 'inspirations_pool.json'
initial_hypothesis_name = 'initial_hypothesis.json'
scored_hypotheses_name = 'scored_hypotheses.json'
llms_ranked_name = "LLMs_ranked_hypotheses.json"
silicon_jury_ranked_hypotheses_name = "silicon_jury_ranked_hypotheses.json"
clustering_details_name = "clustering_details.json"

token_stats_phase1_name = "token_stats_phase1.json"
token_stats_phase1 = {
    'input_tokens': 0,
    'output_tokens': 0,
    'total_tokens': 0
}


def get_checkpoint_stage_and_dir(checkpoint_all_inspirations, checkpoint_inspiration_pool,
                                 checkpoint_initial_hypothesis):
    """
    Determine the checkpoint stage from which to resume and its directory.
    Returns: (stage_number, checkpoint_dir)
    stage_number: 1, 2, or 3 indicates the first stage with a checkpoint.
    checkpoint_dir: directory containing the checkpoint file.
    """
    """Check stages in descending order from 3 to 1."""
    if checkpoint_initial_hypothesis and os.path.exists(checkpoint_initial_hypothesis):
        checkpoint_dir = os.path.dirname(checkpoint_initial_hypothesis)
        return 3, checkpoint_dir
    elif checkpoint_inspiration_pool and os.path.exists(checkpoint_inspiration_pool):
        checkpoint_dir = os.path.dirname(checkpoint_inspiration_pool)
        return 2, checkpoint_dir
    elif checkpoint_all_inspirations and os.path.exists(checkpoint_all_inspirations):
        checkpoint_dir = os.path.dirname(checkpoint_all_inspirations)
        return 1, checkpoint_dir
    return None, None


def phase_one_generate_initial_hypothesis(
        CLL_PATH: str,
        expert_agents: List[BiodiversityExpert],
        inspiration_screener: InspirationScreener,
        grand_expert: GrandExpert,
        all_agents: list,
        sliding_window_size: int,
        screener_window_size: int,
        max_inspirations_window: int,
        min_inspirations_window: int,
        logger: logging.Logger,
        save_dir: str = "phase1_results",
        checkpoint_all_inspirations: str = None,
        checkpoint_inspiration_pool: str = None,
        checkpoint_initial_hypothesis: str = None
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Phase 1-4: Complete workflow from inspiration generation to hypothesis scoring
    Returns: (initial_hypothesis, inspiration_pool, initial_history, ranked_hypotheses)
    """
    logger.info("\n" + "=" * 40)
    logger.info(" " * 8 + "PHASE 1-4: COMPLETE WORKFLOW")
    logger.info("=" * 40 + "\n")

    # ========== Stage 1: Generate inspirations ==========
    all_inspirations_file = os.path.join(save_dir, all_inspirations_name)

    if checkpoint_all_inspirations and os.path.exists(checkpoint_all_inspirations):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CHECKPOINT: Found existing all_inspirations file: {checkpoint_all_inspirations}")
        logger.info(f"Skipping Stage 1 (Inspiration Generation)")
        logger.info(f"{'=' * 60}\n")
        all_inspirations_file = checkpoint_all_inspirations
        logger.info(f"Loading inspirations from checkpoint: {all_inspirations_file}")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: INSPIRATION GENERATION")
        logger.info("=" * 60 + "\n")

        expert = expert_agents[0]
        all_doc_ids = expert.assigned_doc_ids.copy()
        random.shuffle(all_doc_ids)
        total_docs = len(all_doc_ids)

        logger.info(f"Starting sliding window processing...")
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"Window size: {sliding_window_size}")
        logger.info(f"Number of windows: {(total_docs + sliding_window_size - 1) // sliding_window_size}")

        all_window_inspirations = []

        for window_start in range(0, total_docs, sliding_window_size):
            window_end = min(window_start + sliding_window_size, total_docs)
            window_doc_ids = all_doc_ids[window_start:window_end]
            window_num = (window_start // sliding_window_size) + 1
            total_windows = (total_docs + sliding_window_size - 1) // sliding_window_size

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing Window {window_num}/{total_windows}")
            logger.info(f"Documents {window_start + 1} to {window_end} (total: {len(window_doc_ids)})")
            logger.info(f"{'=' * 60}")

        # documents_text_list contains the text extracted for this window.
            documents_text_list = [expert.document_map[doc_id] for doc_id in window_doc_ids
                                   if doc_id in expert.document_map]
            documents_text = "---START OF DOCUMENTS---\n" + "\n---\n".join(
                documents_text_list) + "\n---END OF DOCUMENTS---\n\n"

            # content=Phase_one_Expert_Window_prompt(documents_text,min_inspirations_window,max_inspirations_window),

            inspiration_prompt = Msg(
                name="system",
                content=Phase_one_Expert_Window_prompt(documents_text),
                role="user"
            )

            response = make_api_call_with_retry(
                expert, inspiration_prompt,
                all_agents_to_update=all_agents,
                logger=logger,
                model_config=Expert_MODEL_CONFIG
            )


            window_inspiration = {
                "window_number": window_num,
                "doc_range": f"{window_start + 1}-{window_end}",
                "doc_ids": window_doc_ids,
                "content": response.content
            }
            all_window_inspirations.append(window_inspiration)

            logger.info(f"[Window {window_num} Inspirations]:\n{response.content}\n")

        all_windows_file = os.path.join(save_dir, all_windows_inspirations_name)
        with open(all_windows_file, 'w', encoding='utf-8') as f:
            json.dump(all_window_inspirations, f, indent=4, ensure_ascii=False)
        logger.info(f"\nSaved all windows inspirations to: {all_windows_file}")

        # all_window_inspirations stores inspirations separately for each window.
        # transform_inspirations merges and saves inspirations without window grouping.
        transform_inspirations(all_windows_file, all_inspirations_file)
        logger.info(f"Stage 1 Complete: All inspirations saved to {all_inspirations_file}")

    # ========== Stage 2: Screen and merge with the Screener ==========
    screened_file = os.path.join(save_dir, inspirations_pool_name)

    if checkpoint_inspiration_pool and os.path.exists(checkpoint_inspiration_pool):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CHECKPOINT: Found existing inspiration_pool file: {checkpoint_inspiration_pool}")
        logger.info(f"Skipping Stage 2 (Inspiration Screening)")
        logger.info(f"{'=' * 60}\n")

        with open(checkpoint_inspiration_pool, 'r', encoding='utf-8') as f:
            inspiration_pool = json.load(f)
        logger.info(f"Loaded {len(inspiration_pool)} inspirations from checkpoint")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: INSPIRATION SCREENING")
        logger.info("=" * 60 + "\n")

        with open(all_inspirations_file, 'r', encoding='utf-8') as f:
            all_inspirations = json.load(f)

        total_inspirations = len(all_inspirations)
        logger.info(f"Total inspirations to screen: {total_inspirations}")
        logger.info(f"Screener window size: {screener_window_size}")

        # Collect screened results from all windows.
        screened_inspirations_all = []

        for window_start in range(0, total_inspirations, screener_window_size):
            window_end = min(window_start + screener_window_size, total_inspirations)
            window_inspirations = all_inspirations[window_start:window_end]
            window_num = (window_start // screener_window_size) + 1

            # Skip the current window when it contains no content.
            if not window_inspirations:
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Screener Processing Window {window_num}")
            logger.info(f"Inspirations {window_start + 1} to {window_end} (total: {len(window_inspirations)})")
            logger.info(f"{'=' * 60}")

            # Process inspirations from only the current window.
            raw_inspirations_context = json.dumps(window_inspirations, indent=2, ensure_ascii=False)

            # Base the prompt's minimum and maximum counts on the current window size.
            screener_prompt = Msg(
                name="system",
                content=get_screener_update_prompt(
                    current_inspiration_pool=raw_inspirations_context,
                    current_hypothesis="",
                    min_pool_num=len(window_inspirations) * 3/4,
                    max_pool_num=len(window_inspirations)
                ),
                role="user"
            )

            screener_response = make_api_call_with_retry(
                inspiration_screener, screener_prompt,
                all_agents, logger, GrandExpert_MODEL_CONFIG
            )

            try:
            # Obtain the screened results for the current window.
                screened_window_result = ConvertJSON_list(screener_response)
            # Merge the current window's results into the aggregate list.
                screened_inspirations_all.extend(screened_window_result)
                logger.info(
                    f"Screener window {window_num} processed. Added {len(screened_window_result)} inspirations.")
                logger.info(f"Total inspirations in pool so far: {len(screened_inspirations_all)}")
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Failed to parse JSON from InspirationScreener window {window_num}: {e}")
                logger.warning(f"Skipping this window due to error. No inspirations were added from this batch.")

        # After all batches are processed, assign the aggregate results to inspiration_pool.
        inspiration_pool = screened_inspirations_all

        with open(screened_file, 'w', encoding='utf-8') as f:
            json.dump(inspiration_pool, f, indent=4, ensure_ascii=False)
        reindex_inspiration_ids(screened_file,screened_file)
        logger.info(
            f"Stage 2 Complete: Final inspiration pool with {len(inspiration_pool)} items saved to {screened_file}")

    # ========== Stage 3: Generate initial hypotheses with GrandExpert ==========
    save_initial_hypothesis_path = os.path.join(save_dir, initial_hypothesis_name)

    if checkpoint_initial_hypothesis and os.path.exists(checkpoint_initial_hypothesis):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CHECKPOINT: Found existing initial_hypothesis file: {checkpoint_initial_hypothesis}")
        logger.info(f"Skipping Stage 3 (Hypothesis Generation)")
        logger.info(f"{'=' * 60}\n")

        with open(checkpoint_initial_hypothesis, 'r', encoding='utf-8') as f:
            initial_hypothesis = json.load(f)
        logger.info(f"Loaded {len(initial_hypothesis)} hypotheses from checkpoint")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: HYPOTHESIS GENERATION")
        logger.info("=" * 60 + "\n")

        # # Determine whether to include the reasoning section here.
        # if include_reasoning:
        #     keys_to_keep = ["inspiration_id", "source", "content", "reasoning"]
        # else:
        #     keys_to_keep = ["inspiration_id", "source", "content"]
        # # It is named new_inspiration_pool because reasoning inclusion is configurable.
        # new_inspiration_pool = [
        #     {key: item[key] for key in keys_to_keep}
        #     for item in inspiration_pool
        # ]

        new_inspiration_pool = inspiration_pool

        new_inspiration_pool = json.dumps(new_inspiration_pool, indent=2)

        # documents_text_GrandExpert is the Curated Literature Library.
        documents_text_GrandExpert = load_CLL_json(CLL_PATH, logger)

        generate_hypothesis_prompt = Msg(
            "system",
            get_Phase_one_GrandExpert_prompt(
                documents_text_GrandExpert, new_inspiration_pool, num_of_hypotheses=15
            ),
            role="user"
        )

        # with open('300(without_reasoning).txt', 'w', encoding='utf-8') as f:
        #     f.write(get_Phase_one_GrandExpert_prompt(
        #         documents_text_GrandExpert, new_inspiration_pool, num_of_hypotheses=30
        #     ))
        # exit(0)

        generate_hypothesis_response = make_api_call_with_retry(
            grand_expert, generate_hypothesis_prompt,
            all_agents, logger, GrandExpert_MODEL_CONFIG
        )


        try:
            initial_hypothesis = ConvertJSON_list(generate_hypothesis_response)

            with open(save_initial_hypothesis_path, 'w', encoding='utf-8') as f:
                json.dump(initial_hypothesis, f, indent=4, ensure_ascii=False)
            logger.info(f"Stage 3 Complete: Initial hypothesis saved to {save_initial_hypothesis_path}")

        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse JSON from initial_hypothesis: {e}")

    initial_history = [
        {"inspiration_pool": inspiration_pool},
        {"initial_hypothesis": initial_hypothesis}
    ]

    return initial_hypothesis, inspiration_pool, initial_history


def run_phase1(logger: logging.Logger, sliding_window_size: int,
                          screener_window_size: int,
                          max_inspirations_window: int,
                          min_inspirations_window: int,
                          checkpoint_all_inspirations: str = None,
                          checkpoint_inspiration_pool: str = None,
                          checkpoint_initial_hypothesis: str = None):
    """Main orchestration function with sliding window approach"""

    CLL_PATH = grand_expert.CLL_path

    # Check for a checkpoint.
    checkpoint_stage, checkpoint_dir = get_checkpoint_stage_and_dir(
        checkpoint_all_inspirations,
        checkpoint_inspiration_pool,
        checkpoint_initial_hypothesis
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Expert_MODEL_CONFIG.get("model_name", "unknown_model").replace("/", "_")

    if checkpoint_stage is not None:
        checkpoint_folder_name = os.path.basename(checkpoint_dir)
        run_dir_name = f"Checkpoint_{checkpoint_folder_name}_stage{checkpoint_stage}_{timestamp}"
        Results_dir = os.path.join(checkpoint_dir, run_dir_name)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"CHECKPOINT MODE: Starting from Stage {checkpoint_stage}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        logger.info(f"Results will be saved to: {Results_dir}")
        logger.info(f"{'=' * 60}\n")
    else:
        run_dir_name = f"{model_name}_{timestamp}_window{sliding_window_size}_screener{screener_window_size}"
        Results_dir = os.path.join(Results_PATH, run_dir_name)
        logger.info(f"\nNORMAL MODE: Starting from scratch")
        logger.info(f"Results will be saved to: {Results_dir}")

    os.makedirs(Results_dir, exist_ok=True)
    save_config_summary(Results_dir, CONFIG)
    current_prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.py')
    save_to_txt('prompt.txt', Results_dir, current_prompt_path, logger)

    run_log_dir = os.path.join(Results_dir, "run_log")
    os.makedirs(run_log_dir, exist_ok=True)

    log_file = os.path.join(run_log_dir, f"{file_name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Log file created at: {log_file}")
    logger.info(f"Sliding window size: {sliding_window_size}")
    logger.info(f"Screener window size: {screener_window_size}")

    # Display information for all checkpoints.
    if checkpoint_all_inspirations:
        logger.info(f"Checkpoint (Stage 1): {checkpoint_all_inspirations}")
    if checkpoint_inspiration_pool:
        logger.info(f"Checkpoint (Stage 2): {checkpoint_inspiration_pool}")
    if checkpoint_initial_hypothesis:
        logger.info(f"Checkpoint (Stage 3): {checkpoint_initial_hypothesis}")

    # Phase 1-4: Complete workflow
    initial_hypothesis, initial_pool, initial_history = phase_one_generate_initial_hypothesis(
        CLL_PATH=CLL_PATH,
        expert_agents=expert_agents,
        inspiration_screener=inspiration_screener,
        grand_expert=grand_expert,
        all_agents=all_agents,
        sliding_window_size=sliding_window_size,
        screener_window_size=screener_window_size,
        max_inspirations_window=max_inspirations_window,
        min_inspirations_window=min_inspirations_window,
        logger=logger,
        save_dir=Results_dir,
        checkpoint_all_inspirations=checkpoint_all_inspirations,
        checkpoint_inspiration_pool=checkpoint_inspiration_pool,
        checkpoint_initial_hypothesis=checkpoint_initial_hypothesis
    )

    if not initial_hypothesis:
        logger.error("Failed to generate a initial hypothesis. The process cannot continue.")
        return

    # Save token counts.
    token_stats_phase1_file = os.path.join(Results_dir, token_stats_phase1_name)

    with open(token_stats_phase1_file, 'w', encoding='utf-8') as f:
        json.dump(token_stats_phase1, f, indent=4, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("ALL PHASES COMPLETE!")
    logger.info(f"Total inspiration pool size: {len(initial_pool)}")
    logger.info(f"All results saved in: {Results_dir}")
    logger.info("=" * 60)


# Run Phase 1.
if __name__ == "__main__":
    enable_token_tracking(token_stats_phase1, verbose=False)
    expert_agents, inspiration_screener, grand_expert, novelty_critic, all_agents = create_all_agents(
        CONFIG, logger, phase="phase1")

    sliding_window_size = CONFIG["docs"]["SLIDING_WINDOW_SIZE"]
    screener_window_size = CONFIG["docs"]["SCREENER_WINDOW_SIZE"]
    include_reasoning = CONFIG["docs"]["INCLUDE_REASONING"]
    max_inspirations_window = CONFIG["docs"]["MAX_INSPIRATIONS_WINDOW"]
    min_inspirations_window = CONFIG["docs"]["MIN_INSPIRATIONS_WINDOW"]

    # Checkpoint paths for all four stages.
    # When resuming at Stage 3, all preceding checkpoint paths must also be provided.
    checkpoint_all_inspirations = "45Hypo/gemini-3.1-pro-preview-thinking_20260419_161728_window15_screener100/all_inspirations.json"
    checkpoint_inspiration_pool = "45Hypo/gemini-3.1-pro-preview-thinking_20260419_161728_window15_screener100/inspirations_pool.json"
    checkpoint_initial_hypothesis = None
    #checkpoint_all_inspirations = "90Hypo_round1/gemini-3-pro-preview-thinking_20260125_113548_window15_screener100/all_inspirations.json"
    #checkpoint_inspiration_pool = "90Hypo_round1/gemini-3-pro-preview-thinking_20260125_113548_window15_screener100/inspirations_pool.json"
    #checkpoint_initial_hypothesis = "90Hypo_round1/gemini-3-pro-preview-thinking_20260125_113548_window15_screener100/initial_hypothesis.json"
    #checkpoint_initial_hypothesis = "30Hypo_round2/Checkpoint_30Hypo_round2_stage2_20260325_152205/phase2/90_hypo.json"

    run_phase1(
        logger,
        sliding_window_size=sliding_window_size,
        screener_window_size=screener_window_size,
        max_inspirations_window=max_inspirations_window,
        min_inspirations_window=min_inspirations_window,
        checkpoint_all_inspirations=checkpoint_all_inspirations,
        checkpoint_inspiration_pool=checkpoint_inspiration_pool,
        checkpoint_initial_hypothesis=checkpoint_initial_hypothesis
    )

    logger.info("\nProcess complete.")
