import os
import random
import sys
from typing import List, Dict
from config import *

# Ensure agentscope is installed and dependencies are available
try:
    import agentscope
    from agentscope.agents import DialogAgent
except ImportError:
    print("Error: AgentScope library not found. Please install it using 'pip install agentscope'")
    sys.exit(1)

# Assuming 'prompt.py' and 'utils.py' are in a location accessible by PYTHONPATH
# Add the parent directory to the path to allow for sibling imports

# --- Local Dependencies ---
# These files should exist in your project structure

from utils import load_and_preprocess_json
from prompt import *

# ------------------------------------ Agent Class Definitions ------------------------------------
# -------------------------------------------------------------------------------------------------
class BiodiversityExpert(DialogAgent):
    """Expert Role: Provides inspiration based on literature and topic"""

    def __init__(self, name: str, subfield: str, model_config_name: str,
                 assigned_doc_ids: list, document_map: dict, use_memory: bool = False):
        self.name = name
        self.subfield = subfield
        # IDs of all articles.
        self.assigned_doc_ids = assigned_doc_ids
        # Dictionary mapping article IDs to content; used when mining inspirations in Phase 1.
        self.document_map = document_map
        # Retrieved Literature Library
        self.RLL = []
        # In Phase 2, set these IDs when using a fixed library instead of literature retrieval.
        self.new_documents_ids=[]

        sys_prompt = get_Expert_Sys_prompt(subfield)

        super().__init__(name=name, model_config_name=model_config_name, sys_prompt=sys_prompt, use_memory=use_memory)


# GrandExpert serves as the LeadingExpert here.
class GrandExpert(DialogAgent):
    """Grand Expert Role: Synthesizes, evaluates, generates hypotheses, and guides discussion"""

    def __init__(self, name: str, field: str, research_topic: str, research_background: str, model_config_name: str,
                 CLL_path:str, use_memory: bool = False):
        self.name = name
        self.field = field
        self.research_topic = research_topic
        self.research_background = research_background
        self.CLL_path = CLL_path

        sys_prompt = get_GrandExpert_Sys_prompt(field)

        super().__init__(name=name, model_config_name=model_config_name, sys_prompt=sys_prompt, use_memory=use_memory)


class InspirationScreener(DialogAgent):
    """InspirationScreener Role: Evaluates, filters, and merges inspirations."""
    def __init__(self, name: str, research_topic: str, research_background: str, model_config_name: str, use_memory: bool = False):
        self.name = name
        self.research_topic = research_topic
        self.research_background = research_background
        sys_prompt = get_Screener_Sys_prompt()
        super().__init__(name=name, model_config_name=model_config_name, sys_prompt=sys_prompt, use_memory=use_memory)

class NoveltyCritic(DialogAgent):
    """Novelty Critic Role: """

    def __init__(self, name: str, research_topic: str, research_background: str, model_config_name: str, use_memory: bool = False):
        self.name = name
        self.research_topic = research_topic
        self.research_background = research_background

        sys_prompt = get_Critic_Sys_prompt()

        super().__init__(name=name, model_config_name=model_config_name, sys_prompt=sys_prompt, use_memory=use_memory)
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def create_all_agents(CONFIG, logger, phase="phase1", num_of_experts=6):
    """
    Initializes and returns all the required agents for the debate.

    Args:
        CONFIG: Configuration dictionary
        logger: Logger instance
        phase: "phase1" or "phase2" - determines expert creation strategy
        num_of_experts: Number of experts to create in phase2 (default: 3)

    Returns:
        expert_agents, inspiration_screener, grand_expert, novelty_critic, all_agents
    """
    # --- Parsing config ---
    paths = CONFIG['paths']
    docs_settings = CONFIG['docs']
    agents_config = CONFIG['agents']
    research_config = CONFIG['research']

    # CLL: Curated Literature Library
    CLL_PATH = paths['CLL_PATH']

    INCLUDE_CONCLUSION_CONTENT = docs_settings['INCLUDE_CONCLUSION_CONTENT']
    INCLUDE_INTRO = docs_settings['INCLUDE_INTRO']

    # Load the expert-subfield configuration for the selected phase.
    if phase.lower() == "phase1":
        EXPERT_SUBFIELDS = agents_config.get('PHASE1_EXPERT_SUBFIELDS', {})
    else:
        EXPERT_SUBFIELDS = agents_config.get('PHASE2_EXPERT_SUBFIELDS', {})

    GRANDEXPERT_FIELD = agents_config['GRANDEXPERT_FIELD']

    RESEARCH_BACKGROUND = research_config['RESEARCH_BACKGROUND']
    RESEARCH_TOPIC = research_config['RESEARCH_TOPIC']

    # --- 1. Initialize AgentScope ---
    logger.info("Initializing AgentScope with the provided model configuration...")
    try:
        agentscope.init(model_configs=[
            Expert_MODEL_CONFIG,
            GrandExpert_MODEL_CONFIG,
            Screener_MODEL_CONFIG,
            Critic_MODEL_CONFIG
        ])
    except Exception as e:
        logger.error(f"Failed to initialize AgentScope: {e}")
        return None, None, None, None, None

    # --- 2. Load and Preprocess Data ---
    logger.info(f"Loading Data from: '{CLL_PATH}'")
    if not os.path.exists(CLL_PATH):
        logger.error(f"File not found at '{CLL_PATH}'")
        return None, None, None, None, None

    # all_docs contains the full library; document_map associates each key with its content.
    all_docs, document_map = load_and_preprocess_json(
        CLL_PATH, INCLUDE_CONCLUSION_CONTENT, INCLUDE_INTRO, logger
    )
    if not all_docs:
        logger.error("No documents were loaded. Cannot create agents.")
        return None, None, None, None, None

    # all_doc_ids contains the IDs of all articles.
    all_doc_ids = list(document_map.keys())
    logger.info(f"Total documents loaded: {len(all_doc_ids)}")

    # --- 3. Initialize Expert Agent(s) based on phase ---
    expert_agents = []

    if phase.lower() == "phase1":
        # Phase 1: Single expert with all documents (sliding window approach)
        logger.info("Phase 1: Initializing single Biodiversity Expert agent with sliding window approach...")

        expert_name = "Dr_Expert"
        expert_subfield = EXPERT_SUBFIELDS.get(expert_name, "Species Interaction")

        expert = BiodiversityExpert(
            name=expert_name,
            subfield=expert_subfield,
            model_config_name="Expert_config",
            assigned_doc_ids=all_doc_ids,  # Assign all articles.
            document_map=document_map,
            use_memory=False
        )
        expert_agents = [expert]
        logger.info(
            f"Initialized '{expert_name}' (subfield: {expert_subfield}) with access to all {len(all_doc_ids)} documents.")

    elif phase.lower() == "phase2":
        # Phase 2: Multiple experts with distributed documents
        logger.info(
            f"Phase 2: Initializing {num_of_experts} Biodiversity Expert agents with document distribution...")

        for i in range(num_of_experts):
            expert_name = f"Dr_Expert{i + 1}"
            expert_subfield = EXPERT_SUBFIELDS.get(expert_name, "Species Interaction")

            expert = BiodiversityExpert(
                name=expert_name,
                subfield=expert_subfield,
                model_config_name="Expert_config",
                assigned_doc_ids=[],
                document_map=document_map,
                use_memory=False
            )
            expert_agents.append(expert)

            logger.info(
                f"Initialized '{expert_name}' (subfield: {expert_subfield}) ")
    else:
        logger.error(f"Invalid phase parameter: '{phase}'. Must be 'phase1' or 'phase2'.")
        return None, None, None, None, None

    # --- 4. Initialize Screener Agent ---
    logger.info("Initializing Inspiration Screener agent...")
    inspiration_screener = InspirationScreener(
        name="Dr_Screener",
        research_topic=RESEARCH_TOPIC,
        research_background=RESEARCH_BACKGROUND,
        model_config_name="Screener_config",
        use_memory=False
    )

    # --- 5. Initialize Grand Expert Agent ---
    logger.info("Initializing Grand Expert agent...")
    grand_expert = GrandExpert(
        name="Grand_Professor_Wisdom",
        field=GRANDEXPERT_FIELD,
        research_topic=RESEARCH_TOPIC,
        research_background=RESEARCH_BACKGROUND,
        model_config_name="GrandExpert_config",
        CLL_path = CLL_PATH,
        use_memory=False
    )

    # --- 6. Initialize Novelty Critic Agent ---
    logger.info("Initializing Novelty Critic Agent...")
    novelty_critic = NoveltyCritic(
        name="Dr_Novelty_Critic",
        research_topic=RESEARCH_TOPIC,
        research_background=RESEARCH_BACKGROUND,
        model_config_name="Critic_config",
        use_memory=False
    )

    logger.info("All agents have been successfully created.")

    global all_agents
    all_agents = expert_agents + [inspiration_screener, grand_expert, novelty_critic]

    return expert_agents, inspiration_screener, grand_expert, novelty_critic, all_agents



