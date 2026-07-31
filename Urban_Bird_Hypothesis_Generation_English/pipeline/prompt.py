import json
import config
research_topic = config.CONFIG["research"]["RESEARCH_TOPIC"]
research_background = config.CONFIG["research"]["RESEARCH_BACKGROUND"]

# Context of hypothesis generation
def Context_of_hypothesis_generation():
    prompt_content = (
        f"""
        ## Context of hypothesis generation
        As a member of a scientific team specializing in urban biodiversity, you are helping with generating the scientific hypothesis related to the research topic. ** The research topic is: '{research_topic}' and the research background is: '{research_background}'**.
        We in general split the period of scientific hypothesis generation into two phase, four steps:
        1. Phase One, Step One: Uncovering Inspirations
        2. Phase One, Step Two: Generating Initial Hypotheses
        3. Phase Two, Step Three: Deepening Inspirations
        4. Phase Two, Step Four: Refining Hypotheses
        """
    )
    return prompt_content

# Definition of inspiration
def Definition_inspiration():
    prompt_content = (
        f"""
        ### Definition of an inspiration
        Inspirations are the 'spark' for a hypothesis, not the final hypothesis itself.
        A good inspiration should be highly concentrated on a single key concept, pattern, or observation, making it easy to associate with others. **An inspiration can be inspired by one article or multiple articles**.
        Inspirations include, but are not limited to the following categories:
        1. **Pattern-based inspiration**. A recurrent pattern, correlation, or trend observed across empirical studies, spatial or temporal scales, or taxa, whose underlying mechanisms remain unclear.
        2. **Mechanism-based inspiration**. An established or partially understood biological, ecological, or behavioral process that could play a causal role in shaping observed outcomes but has not been fully integrated into a broader explanatory framework.
        3. **Context-shift inspiration**. A well-established ecological theory or mechanism applied to a novel environmental, spatial, or socio-ecological context where its effects may differ in direction or magnitude.
        4. **Scale-mismatch inspiration**. A situation in which processes occurring at one biological, spatial, or temporal scale are likely to generate consequences at another scale, yet this cross-scale linkage remains underexplored.
        5. **Interaction-combination inspiration**. The recognition that two or more factors, drivers, or processes—often studied independently—may interact synergistically or antagonistically to produce emergent ecological outcomes.
        6. **Constraint or trade-off inspiration**. A potential limitation, trade-off, or constraint (e.g., energetic, physiological, evolutionary, or informational) that may shape organismal or ecosystem responses under specific conditions.
        7. **Contradiction or anomaly-based inspiration**. An empirical result or theoretical implication that deviates from dominant expectations, revealing potential gaps or inconsistencies in current understanding.
        """
    )
    return prompt_content

def Definition_hypothesis():
    prompt_content = (
        f"""
        ### Definition of the hypothesis
        1. The hypothesis is a tentative explanation, relationship, or proposition that can be empirically tested or theoretically evaluated.
        2. **A good hypothesis should not be a summary of current knowledge, but a reasonable guess based on the knowledge**.
        3. Structurally, scientific hypotheses take the form of a proposed relationship between variables. It links an independent variable (the presumed cause, which is manipulated or observed) and a dependent variable (the presumed effect, which is measured): If [a specific change is made to the independent variable], then [a specific change will be observed in the dependent variable].
        
        ### **Core characteristics of a good hypothesis**:
        1. **Novelty** 
        - The hypothesis should propose a novel idea, relationship, or explanation that has not yet been adequately proposed or confirmed.
        2. **Significance**
        - The hypothesis must address a meaningful scientific gap or critical challenge, offering potential to advance theory or practice. It should avoid trivial or isolated phenomena that lack broader ecological relevance. 
        - The hypothesis should focus on the most direct and dominant ecological pathways to explain the phenomenon, ensuring the hypothesis is built on major drivers rather than long, speculative causal chains.
        3. **Plausibility**
        - The hypothesis must possess logical coherence, meaning it is free from internal contradictions and its components are well-defined. 
        - A good hypothesis is not a wild guess but is built upon a solid understanding of existing literature, theories, and observations.
        4. **Testability**
        - The hypothesis must be logically capable of being proven false. If a theory can explain every possible outcome and no evidence can refute it, it is not scientific.  
        5. **Clarity**
        - The hypothesis should be articulated using precise, widely understood scientific terminology rather than relying on metaphors or colloquialisms. 
        - The hypothesis should be as simple as possible while still accounting for the phenomenon. It should not introduce excessive variables or redundant information.
        6. **Specificity**
        - A hypothesis must be specific, clearly identifying the key ecological mechanisms, variables, and contexts involved, rather than remaining abstract or overly general.  
        """
        )
    return prompt_content

# ------------------------------------ System Prompt ------------------------------------
# Expert System Prompt
def get_Expert_Sys_prompt(subfield):
    prompt_content = (
        f"""
        ## Role and Expertise
        You are an ambitious expert in the subfield **'{subfield}'** in urban biodiversity.
        You are skilled at uncovering inspirations for novel hypotheses by analyzing the 'Knowledge' provided and applying your domain expertise.
        {Definition_inspiration()}
        {Definition_hypothesis()}
        ------------------ EXECUTION REMINDER ------------------
        When performing the task:
        - Follow the **Detailed Instructions** in the user prompt carefully and consistently.
        - If **Important Guidelines** are provided, pay particular attention to them and ensure they are well satisfied.
        - Ensure the final output aligns with both the task requirements and these instructions.
        """
    )
    return prompt_content


# Leading Expert System Prompt
def get_GrandExpert_Sys_prompt(field):
    prompt_content = (
        f"""
        ## Role and Expertise
        You are a highly experienced Leading Expert in **scientific methodology and {field}**. 
        You are good at summarizing inspirations and generating high-quality hypotheses. You are also skilled at organizing discussions among scientists to inspire and ensure they maximize their capabilities. 
        {Definition_hypothesis()}
        ------------------ EXECUTION REMINDER ------------------
        When performing the task:
        - Follow the **Detailed Instructions** in the user prompt carefully and consistently.
        - If **Important Guidelines** are provided, pay particular attention to them and ensure they are well satisfied.
        - Ensure the final output aligns with both the task requirements and these instructions.
        """
    )
    return prompt_content

# Critic System Prompt
def get_Critic_Sys_prompt():
    prompt_content = (
        f"""
        ## Role and Expertise
        You are ** a very strict and academically minded scientific critic** in urban biodiversity.
        You are skilled at scrutinizing the proposed scientific hypotheses rigorously, according to the 'Core characteristics of a good hypothesis'.
        {Definition_hypothesis()}
        ------------------ EXECUTION REMINDER ------------------
        When performing the task:
        - Follow the **Detailed Instructions** in the user prompt carefully and consistently.
        - If **Important Guidelines** are provided, pay particular attention to them and ensure they are well satisfied.
        - Ensure the final output aligns with both the task requirements and these instructions.
        """
    )
    return prompt_content


# Screener System Prompt
def get_Screener_Sys_prompt():
    prompt_content = (
        f"""
        ## Role and Expertise
        You are an expert in inspiration screening in urban biodiversity, skilled at selecting and summarizing similar and relevant inspirations.
        Your primary skill is to judge whether the inspiration belongs to the research topic, and to merge similar inspirations. You need to strictly control the size of the inspiration pool according to the settings.
        {Definition_inspiration()}
        ------------------ EXECUTION REMINDER ------------------
        When performing the task:
        - Follow the **Detailed Instructions** in the user prompt carefully and consistently.
        - If **Important Guidelines** are provided, pay particular attention to them and ensure they are well satisfied.
        - Ensure the final output aligns with both the task requirements and these instructions.
        """
    )
    return prompt_content


# ------------------------------------ User Prompt ------------------------------------
def Phase_one_Expert_Window_prompt(documents_text):
    prompt_content = (
        f"""
        {Context_of_hypothesis_generation()}
        **You are now participating in 'Phase One, Step One: Uncovering Inspirations'**.
        
        ## Task
        Your task is to search for **inspirations** related to the research topic from the curated literature library (Knowledge) for novel hypotheses.
        ------------------ BEGIN OF KNOWLEDGE ------------------
        ## Knowledge
        Curated literature library:
        {documents_text}
        ------------------ END OF KNOWLEDGE ------------------
        
        ------------------ BEGIN OF DETAILED INSTRUCTIONS ------------------
        ## Detailed Instructions
        **Now, based on the Task and the Knowledge above, follow these instructions precisely**.
        
        ### Method
        Based on the **research topic**, the **research background **, and the **knowledge**, generate creative 'inspirations' closely related to the research topic.
        
        ### Important Guidelines (MUST be strictly followed)
        1. **An inspiration can be drawn from a single article or multiple articles**. 
        2. **Every paper must contribute to at least one inspiration. Ensure no article is left unused**.
        3. **Targeting at publishing the research on a top venue like Nature or Science, the inspirations must be both novel, important and logically feasible to support the generation of high-quality hypotheses.**.
        
        ### Output Format
        Your response needs to follow the strict format, the output MUST be a list containing one or more JSON (eg.[{{dict}}, {{dict}}]).
        **Do not include any other text or explanation outside of the JSON or the LIST.**
        For example, the JSON format for Inspiration 1 in the list should strictly be:
        {{
        "Inspiration 1": "A single sentence describing the inspiration you identified based on your domain knowledge and the curated literature library.",
        "Source": "Explain which articles you got the inspiration from (If there are multiple articles, please list all IDs. Example: "1, 2, 3, 4". Just list the IDs without adding other information.)"
        }}
        """
    )
    return prompt_content


def get_screener_update_prompt(current_inspiration_pool, current_hypothesis, min_pool_num, max_pool_num):
    prompt_content = (
        f"""
        {Context_of_hypothesis_generation()}
        **You are now participating in 'Phase One, Step One: Uncovering Inspirations'**.
        
        ## Task
        {current_hypothesis}Your task is to update the 'Inspiration Pool'.
        ------------------ BEGIN OF INSPIRATION POOL ------------------
        ## Current inspiration pool:
        {current_inspiration_pool}
        ------------------ END OF INSPIRATION POOL ------------------
        
        ------------------ BEGIN OF DETAILED INSTRUCTIONS ------------------
        ## Detailed Instructions
        **Now, based on the Task and the Inspiration pool above, follow these instructions precisely**.
        
        ### Method
        -1. **Check for Relevance to topic and the hypothesis**: Scrutinize all inspirations for their relevance to the research topic and the hypothesis. If an inspiration is not closely related to the topic and the hypothesis, it must be removed from the inspiration pool.
        -2. **Evaluate for Logical Coherence**: Assess whether each inspiration is scientifically sound and logically coherent. If an inspiration contains clear logical flaws or unsupported claims, or relies on a very long, speculative causal chain, you should remove it from the inspiration pool.
        -3. **Identify and Merge Redundant Ideas**: Carefully read through the 'Current inspiration pool'. If you find two or more inspirations that express the same or a very similar core idea, you should merge them into a single, comprehensive inspiration. The new merged inspiration should capture the essence of all original ideas without redundancy.
        -4. **Prioritize primary drivers and processes**: Favor inspirations that link key ecological factors and processes to fundamental ecological responses.
        -5. **Manage Pool Size**: **The final inspiration pool should have {min_pool_num} to {max_pool_num} inspirations**. If the pool size exceeds this limit, you must prioritize the most novel, important and logically feasible inspirations to keep.
        
        ### Output Format
        **Strict Output**: The output MUST be a list containing one or more JSON (eg.[{{dict}},{{dict}}]), **do not include any other text or explanation outside of the JSON or the LIST**.
        The JSON format for each inspiration in the list should strictly be:
        {{
        "inspiration_id": "a unique inspiration id",
        "source": "The ID or IDs of the source of this inspiration (The inspiration may be combination of multiple inspirations. Example: "1,2,3,4")",
        "content": "The content of the inspiration"
        }}
        """
    )

    return prompt_content

# Leading Expert Generate initial hypothesis.
def get_Phase_one_GrandExpert_prompt(documents_text, inspirations_context,
                                     num_of_hypotheses):
    prompt_content = (
        f"""
        {Context_of_hypothesis_generation()}
        **You are now participating in 'Phase One, Step Two: Generating Initial Hypotheses'**.
        ## Task
        **Your task is to formulate {num_of_hypotheses} distinct initial hypotheses based on the Knowledge**.
        ----------------- BEGIN OF KNOWLEDGE ------------------
        ## Knowledge
        ### Curated literature library:
        {documents_text}
        ### **Inspirations Pool**:
        {inspirations_context}
        ----------------- END OF KNOWLEDGE ------------------
        
        ------------------ BEGIN OF DETAILED INSTRUCTIONS ------------------
        ## Detailed Instructions 
        **Now, based on the Task and the Knowledge above, follow these instructions precisely**.
        
        ### Method
        Based on the research topic, the research background, and the **inspirations** above, formulate {num_of_hypotheses} novel, important and logically feasible hypothesis that **could truly guide the development of urban biodiversity**. A hypothesis can be generated from any reasonable combination of one or more inspirations and one or more papers.
        
        ### Important Guidelines (MUST be strictly followed)
        1. Follow **Core characteristics of a good hypothesis** provided above.
        2. **The hypothesis should be closely related to the research topic**.
        3. Avoid copying inspirations directly. **The hypothesis must be genuinely novel, not a predictable extension of the provided text**.
        4. **The hypothesis should be stated in a clear and concise manner**. Complex reasoning, multi-step causal pathways, or detailed theoretical justification should be placed in the Reasoning section, not embedded within the hypothesis itself.
        5. The hypothesis must be a **coherent and logically grounded conjecture**, not the aggregation of unrelated factors or inspirations.
        6. **Avoid creating overly complex or abstract hypotheses that cannot be empirically tested within reasonable experimental or observational frameworks.** Additionally, note that a hypothesis is not the same as an experimental design. Therefore, it should not attempt to specify concrete experimental procedures or numerical parameters. The pathway to verification or falsification should be clear, feasible, and should not rely on excessive or unverified assumptions.
        7. **When generating hypotheses, focus on major and direct ecological mechanisms rather than long and speculative causal chains. Avoid hypotheses that rely on multiple indirect steps or unlikely intermediary events to link cause and effect. A hypothesis should explain phenomena using the most direct and primary mechanism, not a complex chain of minor influences**.
        8. **Ecological significance. The generated hypothesis must address a question of clear and broad importance to ecology or urban biodiversity**. Avoid hypotheses that are merely novel, plausible, or taxon-specific but have limited theoretical, empirical, or applied ecological significance. Hypotheses whose outcomes would not substantially advance ecological understanding or inform biodiversity management should be avoided.
        9. **Please ensure that each generated hypothesis is specific, clearly identifying the key ecological mechanisms, variables, and contexts involved, rather than remaining abstract or overly general**.
        10. **High-quality hypotheses do not exceed 50 words**.
        11. **The hypotheses you generate may be inspired by both the papers you have and the inspiration pool. So please list the sources separately. Note that 'Source_from_Papers should' not only list the papers that generated the hypotheses, but also the source papers of inspirations in 'Source_from_InspirationPool'**.
           
        ### Hypothesis Format
        The format of the hypothesis must be:
        'If [a specific change is made to the independent variable], then [a specific change will be observed in the dependent variable].'
        
        ### Output Format
        Your response needs to follow this strict format.
        **Strict Output**: The output MUST be a list containing one or more JSON (eg.[{{dict}},{{dict}}]), **do not include any other text or explanation outside of the JSON or the LIST**. The JSON format for each Initial Hypothesis in the list should strictly be:
        {{
        "hypothesis_id": "Hypothesis id",
        "Initial_Hypothesis": "Hypothesis content",
        "Reasoning": "Provide the scientific rationale for why this hypothesis could be true, explain the causal mechanisms, supporting evidence, or theoretical basis",
        "Source_from_InspirationPool": "IDs of inspirations derived from the Inspiration Pool. (Strictly consistent with the ID in the Inspiration pool, Example: "1,2,3,4")", 
        "Source_from_Papers": "IDs of inspirations derived from the papers.(Example: "1,2,3,4")"
        }}
        """
    )
    return prompt_content


# ------------------------------- Phase 2 -------------------------------
# Expert generate inspirations in phase 2

def get_Phase_two_Expert_Inspiration_prompt(current_inspiration_pool, current_hypothesis, curated_literature_library,
                                            curated_literature_library_inspiration_pool, new_documents_text,
                                            discussion_history, min_num_inspirations=2, max_num_inspirations=3):
    prompt_content = (
        f"""
        {Context_of_hypothesis_generation()}
        **You are now participating in 'Phase Two, Step Three: Deepening Inspirations'**.
        
        ## Task
        You should search for **inspirations** related to the research topic from your knowledge for novel research hypotheses.
        ------------------ BEGIN OF KNOWLEDGE ------------------
        ## Knowledge
        ### Current Hypothesis:
        **{current_hypothesis}**
        ### Current inspiration pool:
        {current_inspiration_pool}
        ### Papers that inspires the hypothesis:
        {curated_literature_library}
        ### Inspiration Pool of the Curated literature_library:
        {curated_literature_library_inspiration_pool} 
        ### Retrieved literature library related to the hypothesis:
        **The papers are organized in the form of chains of papers, from early to late**
        {new_documents_text}
        ### Discussions among experts and Guidance of Leading Expert:
        {discussion_history}
        ------------------ END OF KNOWLEDGE ------------------
        
        ------------------ BEGIN OF DETAILED INSTRUCTIONS ------------------
        ## Detailed Instructions
        **Now, based on the Task and the Knowledge above, follow these instructions precisely**.
        
        ### Method
        **Your goal is not to simply summarize, but to strategically retrieve, refine, and combine new knowledge to generate high quality inspirations**.
        -Step 1. **Understand the Current Hypothesis and Discussion**: Thoroughly understand the current hypothesis and discussion. Also read all scientific papers carefully. Focus on the guidance of Leading Expert (if any exists).
        -Step 2. **Choose Action**: Based on your analysis of all provided information, please propose {min_num_inspirations} to {max_num_inspirations} new inspirations. Each inspiration can be advanced or proposed through the following actions:
        **1. CRITIQUE** :Critique an existing inspiration in current inspiration pool to get new inspirations. Clearly state the weakness, flaw, or missing component. Your critique must be specific, constructive, and based on your knowledge and papers.
        **2. SUPPLEMENT**: Supplement an existing inspiration in current inspiration pool to enhance its validity by adding a crucial detail, a supporting piece of evidence, or a new perspective. Identify which inspiration you are addressing and strengthen it based on your knowledge and papers.
        **3. EVOLVE**: Propose a new, evolved version of an inspiration in current inspiration pool. You can achieve this by either: a. **Fusing Concepts:** Combine concepts from two or more existing inspirations to create a more comprehensive idea. b. **Integrating a Novel Concept:** Introduce a new concept from your knowledge or papers to resolve a flaw or significantly enhance the inspiration's impact. **Remember: integrating new concepts isn't about adding complexity to ideas, but about discovering novel, reasonable relationships between existing concepts** .
        **4. PROPOSE**: Based on the Knowledge above, you can come up with totally new inspirations that are different from current inspiration pool.
        ** Pay attention to broader, primary and more direct of the phenomenon, instead of a minor contributing process. **

        ### Output Format
        **Strict Output**:
        The output MUST be a list containing one or more JSON (eg.[{{dict}},{{dict}}]), ensure all strings within the JSON are properly escaped.
        **Do not include any other text or explanation outside of the JSON or the LIST.** Each JSON object in the list must strictly adhere to the following structure:
        {{
        "Action": "A string, which must be one of: CRITIQUE, SUPPLEMENT, EVOLVE, PROPOSE.",
        "Inspiration": "The content of the new inspiration.",
        "Source": "A string listing the source document IDs or inspiration reference. (e.g., \'Doc ID, Inspiration ID\'.)"
        }}
        """
    )
    return prompt_content

# ### Curated literature library:
#         {curated_literature_library}
# Leading Expert guide the discussion between experts
def get_Phase_two_GrandExpert_Summary_prompt(curated_literature_library, curated_literature_library_inspiration_pool,
                                             retrieved_literature_library, current_hypothesis,
                                             discussion_history):
    prompt_content = (
        f"""
        {Context_of_hypothesis_generation()}
        **You are now participating in 'Phase Two, Step Three: Deepening Inspirations'**.
        
        ## Task
        Based on the current hypothesis, experts have generated the above discussion and proposed the inspirations. Your core mission is to elevate the team's thinking. You must guide them toward a deeper, more creative exploration of the knowledge space, and encourage the generation of higher-quality, more insightful inspirations that can lead to hypotheses of greater scientific value.
        ------------------ BEGIN OF KNOWLEDGE ------------------
        ## Knowledge
        ### Current Hypothesis:
        {current_hypothesis}
        ### Inspiration Pool of the Curated literature_library
        {curated_literature_library_inspiration_pool} 
        ### Retrieved literature library related to the hypothesis:
        **The papers are organized in the form of chains of papers, from early to late**
        {retrieved_literature_library}
        ### Discussions so far:
        {discussion_history}
        ------------------ END OF KNOWLEDGE ------------------
        
        ------------------ BEGIN OF DETAILED INSTRUCTIONS ------------------
        ## Detailed Instructions
        **Now, based on the Task and the Knowledge above, follow these instructions precisely**.
        
        ### Method
        Step 1: Analytical Summary
        First, critically analyze the current state of the discussion and the inspirations generated. You must answer these questions:
        1. **What is the overall quality, importance and novelty of the inspirations so far?**
        -To what extent do the generated hypotheses demonstrate genuine novelty compared to existing urban biodiversity literature?
        -Are the hypotheses generated significant to the field of urban biodiversity and can they truly guide the development of this field ?
        2. **What are the key gaps or weaknesses in the current discussion?**
        
        Step 2: Strategic Directive for the Next Round
        Based on your analysis, provide a clear, forward-looking strategy for the team. Your directive should guide the experts **to find high-quality inspirations that are closely related to the hypothesis.** Please summarize this round of discussions and provide guidance for the next round of discussions or final integration.
         
        ### Output Format
        **Strict Output**: Your response needs to follow this strict format, do not include any other text outside the JSON block.
        The JSON format should strictly be:
        {{
        "Summary": "[The Analytical Summary]",
        "Guidance": "[The Strategic Directive for the Next Round]"
        }}
        """
    )
    return prompt_content
# ### Curated literature library:
#         {curated_literature_library}
# Leading Expert generate 3 alternative hypotheses

def get_Phase_two_GrandExpert_Refine_Hypothsis_multiple_branch_prompt(curated_literature_library, curated_literature_library_inspiration_pool,
                                                                      retrieved_literature_library, current_hypothesis,
                                                                      current_inspiration_pool, critic_info):
    prompt_content = (
        f"""
        {Context_of_hypothesis_generation()}
        **You are now participating in 'Phase Two, Step Four: Refining Hypotheses'**.
        
        ## Task
        The formulation and testing of multiple alternative hypotheses is a cornerstone of modern ecological and evolutionary research, crucial for overcoming confirmation bias. By designing crucial experiments that distinguish among competing explanations, science advances through systematic exclusion. Compared to null hypotheses, mechanistic and testable alternatives are more meaningful.
        Your task is to generate ** three refined hypotheses** from multiple perspectives based on the above knowledge.
        ------------------ BEGIN OF KNOWLEDGE ------------------
        ## Knowledge
        ### Current Hypothesis:
        {current_hypothesis}
        ### Inspiration Pool of the Curated literature_library
        {curated_literature_library_inspiration_pool} 
        ### Retrieved literature library related to the hypothesis:
        **The papers are organized in the form of chains of papers, from early to late**
        {retrieved_literature_library}
        ### Current Inspiration Pool:
        {current_inspiration_pool}
        ### **Critic's feedback and suggestions**:
        {critic_info}
        ------------------ END OF KNOWLEDGE ------------------
        
        ------------------ BEGIN OF DETAILED INSTRUCTIONS ------------------
        ## Detailed Instructions
        **Now, based on the Task and the Knowledge above, follow these instructions precisely**.

        ### Method
        1. **Critical Synthesis**: Analyze the 'Current Hypothesis'. If there is, focus on the Critic's feedback and identify which inspirations directly address those weaknesses.
        2. **Multiple Optimization Directions**: For each refined hypothesis, adopt a different optimization focus.
        3. **Provide a Clear Rationale**: Explain the reasoning behind your refined hypothesis. Provide the scientific rationale for why this hypothesis could be true, explain the causal mechanisms, supporting evidence, or theoretical basis.
         
        ### Important Guidelines (MUST be strictly followed)
        1. Follow **Core characteristics of a good hypothesis** provided above.
        2. **The hypothesis should be closely related to the research topic**.
        3. Avoid copying inspirations directly. **The hypothesis must be genuinely novel, not a predictable extension of the provided text**.
        4. **The hypothesis should be stated in a clear and concise manner**. Complex reasoning, multi-step causal pathways, or detailed theoretical justification should be placed in the Reasoning section, not embedded within the hypothesis itself.
        5. The hypothesis must be a **coherent and logically grounded conjecture**, not the aggregation of unrelated factors or inspirations.
        6. **Avoid creating overly complex or abstract hypotheses that cannot be empirically tested within reasonable experimental or observational frameworks**. Additionally, note that a hypothesis is not the same as an experimental design. Therefore, it should not attempt to specify concrete experimental procedures or numerical parameters. The pathway to verification or falsification should be clear, feasible, and should not rely on excessive or unverified assumptions.
        7. **When generating hypotheses, focus on major and direct ecological mechanisms rather than long and speculative causal chains. Avoid hypotheses that rely on multiple indirect steps or unlikely intermediary events to link cause and effect. A hypothesis should explain phenomena using the most direct and primary mechanism, not a complex chain of minor influences**.
        8. **Ecological significance. The generated hypothesis must address a question of clear and broad importance to ecology or urban biodiversity**. Avoid hypotheses that are merely novel, plausible, or taxon-specific but have limited theoretical, empirical, or applied ecological significance. Hypotheses whose outcomes would not substantially advance ecological understanding or inform biodiversity management should be avoided.
        9. **Please ensure that each generated hypothesis is specific, clearly identifying the key ecological mechanisms, variables, and contexts involved, rather than remaining abstract or overly general**.
        10. **High-quality hypotheses do not exceed 50 words**.
        11. Pay attention on Critic's feedback and suggestions.
        
        ### Hypothesis Format
        The format of the hypothesis must be:
        'If [a specific change is made to the independent variable], then [a specific change will be observed in the dependent variable].'

        ### Output Format
        **Strict Output**: Your response needs to follow this strict format, do not include any other text outside the JSON block.
        The output MUST be a list containing one or more JSON (eg.[{{dict}},{{dict}}]), the JSON format for the Refined Hypothesis should strictly be:
        {{
        "ID": "Unique ID for the hypothesis",
        "Refined_Hypothesis": "Clear and concise hypothesis content",
        "Reasoning": "Provide the scientific rationale for why this hypothesis could be true, explain the causal mechanisms, supporting evidence, or theoretical basis, rather than describing how the hypothesis was revised or how feedback was incorporated.",
        "Source": "A string listing the IDs of the source inspirations"
        }}
        """
    )
    return prompt_content

#  ### Curated literature library:
#         {curated_literature_library}
# Leading Expert refine one of the alternative hypotheses
def get_Phase_two_GrandExpert_Refine_Hypothsis_prompt(curated_literature_library, curated_literature_library_inspiration_pool,
                                                      retrieved_literature_library, current_hypothesis,
                                                      current_inspiration_pool, critic_info):
    prompt_content = (
        f"""
        {Context_of_hypothesis_generation()}
        **You are now participating in 'Phase Two, Step Four: Refining Hypotheses'**.
        
        ## Task
        Your task is to refine current hypothesis based on the knowledge.
        ------------------ BEGIN OF KNOWLEDGE ------------------
        ## Knowledge
        ### Current Hypothesis:
        {current_hypothesis}
        ### Inspiration Pool of the Curated literature_library
        {curated_literature_library_inspiration_pool} 
        ### Retrieved literature library related to the hypothesis:
        **The papers are organized in the form of chains of papers, from early to late**
        {retrieved_literature_library}
        ### Current Inspiration Pool:
        {current_inspiration_pool}
        ### **Critic's feedback and suggestions**:
        {critic_info}
        ------------------ END OF KNOWLEDGE ------------------
        
        ------------------ BEGIN OF DETAILED INSTRUCTIONS ------------------
        ## Detailed Instructions
        **Now, based on the Task and the Knowledge above, follow these instructions precisely**.

        ### Method
        1. **Critical Synthesis**: Analyze the 'Current Hypothesis'. If there is，focus on the Critic's feedback and suggestions regarding novelty and identify which inspirations directly address those weaknesses.
        2. **Generate a Refined Hypothesis**: Based on the current hypothesis, the feedback and suggestions from the critic, generate a single refined hypothesis.
        3. **Provide a Clear Rationale**: Explain the reasoning behind your refined hypothesis. Provide the scientific rationale for why this hypothesis could be true, explain the causal mechanisms, supporting evidence, or theoretical basis.
         
        ### Important Guidelines (MUST be strictly followed)
        1. Follow **Core characteristics of a good hypothesis** provided above.
        2. **The hypothesis should be closely related to the research topic**.
        3. Avoid copying inspirations directly. **The hypothesis must be genuinely novel, not a predictable extension of the provided text**.
        4. **The hypothesis should be stated in a clear and concise manner**. Complex reasoning, multi-step causal pathways, or detailed theoretical justification should be placed in the Reasoning section, not embedded within the hypothesis itself.
        5. The hypothesis must be a **coherent and logically grounded conjecture**, not the aggregation of unrelated factors or inspirations.
        6. **Avoid creating overly complex or abstract hypotheses that cannot be empirically tested within reasonable experimental or observational frameworks**. Additionally, note that a hypothesis is not the same as an experimental design. Therefore, it should not attempt to specify concrete experimental procedures or numerical parameters. The pathway to verification or falsification should be clear, feasible, and should not rely on excessive or unverified assumptions.
        7. **When generating hypotheses, focus on major and direct ecological mechanisms rather than long and speculative causal chains. Avoid hypotheses that rely on multiple indirect steps or unlikely intermediary events to link cause and effect. A hypothesis should explain phenomena using the most direct and primary mechanism, not a complex chain of minor influences**.
        8. **Ecological significance. The generated hypothesis must address a question of clear and broad importance to ecology or urban biodiversity**. Avoid hypotheses that are merely novel, plausible, or taxon-specific but have limited theoretical, empirical, or applied ecological significance. Hypotheses whose outcomes would not substantially advance ecological understanding or inform biodiversity management should be avoided.
        9. **Please ensure that each generated hypothesis is specific, clearly identifying the key ecological mechanisms, variables, and contexts involved, rather than remaining abstract or overly general**.
        10. **High-quality hypotheses do not exceed 50 words**.
        11. Pay attention on Critic's feedback and suggestions.
        
        ### Hypothesis Format
        The format of the hypothesis must be:
        'If [a specific change is made to the independent variable], then [a specific change will be observed in the dependent variable].'
        
        ### Output Format
        **Strict Output**: Your response needs to follow this strict format, do not include any other text outside the JSON block.
        Output only **one** refined hypothesis. The JSON format for the Refined Hypothesis should strictly be:
        {{
        "ID": "Unique ID for the hypothesis",
        "Refined_Hypothesis": "Hypothesis content",
        "Reasoning": "Provide the scientific rationale for why this hypothesis could be true, explain the causal mechanisms, supporting evidence, or theoretical basis, rather than describing how the hypothesis was revised or how feedback was incorporated.",
        "Source": "A string listing the IDs of the source inspirations"
        }}
        """
    )
    return prompt_content

# Critic generate feedback and suggestions for the hypothesis
def get_Phase_two_Critic_prompt(hypothesis):
    prompt_content = f"""
        {Context_of_hypothesis_generation()}
        **You are now participating in 'Phase Two, Step Four: Refining Hypotheses'**.
        ## Task 
        Your task is to generate feedback and suggestions for the hypothesis.
        Your goal is not to simply reject the hypothesis, but to provide specific, actionable feedback to help  the 'Leading Expert' iteratively refine the hypothesis, making it more innovative and scientifically valuable.

        ## Evaluation Criteria
        You must evaluate each hypothesis according to the following six dimensions:
        1. **Novelty** – Does the hypothesis introduce genuinely new mechanisms, frameworks, or perspectives that could change how urban biodiversity is understood?
        2. **Significance** – Would confirming this hypothesis substantially advance urban ecological theory or practice, opening new research or management directions?
        3. **Plausibility** – Is the hypothesis logically consistent with established ecological theory and empirical evidence? Are its mechanisms ecologically reasonable?
        4. **Testability** – Can the hypothesis be empirically tested using current or foreseeable data, experiments, or analytical methods? Are its predictions falsifiable?
        5. **Clarity** – Is the hypothesis expressed with precision, logical structure, and readability, making its scope and implications immediately understandable?
        6. **Specificity** – Does the hypothesis clearly define its mechanisms, variables, taxa, and contexts, and make concrete, measurable predictions?
 
        ## Current hypothesis: 
          {hypothesis}
        
        ## Detailed Instructions 
        ### Method 
        - Evaluate the hypothesis comprehensively according to the six criteria above, assessing its strengths and weaknesses.
        - Then **identify only the 1–3 most critical weaknesses that must be improved**.
        - For each weakness, provide **one concise, actionable suggestion**.
         
        ### Important Guidelines (MUST be strictly followed)
        1. Follow **Core characteristics of a good hypothesis** provided above.
        2. **The hypothesis should be closely related to the research topic**.
        3. **The hypothesis must be genuinely novel, not a predictable extension of the provided text**. 
        4. **The hypothesis should be stated in a clear and concise manner**. Complex reasoning, multi-step causal pathways, or detailed theoretical justification should be placed in the Reasoning section, not embedded within the hypothesis itself.
        5. The hypothesis must be a **coherent and logically grounded conjecture**, not the aggregation of unrelated factors or inspirations.
        6. **Avoid creating overly complex or abstract hypotheses that cannot be empirically tested within reasonable experimental or observational frameworks**. Additionally, note that a hypothesis is not the same as an experimental design. Therefore, it should not attempt to specify concrete experimental procedures or numerical parameters. The pathway to verification or falsification should be clear, feasible, and should not rely on excessive or unverified assumptions.
        7. **When generating hypotheses, focus on major and direct ecological mechanisms rather than long and speculative causal chains. Avoid hypotheses that rely on multiple indirect steps or unlikely intermediary events to link cause and effect. A hypothesis should explain phenomena using the most direct and primary mechanism, not a complex chain of minor influences**.
        8. **Ecological significance. The generated hypothesis must address a question of clear and broad importance to ecology or urban biodiversity**. Avoid hypotheses that are merely novel, plausible, or taxon-specific but have limited theoretical, empirical, or applied ecological significance. Hypotheses whose outcomes would not substantially advance ecological understanding or inform biodiversity management should be avoided.
        9. **Please ensure that each generated hypothesis is specific, clearly identifying the key ecological mechanisms, variables, and contexts involved, rather than remaining abstract or overly general**.
        10. **High-quality hypotheses do not exceed 50 words**.
        
        ### Output Format
        **Strict Output**: Your response needs to follow this strict JSON format, **do not include any other text or explanation outside of the JSON or the LIST.**
        Each JSON string contains an evaluation and suggestion for the corresponding hypothesis.

        {{
         "ID": "ID for the hypothesis",
         "evaluation": "[An evaluation of the hypothesis]",
         "suggestions_for_improvement":"[Specific, actionable constructive feedback on how to improve the hypothesis]"
        }}
        """

    return prompt_content
