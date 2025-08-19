
SYSTEM_PROMPT_answerself_4='''Do you have enough knowledge to solve check the faculty of the claim, 
Guidance: Address this problem by following the steps below: 
Step 1 – Reasoning with Existing Knowledge:
Use your internal knowledge (from training data or general world knowledge) to analyze the claim. If you have sufficient information, explain your reasoning clearly and state whether the claim is True, False, or Unverifiable.
If you have enough confidence(range 1-10. The more confident I get, the closer your confidence score is to 10.) to solve this, please give me your answer and your explanation.

Step 2 – Answer in the following format:
Your answer should choose option in (A) Conflicting Evidence/Cherrypicking. (B) Not Enough Evidence. (C) Consistent with the facts. (D) Inconsistent with the facts. (E) I don't know.
Your explanation should contain key items in claim, and be directly and succinctly.
<think>
your reasoning process
</think>
<answer>your option</answer>
<confidence>your confidence</confidence>
<explanation>
your explanation
</explanation>
'''

Graph_generation = '''

You are an expert in fact verification graph construction. 
Given the Claim and Evidence below, follow the 5-step extraction process to output a JSON-formatted Claim–Evidence Semantic Graph.

Claim: "{CLAIM}"; Evidence: "{EVIDENCE}"
Outcome: "{LABEL LIST}"

Rules:
- Identify Claim Nodes, Evidence Nodes, and merge them if they refer to the same meaning. Outcome Nodes for each possible label.
- Extract Internal Linking Edges within Claim and within Evidence.
- Extract Claim–Evidence Linking Edges with relation type and score.
- Extract Claim–Outcome Edge with probability.
- Output only valid JSON, no explanations.'''

Fact_ver_prompt = '''You are an expert in probabilistic fact verification.
Given the Claim, Evidence, and the Claim–Evidence Semantic Graph (JSON), follow the 5-step reasoning process strictly.

Claim: "{CLAIM}"; Evidence: "{EVIDENCE}"

Graph: {JSON_GRAPH}

Steps:
1. Formalize the Claim as a probabilistic formula (e.g., P(A|context)).

2. Gather all relevant Claim–Evidence Linking Edges and their probabilities.

3. Deduce and calculate the probability: combine probabilities based on graph structure, adjusting for internal linking and conflict edges.

4. Classify the claim into SUPPORTS, REFUTES, NOT ENOUGH INFO, or CONFLICT EVIDENCE based on probability thresholds.

5. Generate an explanation summarizing the reasoning and key evidence.

Output in JSON format:
'''