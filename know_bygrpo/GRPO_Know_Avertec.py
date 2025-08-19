from peft import LoraConfig, TaskType, PeftModel, get_peft_model
import re
from pathlib import Path
import sys
import os
import random
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from loguru import logger 
from pprint import pformat
from datetime import datetime
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 必须放在torh 之前 选择机器
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
import torch
import json
from datasets import Dataset
from transformers import TrainerCallback
import math
from util import global_z as G
from util import filetool as F
from util.filetool import MyLogging
from functools import partial
from trl import GRPOTrainer, GRPOConfig
import spacy
from collections import defaultdict

# 可以修改的变量
Semantic_Threshold = 0.3
Enity_Weight = 0.7

# 定义常量
Conflicting_Evidence_lable = 'Conflicting Evidence/Cherrypicking'
Conflicting_Evidence = 'A'
Not_Enough_Evidence_label = 'Not Enough Evidence'
Not_Enough_Evidence = 'B'
Consistent_label = 'Supported'
Consistent = 'C'
Inconsistent_lable = 'Refuted'
Inconsistent = 'D'
DONTKONW = 'E'
WRONG_ANSWER = 'Z'

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
# wandb_key ='6dfac2f41f879ea9141e9a159abbab52ca11ae84'

model_name = 'DeepseekR1-Distill-Llama-8B' #'DeepSeek0528-Qwen3-8B'
model_path ='/data/DeepseekR1-Distill-Llama-8B' # DeepseekR1-Distill-Llama-8B  DeepSeek-R1-Distill-Qwen-7B

# 配置日志
now = datetime.now()
process_logpath = f"/disk2/zhZH/mmfc/KNOW_log/fcknowProcess-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}.json"
MyLogging.set_logger(print_level="DEBUG", logfile_level="DEBUG", log_file= process_logpath)

# 1. 加载模型
model_Path = Path(model_path)
# model, tokenizer = load_model(base_model_path=str(model_Path))

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

deepmodel = AutoModelForCausalLM.from_pretrained(model_path,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)

# avertic 中的标签有 'Supported', 'Refuted', 冲突证据'Conflicting Evidence/Cherrypicking'用真实数据说谎, 'Not Enough Evidence'
# 阶段一 只使用'Supported', 'Refuted'的数据进行训练


class Answer():
    def __init__(self, answerdata):
       self.answer = answerdata['answer']
       self.answer_type = answerdata['answer_type']
       self.source_url = answerdata['source_url'] # 来源网站
       self.source_medium = answerdata['source_medium'] 
       self.cached_source_url = answerdata['cached_source_url'] # 缓存


class Avertic():
    def __init__(self, data):
        self.claim = data['claim']
        self.label = data['label']
        self.claim_date = data['claim_date']
        self.required_reannotation = data['required_reannotation']
        self.reporting_source= data['reporting_source'] # 首次发布声明的网站或组织，例如 Facebook、CNN。
        self.fact_checking_article = data['fact_checking_article'] # 包含claim的事实核查文章的url
        self.cached_original_claim_url = data['cached_original_claim_url'] # 原始索赔 URL 的 archive.org 链接。
        self.location_ISO_code = data['location_ISO_code'] # 声明的国家的 ISO 代码。(对检索非常有用) 
        '''
        https://www.jianshu.com/p/9911b2b4d857 ISO信息
        '''
        self.fact_checking_strategies = data['fact_checking_strategies'] # 事实核查策略集合
        self.qas = data['questions']
        self.evidence = self._evidence(self.qas)
        self.webinfor = self._formatwebinfor(self.qas)

    # 构建QA对
    def _evidence(self, qas):
        evidencelist = []
        for e in qas:
            qadict = dict()
            question = e['question']
            answers = []
            for a in e['answers']:
                answer = Answer(a)
                answers.append(answer)
            qadict = {'question':question, 'answer':answers}
            evidencelist.append(qadict)

        return evidencelist
    
    # 根据answer内容构建网络上相关信息，形成一个文档用于检索
    def _formatwebinfor(self, qas):
        webinfor = []
        answer = ''
        for e in qas:
            for a in e['answers']:
                answer = a['answer']
                webinfor.append(answer)
    
        return " ".join(webinfor)



def extract_answer(text: str) -> str:
    patterns = {
        "think": r"<think>(.*?)</think>",
        "answers": r"<option>\s*(\([ABCDE]\)|[ABCDE])\s*</option>",
        "confidences": r"<confidence>\s*(\d+)\s*</confidence>",
        "explanations": r"<explanation>(.*?)</explanation>"
    }
    bad_sen = ['<think>\nyour reasoning process\n</think>','<option>your option</option>','<confidence>your confidence</confidence>','<explanation>\nyour explanation\n</explanation>']
    
    rewards = []
    
    cleaned_completion = re.sub(bad_sen[0],"",text)
    for bad in bad_sen[1:]:
        cleaned_completion = re.sub(bad, "", cleaned_completion)

    # text_ = copy.deepcopy(text)
    results = {}
    for tag_name,pattern in patterns.items():
        match = re.search(pattern, cleaned_completion, re.DOTALL)
        # results[tag_name] = bool(match)
        
        # print(f"{tag_name}")
        # 可选：提取匹配内容
        if match:
            results[tag_name] = match.group(1).strip()
            
            
        else:
            results[tag_name] = ""
    
    # answer = re.findall(pattern_answers, text_.replace('\n', ''))
    # explanation = re.findall(pattern_explanations, text_.replace('\n', ''))[0]
    answer = results['answers']
    confidence_resonse = results['confidences']
    think = results['think']
    explanation = results['explanations']
    
    confidence = 0.0
    confdence_pattern = patterns['confidences']
    if confidence_resonse != "":
        # 判断 confidence 是否为0-10范围
        matches = re.findall(confdence_pattern, cleaned_completion)
        for num_str in matches:
            try:
                num = float(num_str)
                if num > 10.0:
                    confidence = 10.0
                elif num <0.0:
                    confidence = 0.0
                elif (0.0 <= num <= 10.0):
                    confidence = num
                else:
                    confidence = 0.0
            except ValueError:
                logger.debug(f'extract_answer confidece wrong\n {cleaned_completion}\n matches = {matches}')
                
    '''
        Conflicting_Evidence_lable = 'Conflicting Evidence/Cherrypicking'
        Conflicting_Evidence = 'A'
        Not_Enough_Evidence_label = 'Not Enough Evidence'
        Not_Enough_Evidence = 'B'
        Consistent_label = 'Supported'
        Consistent = 'C'
        Inconsistent_lable = 'Refuted'
        Inconsistent = 'D'
        DONTKONW = 'E'
        WRONG_ANSWER = 'Z'
    '''
    if 'C' in answer : # == 'don't know':
        answer = Consistent_label
    elif 'A' in answer: # == '符合事实':
        answer = Conflicting_Evidence_lable
    elif 'B' in answer:
        answer = Not_Enough_Evidence_label
    elif 'D' in answer:
        answer = Inconsistent_lable
    elif 'E' in answer:
        answer = DONTKONW
    else:
        answer = WRONG_ANSWER

    return answer, confidence, think, explanation

def check_answer(truelabel, LLManswer):
    if truelabel==LLManswer:
        return True
    if LLManswer == DONTKONW:
        return DONTKONW

    return False

def check_answer4confidence(label, LLManswer):
    if label == DONTKONW:
        return 0.0    
    if label==LLManswer:
        return 1.0

    return -1.0

'''
设置奖励函数 - 废弃
误认为是将所有prompt的结果都返回的answers
'''
def reward_firststage_history(prompts, completions, ground_truth,  **kwargs):
    """
        五个claim的问答对
        输出答案和置信度
        使用R_correct+R_punished
    """
    log_qa = dict() # 日志记录问答对
    groupsize = 5
    group_num = 1
    rewards = []
    Query = prompts[0][-1]["content"]
    logger.info(f"query : {Query}")
    # log_qa["query"] = Query
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    for i in range(0,len(extracted_responses),groupsize):
        completiongroup = extracted_responses[i:i+groupsize]
        truelabel_group= ground_truth[i:i+groupsize]
        group_num += 1
        
        reward = 0.0 # 总奖励
        r_correct = 0.0 # 正确判断的奖励
        r_wrong = 0.0 # 错误判断的惩罚
        r_confidence = 0.0 # 知识边界的奖励
        incorrect_num = 0
        dontkonw_num = 0
        # avertic 中的标签有 'Supported', 'Refuted', 冲突证据'Conflicting Evidence/Cherrypicking'用真实数据说谎, 'Not Enough Evidence'
        for comp, true_label in zip(completiongroup,truelabel_group):
            LLManswer = comp[0]
            confidence = comp[1]
            if LLManswer == WRONG_ANSWER: # 格式错误
                r_wrong += 10
                incorrect_num += 1
                continue
            label = true_label
            check = check_answer(label, LLManswer)
            if check==True: # 回答正确 True
                r_correct += 2.0  * confidence
            elif check == DONTKONW: # 回复不知道
                dontkonw_num += 1
                r_correct += 1.0 * confidence
            else: # 回答错误
                incorrect_num += 1
                r_wrong += 2.0 * confidence
        # 盲目判断的惩罚
        r_confidence = incorrect_num/(dontkonw_num + incorrect_num) 
        # 总奖励
        reward = r_correct - r_wrong - r_confidence
        # 为了让数值更显著，可以再乘以一个系数
        reward *= 10  
        for i in range(groupsize):
            rewards.append(float(reward))
        
        
    return rewards


'''
    设置奖励函数 废弃
'''
def reward_firststage(prompts, completions, ground_truth,  **kwargs):
    """
        group个回答
        输出答案和置信度
        使用R_correct+R_punished
    """
    log_qa = dict() # 日志记录问答对
    groupsize = 5
    group_num = 1
    rewards = []
    Query = prompts[0][-1]["content"]
    logger.info(f"query : {Query}")
    # log_qa["query"] = Query
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    
    for i in range(0,len(extracted_responses),groupsize):
        completiongroup = extracted_responses[i:i+groupsize]
        truelabel_group= ground_truth[i:i+groupsize]
        group_num += 1
        
        # avertic 中的标签有 'Supported', 'Refuted', 冲突证据'Conflicting Evidence/Cherrypicking'用真实数据说谎, 'Not Enough Evidence'
        for comp, true_label in zip(extracted_responses,ground_truth):
            
            reward = 0.0 # 总奖励
            r_correct = 0.0 # 正确判断的奖励
            r_wrong = 0.0 # 错误判断的惩罚
            r_confidence = 0.0 # 知识边界的奖励
            incorrect_num = 0
            dontkonw_num = 0
            
            LLManswer = comp[0]
            confidence = comp[1]
            if LLManswer == WRONG_ANSWER: # 格式错误
                r_wrong += 10
                incorrect_num += 1
                continue
            label = true_label
            check = check_answer(label, LLManswer)
            if check == True: # 回答正确
                r_correct += 2.0  * confidence
            elif check == DONTKONW: # 回复不知道
                dontkonw_num += 1
                r_correct += 1.0 * confidence
            else: # 回答错误
                incorrect_num += 1
                r_wrong += 2.0 * confidence
        # 盲目判断的惩罚
        r_confidence = incorrect_num/(dontkonw_num + incorrect_num)
        # 总奖励
        reward = r_correct - r_wrong - r_confidence
        # 为了让数值更显著，可以再乘以一个系数
        reward *= 10
        for i in range(groupsize):
            rewards.append(float(reward))
        
        
    return rewards


'''
    未用到
'''
def load_model(base_model_path: str, lora_adapter_path=None):
    """加载原始模型和分词器"""

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                      trust_remote_code=True)
    if lora_adapter_path:
        new_model = PeftModel.from_pretrained(model,
                                              lora_adapter_path,
                                              is_trainable=True)
        new_tokenizer = AutoTokenizer.from_pretrained(
            lora_adapter_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        new_model.train()
        return new_model, new_tokenizer
    # model.to(DEVICE)

    return model, tokenizer

def format_claimdate(claim_date):
    formatted_date = ''
    # 解析原始格式
    if claim_date is None:
        formatted_date = 'no_date_information'
    else:
        date_obj = datetime.strptime(claim_date, "%d-%m-%Y")
        # 格式化为中文日期
        formatted_date = date_obj.strftime("%Y年%m月%d日")
        # print(formatted_date)  # 输出: 2020年08月25日
    return formatted_date



def process_data(examples,tokenizer):
    # 使用chat模板格式化
    prompts = []
    for example in examples["prompt"]:
        formatted = tokenizer.apply_chat_template(example,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        prompts.append(formatted)

    # Tokenize处理
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "ground_truth": examples["ground_truth"]
    }
    





# def similarity_causal(prompt,explaination, ground_truth, confidence):
#     """
#         使用因果LLM评估置信度的函数 
#     """
#     # 这里可以根据explaination和ground_truth来计算置信度的得分
#     # 假设explaination是一个字符串，ground_truth是一个字符串
#     # ground_explain = ground_truth['explanation']
#     claim =getclaimformprompt(prompt)
#     evaluate_causal_prompt = G.evaluate_causal.format(text=claim, ground_truth=ground_truth)
#     evaluate_goals = cleanup_evaluate_prompt(evaluate_causal_prompt)
#     #正确的自信
#     return evaluate_goals_confidence


def similarity_yuyi(longtxt, text, entity_weight=0.7, semantic_threshold=0.5):
    """
    计算长文本与短文本的相似度，过滤噪声并验证推理脉络
    
    参数:
    longtxt (str): 长文本 ground truth
    text (str): 短文本  llm输出的explanation
    entity_weight (float): 实体匹配权重
    semantic_threshold (float): 语义相似度阈值
    
    返回:
    tuple: (相似度分数(semantic_threshold ~ 1), 相关段落, 关键句子索引)
    """
    nlp = spacy.load("en_core_web_sm")
    # 处理文本
    long_doc = nlp(longtxt)
    llm_doc = nlp(text)
    
     # 提取短文本中的实体
    llm_entities = {ent.text for ent in llm_doc.ents}
    
    # 将长文本分割为句子并计算相似度
    sentences = list(long_doc.sents)
    sentence_scores = []
    for i, sent in enumerate(sentences):
        # 1. 实体匹配分数：短文本实体在句子中出现的比例
        entity_matches = sum(1 for ent in sent.ents if ent.text in llm_entities)
        entity_score = entity_matches / max(1, len(llm_entities))
        
        # 2. 语义相似度分数：基于词向量
        semantic_similarity = llm_doc.similarity(sent)
        
        # 3. 综合分数：实体匹配与语义相似度的加权和
        combined_score = entity_weight * entity_score + (1 - entity_weight) * semantic_similarity
        
        sentence_scores.append((i, combined_score, entity_score, semantic_similarity))
    
    # 筛选出高于阈值的句子
    relevant_sentences = [
        (i, score) for i, score, _, _ in sentence_scores 
        if score >= semantic_threshold
    ]
    
    if not relevant_sentences:
        # 若没有高于阈值的句子，返回最高分数的句子
        top_sentence = max(sentence_scores, key=lambda x: x[1])
        # return (top_sentence[1], str(sentences[top_sentence[0]]), [top_sentence[0]])
        return top_sentence[1]
    
    
    # 提取相关段落（连续的相关句子）
    relevant_sentences.sort(key=lambda x: x[0])
    start_idx = relevant_sentences[0][0]
    end_idx = relevant_sentences[-1][0]
    
    # 计算整体相似度分数（相关句子的平均分数）
    avg_score = sum(score for _, score in relevant_sentences) / len(relevant_sentences)
    
    # 提取相关段落
    relevant_paragraph = " ".join([str(sentences[i]) for i in range(start_idx, end_idx + 1)])
    
    # return (avg_score, relevant_paragraph, [i for i, _ in relevant_sentences])
    return avg_score # (semantic_threshold ~ 1)


def reward_Confidence(prompts, completions, ground_truth, **kwargs):
    """
        奖励函数：奖励正确的自信和惩罚错误的自信
        C = confidence, [0-10]
        S = similarity of think and article [0-1]
        IsCorrect * (C/S)
    """
    q = prompts[0][-1]["content"]
    
    rewards = []
    for i in range(len(completions)):
        completion = completions[i][0]["content"]
        label_predict, c, think, explaination = extract_answer(completion)
        label = ground_truth[i]["true_label"]
        check = check_answer4confidence(label, label_predict) # 1.0 / -1.0
        true_article = ground_truth[i]["expalnation_content"]
        s = similarity_yuyi(true_article, explaination,Enity_Weight,Semantic_Threshold)  # (0.5-1)
        score_confidence = 0.0
        # if s  < Semantic_Threshold: # 语义相似度小于Semantic_Threshold时  错误的推理
        #     # sc = 0.0
        #     if check == 0.0: # 回答不知道
        #         score_confidence = 0.99
        #     elif check == 1.0: # 回答正确 但推理错误 = 错误的自信
        #         score_confidence = -0.9
        #     else: # 回答错误 且推理错误 = 正确的自信
        #         score_confidence = 0.01
        # else:
        #     sc = c / (s * 10) # c=(0-10)，s=(0.5-1)
        #     if check==1.0:
        #         score_confidence = check * sc 
        #     else: # 正确的推理 但回答错误
        #         score_confidence = -1.0 * sc 
        if c>1.0:
            c = c/10.0
        score_confidence = 1+(c-s)*(s-c)
        rewards.append(score_confidence * 10.0)
        
        # 记录日志
        rand = random.random()
        if rand<0.1:
            full_q_a = dict()
            full_q_a["name"] = 'confidence_score'
            full_q_a["fullprompt"] = q
            full_q_a["Response"] = completion
            full_q_a["truelabel"] = label
            full_q_a["label_predict"] = label_predict
            full_q_a["check"] = check
            full_q_a["confidence"] = c
            full_q_a["similarity"] = s            
            full_q_a['confidence_reward'] = score_confidence
            full_q_a['explaination'] = explaination
            logger.info(f"reward_Confidence:\n{pformat(full_q_a)}")
        
    return rewards # 范围在0-10

'''
R_correct 
R(explain, causal_graph,confidence) 错误自信和错误的不自信
'''
# 一组8个回复 相对的奖励值
def reward_corret(prompts, completions, ground_truth, **kwargs):
    """
    奖励函数：奖励正确的回答
    reward范围在-10~10
    """
    q = prompts[0][-1]["content"]
          
        
    rewardslist = []
    for i in range(len(completions)):
        response = completions[i][0]["content"]
        label_predict, confidence, think, explaination= extract_answer(response)
        label = ground_truth[i]["true_label"]
        claim_en = ground_truth[i]["claim"]
        check = check_answer(label, label_predict)
        rewards = 0.0
        if check == True: # 回答正确
            rewards = 2.0  # 正确回答奖励1.0
        elif check == DONTKONW: # 回复不知道
            rewards = 0.0
        else: # 回答错误 或格式错误
            rewards = -1.0  # 错误回答惩罚confidence
        rewardslist.append(10.0* rewards)
        
        # 记录日志
        rand = random.random()
        if rand<0.1:
            full_q_a = dict()
            full_q_a["name"] = 'correct_reward'
            full_q_a["fullprompt"] = q
            full_q_a["truelabel"] = label
            full_q_a["claim_en"] = claim_en
            full_q_a["Response"] = response
            full_q_a["label_predict"] = label_predict
            full_q_a["confidence"] = confidence
            full_q_a['think'] = think
            full_q_a['explaination'] = explaination
            full_q_a['correct_reward'] = rewards*10.0
            logger.info(f"reward_corret\n{pformat(full_q_a)}")
        
    return rewardslist  # 范围在-10~10

def format_correct(prompts, completions, ground_truth, **kwargs):
    
    # pattern_think = r"<think>(.*?)</think>"
    # pattern_answers = r"<answer>(.*?)</answer>"
    # pattern_confidences = r"<confidence>(.*?)</confidence>"
    # pattern_explanations = r"<explanation>(.*?)</explanation>"
    # answer = re.findall(pattern_answers, completion.replace('\n', ''))
    # confidence = re.findall(pattern_confidences, completion.replace('\n', ''))
    # explanation = re.findall(pattern_explanations, completion.replace('\n', ''))[0]
    # think = re.findall(pattern_think, completion.replace('\n', ''))[0]   
    
    patterns = {
        "think": r"<think>(.*?)</think>",
        "answers": r"<option>\s*(\([ABCDE]\)|[ABCDE])\s*</option>",
        "confidences": r"<confidence>\s*(\d+)\s*</confidence>",
        "explanations": r"<explanation>(.*?)</explanation>"
    }
    bad_sen = ['<think>\nyour reasoning process\n</think>','<option>your option</option>','<confidence>your confidence</confidence>','<explanation>\nyour explanation\n</explanation>']
    
    rewards = []
    
    for i in range(len(completions)):
        full_q_a = dict()
        completion = completions[i][0]["content"]
        cleaned_completion = re.sub(bad_sen[0],"",completion)
        for bad in bad_sen[1:]:
            cleaned_completion = re.sub(bad, "", cleaned_completion)
        results = {}
        reward = 0.0
        
        for tag_name, pattern in patterns.items():
            # 使用 re.search() 检查是否存在匹配
            match = re.search(pattern, cleaned_completion, re.DOTALL | re.IGNORECASE)
            results[tag_name] = bool(match)
            
            if match:
                tag_content = match.group(1).strip()
                if tag_name == 'think':
                    reward += 0.25
                if tag_name =='explanations':
                    reward += (500.0-len(tag_content))/(500.0*4.0)  # 范围<=0.25
                    full_q_a['exp_len_reward'] = (500.0-len(tag_content))/(500.0*4.0)
                elif tag_name == 'answers':
                    if 'A'in tag_content or 'B' in tag_content or 'C' in tag_content or 'D'in tag_content or 'E' in tag_content:
                        reward += 0.25
                    else:
                        reward += -0.25
                elif tag_name == 'confidences':
                    matches = re.findall(pattern, cleaned_completion)
                    conf_flag = 0
                    for num_str in matches:
                        try:
                            num = float(num_str)
                            if 0.0 <= num <= 10.0:
                                reward += 0.25  # 返回第一个不满足条件的数字
                                conf_flag = 1
                                break
                        except ValueError:
                            logger.debug(f'format_correct-match confidece wrong {cleaned_completion}, \nextract_matches = {matches}')
                            continue
                    if conf_flag == 0:
                        reward += -0.25
                    full_q_a[tag_name] = tag_content
            else:
                reward -= 0.25
                full_q_a[tag_name] = "no match"
            
            
        
        rand = random.random()
        if rand<0.1:
            # 记录日志
            q = prompts[0][-1]["content"]
            full_q_a["name"] = 'format_correct'
            full_q_a["fullprompt"] = q
            full_q_a["Response"] = completion
            full_q_a["cleaned_completion"] = cleaned_completion
            full_q_a['format_correct'] = reward*10.0
            logger.info(f"format_correct\n {pformat(full_q_a)}")
                    
        rewards.append(reward * 10.0)
    
    return rewards


if __name__ == "__main__":
    # 训练集、测试集的路径
    AVERTIC_trainpath = Path('/data2/zhZH/mmfc/data/AVERTIC/Train1k_CNSR10-id_prompt.json')#Path('/data2/zhZH/mmfc/data/AVERTIC/Traindev_CNSR10-id_prompt.json')
    AVERTIC_testpath = '/data2/zhZH/mmfc/data/AVERTIC/Evaldev_CNSR10-id_prompt.json'
    
    # 保存的模型和模型训练参数变化
    saved_path = Path("/disk2/zhZH/mmfc/model")
    saved_model_name = saved_path / f"KNOW-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}"
    logpath = f"/disk2/zhZH/mmfc/runs/KNOWTensor-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}"
    
    
    # 2.构建阶段一的prompt 已构建存储完毕
    # Know_train_data, Know_test_data = F.read_jsonlist_file4Know(AVERTIC_trainpath,0.8)
    # train_prompts = knowstage_generatePrompt(Know_train_data)
    # test_prompts = knowstage_generatePrompt(Know_test_data)
    
    train_prompts = F.read_jsonlist_file(AVERTIC_trainpath)
    train_dataset = Dataset.from_list(train_prompts)
    train_dataset = train_dataset.map(process_data, batched=True, fn_kwargs={"tokenizer": tokenizer})
    
    test_prompts = F.read_jsonlist_file(AVERTIC_testpath)
    test_dataset = Dataset.from_list(test_prompts)
    test_dataset = train_dataset.map(process_data, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # 3. 配置GRPO
    # print(torch.cuda.is_available())
    # trainer = first_stage_train(model,tokenizer,train_dataset, test_dataset,saved_model_name) # DeepseekR1-Distill-Llama-8B
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
            "down_proj"
        ],
        # target_modules=["q_proj", "k_proj", "v_proj", "down_proj"],
        # inference_mode=False,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
    )
    training_args = GRPOConfig(
        # 这里面包括很多RL训练的参数
        output_dir=saved_model_name,
        do_train = True,
        learning_rate=1e-5,
        logging_steps=1,
        max_steps=1500, # 1500
        save_steps=500,  # 保存检查点之间的步数,500
        eval_steps=500, # 500
        eval_strategy="steps",
        gradient_accumulation_steps=4,
        max_completion_length=1000, # 最大生成长度
        per_device_train_batch_size=8,
        num_generations=8, # 一组生成的回复
        lr_scheduler_type="cosine_with_min_lr", 
        max_grad_norm=1.0,  # 梯度裁剪
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        bf16=True,  # 启用 FP16 混合精度
        temperature=0.7,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=22,
        report_to="tensorboard",
        logging_dir=logpath,     # 日志保存目录
        # 其他可调参数，比如 batch_size, kl_coeff, etc...
    )

    trainer = GRPOTrainer(
        model=deepmodel, 
        processing_class=tokenizer,
        reward_funcs=[format_correct, reward_corret,reward_Confidence],  # 也可以传列表
        args=training_args,
        train_dataset=train_dataset,  # 要有"prompt"列
        eval_dataset=test_dataset,
        peft_config=peft_config,  # 挂LoRa
    )

    # 4. 开始训练
    train_result = trainer.train()
    # 5. 保存训练后的模型
    deepmodel.save_pretrained(saved_model_name)
    # trainer.save_model(saved_model_name)
    tokenizer.save_pretrained(saved_model_name)

#     run = wandb.init(
#     # Set the wandb entity where your project will be logged (generally your team name).
#     entity="my-awesome-team-name",
#     # Set the wandb project where this run will be logged.
#     project="my-awesome-project",
#     # Track hyperparameters and run metadata.
#     name = 'first_stage',
#     config={
#         "learning_rate": 0.02,
#         "architecture": "CNN",
#         "dataset": "CIFAR-100",
#         "epochs": 10,
#     },
# )

    ''' 阶段二 生成推理链 
    # 1. 加载模型
    model_Path = Path(saved_model_name)
    model, tokenizer = load_model(base_model_path=str(model_Path))
    saved_path2 = Path("/data2/zhZH/mmfc/model/stage2")
    saved_model_name2 = saved_path2 / f"two-{saved_model_name}-{time.strftime('%y-%m-%d',time.localtime)}"

    Know_train_data, Know_test_data = F.read_jsonlist_file4Reason(AVERTIC_trainpath, AVERTIC_testpath,0.8)
    
    '''



