'''
    测试1：自己回答 or 生成子query
    模型：qwen + know_lora  vs qwen
    llama3.1 +know_lora vs llama3.1
    
    指标：准确度和置信度，
    
    存每个案例的结果，
    testanswer.append({
                "test_id":test_id, 'type': 'answerself', "predict":{'think':xx, 'answer':answer, 'confidence':confidence, 'explanation':explanation},"truthlabel":true_label, 'url_expalnation':url_expalnation
                }) 

'''
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
import numpy as np
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
from FlagEmbedding import FlagReranker
import math
from util import global_z as G
from util import filetool as F
from util import analysistool as An
from util.filetool import MyLogging
from functools import partial
from trl import GRPOTrainer, GRPOConfig
import spacy
from collections import defaultdict


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
reranker = FlagReranker('/data/bge-reranker-v2-m3', use_fp16=True)

# 定义常量
WRONG = '0'
CORRECT = '1'
DONTKONW = '2'
WRONG_ANSWER = '4'

webbase_test_path = '/data2/zhZH/mmfc/data/AVERTIC/web/devwebinfor' # web信息在的文件夹


# 可以改的变量
Paranum = 3 # 选取的web段落个数
Causalgraph_Groupnum = 8

'''
    如果要修改 model
    1. model_name 参数要改，对应的是日志
    2.model tokenizer要改 对应的是使用的模型
    3.generate_response 的函数要改，对应的是model的生成回复的参数和模型
'''


# 启用的模型
base_model_name = 'DeepseekR1-Distill-Llama-8B'
basemodel4llama_path = '/data/DeepseekR1-Distill-Llama-8B'
lora4llama_path = '/disk2/zhZH/mmfc/model/KNOW-DeepseekR1-Distill-Llama-8B-2025-07-31-19-24/checkpoint-500'#'/disk2/zhZH/mmfc/model/KNOW-DeepseekR1-Distill-Llama-8B-2025-07-08-11-03/checkpoint-1500'
know_model_name = 'DeepSeek0528-Qwen3-8B-KNOW'
causal_lora_path = '/disk2/zhZH/causalsft-DeepSeek0528-Qwen3-8B-2025-06-24-09-17/checkpoint-730'
causal_model_name = 'DeepSeek0528-Qwen3-8B-Casual'

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
        new_model.to(DEVICE)
        return new_model, new_tokenizer
    model.to(DEVICE)

    return model, tokenizer

base_model,base_tokenizer = None, None # load_model(base_model_path=basemodel_path)
know_model, know_tokenizer = load_model(base_model_path=basemodel4llama_path, lora_adapter_path=lora4llama_path) # None, None #
causal_model, causal_tokenizer = None, None #load_model(base_model_path=basemodel_path,lora_adapter_path=causal_lora_path)

CONFIG = {
    "model_name": "",
    "MT_stage1":[],
    "MT_stage2": [],
    'stage1_maxlen':1000,
    'stage2_maxlen':1000,
    'generate_config_stage1': {},
    'generate_config_stage2': {}
}



# 基础模型
def generate_response_satge1( full_prompt):
    model, tokenizer = CONFIG['MT_stage1']
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CONFIG['stage1_maxlen'],
    ).to(DEVICE)
    inputs =  {k: v.to(DEVICE) for k, v in inputs.items()} # 修改 变量都移动到DEVICE里面
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(**inputs, **CONFIG['generate_config_stage1'])
    
    # 解码并提取回答部分
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant: ")[-1].strip()
    
# 获得规划的回复 使用知识边界模型
def generate_response_satge2(full_prompt):
    model, tokenizer = CONFIG['MT_stage2']
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CONFIG['stage2_maxlen'],
    ).to(DEVICE)
    inputs =  {k: v.to(DEVICE) for k, v in inputs.items()} # 修改 变量都移动到DEVICE里面
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(**inputs, **CONFIG['generate_config_stage2'])

    # 解码并提取回答部分
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant: ")[-1].strip()


# 初始化函数映射
def init_model(model_name):
    generate_config_base = {}
    # {
    #         "max_new_tokens": 2048,
    #         "temperature": 0.7,
    #         "top_p": 0.9,
    #         "do_sample": True,
    #         "repetition_penalty": 1.1,
    #         "pad_token_id": base_tokenizer.pad_token_id,
    # }
    # generate_config_know = {}
    generate_config_causal = {}
    generate_config_know = {
        "max_new_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "pad_token_id": know_tokenizer.pad_token_id,
    }
    # generate_config_causal = {
    #         "max_new_tokens": 4096,
    #         "temperature": 0.7,
    #         "top_p": 0.9,
    #         "do_sample": True,
    #         "repetition_penalty": 1.1,
    #         "pad_token_id": causal_tokenizer.pad_token_id,
    # }
    mappings = {
        "basemodel": ([base_model,base_tokenizer],[base_model,base_tokenizer],2048,2048,generate_config_base,generate_config_base),
        "know4basellama3": ([know_model,know_tokenizer],[know_model,know_tokenizer],1000,1000, generate_config_know,generate_config_know)
    }
    
    if model_name in mappings:
        CONFIG["model_name"] = model_name
        CONFIG["MT_stage1"], CONFIG["MT_stage2"],CONFIG["stage1_maxlen"], CONFIG["stage2_maxlen"],CONFIG["generate_config_stage1"], CONFIG["generate_config_stage2"] = mappings[model_name]
        logger.info(f"已切换到 {model_name}")
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

        
def top_n_indices(arr, n):
    """返回浮点数组中前n个最大值的索引"""
    if n <= 0:
        return []
    # 获取排序后的索引（升序）
    indices = np.argsort(arr)
    # 返回最后n个索引（即最大的n个）
    return indices[-n:][::-1]  # [::-1] 用于反转顺序，使最大的在前

# 根据子问题列表，收集语义最相关的webinfor/网页的一个段落
# 段落个数有没有定义 ??
# 返回[[query, webcontentlist],[]...]
def collectwebinfor(querylist, test_id,paranum=1):
    webtestpath = Path(webbase_test_path)/f'testid_{test_id}_1.0.json'
    web_testdata = F.readonjson(webtestpath)# .json文件里面是巨大的json数据，key=has_query,item=[]
    web_hasquery = []
    webcontent_hasquery = []
    returnweblist = []
    for hasquery,webjsonlist in web_testdata.items():
        web_hasquery.append(hasquery)
        webcontent = []
        for webjson in webjsonlist:
            onewebpage = webjson['url2text']
            for webpargrapg in onewebpage:
                webcontent.append(webpargrapg)# 所有paragraph都放到一个数组里
        if len(webcontent) != 0:
            webcontent_hasquery.append(webcontent) 
    
    # 分解出来的子query列表
    for index in range(len(querylist)):
        query = querylist[index]
        caculate = []
        # 和子query最相似的has_query
        for hasquery in web_hasquery:
            onepair = [query, hasquery]
            caculate.append(onepair)
        scores = reranker.compute_score(caculate)
        maxindex = scores.index(max(scores)) 
        
        # 和子query最相似的网页内容
        webcontent = webcontent_hasquery[maxindex] 
        caculate_webcontent = []
        for para in webcontent:
            onepair = [query, para]
            caculate_webcontent.append(onepair)
        para_scores = reranker.compute_score(caculate_webcontent)
        maxindexs = top_n_indices(para_scores,paranum)#scores.index(max(para_scores)) # 和子query最相似的has_query
        # 前Paranum个最相似的句子
        paragraph = ''
        for i in maxindexs:
            paragraph = f'{paragraph}{webcontent[i]}\n'# 和子query最相似的网页段落
        
        qw = [query,paragraph]
        returnweblist.append(qw)
    
    return returnweblist


def comprise_webinfor(webinfor): # 将子query的列表组合成一段话webinfor= [[q,w],[q,w],...]
    webarticle = ''
    for qw in webinfor:
        q = qw[0]
        w = qw[1]
        webarticle= f'{webarticle}Question:{q}, Answer:{w} \n'    
    return webarticle

# -------------------------------------------------
# -------------------------------------------------
# -----------------causal--------------------------------
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def merge_ner_entities(ner_results: List[Dict], min_score: float = 0.85) -> List[str]:
    merged_entities = []
    current_entity = ""
    current_type = None
    last_index = -1

    for entry in ner_results:
        entity_tag = entry['entity']  # e.g., 'B-PER'
        entity_type = entity_tag.split('-')[-1]  # 'PER'
        word = entry['word']
        score = entry['score']
        index = entry['index']

        if score < min_score:
            continue  # filter out low-confidence predictions

        if entity_tag.startswith('B') or (last_index != index - 1):
            if current_entity:
                merged_entities.append(current_entity.strip())
            current_entity = word
            current_type = entity_type
        elif entity_tag.startswith('I') and entity_type == current_type:
            # Handle wordpiece tokens like "##land"
            if word.startswith("##"):
                current_entity += word[2:]
            else:
                current_entity += " " + word
        else:
            if current_entity:
                merged_entities.append(current_entity.strip())
            current_entity = ""
            current_type = None

        last_index = index

    # Add the last collected entity
    if current_entity:
        merged_entities.append(current_entity.strip())

    # Remove duplicates (case-insensitive)
    final_entities = []
    seen = set()
    for ent in merged_entities:
        ent_lower = ent.lower()
        if ent_lower not in seen:
            seen.add(ent_lower)
            final_entities.append(ent)

    return final_entities


'''
替换 E中的实体，
同类型（entity type 相同）
任务无关、与原实体因果无关（例如你在问“中国”，就不要用“北京”或“中共”作替代）
'''
# 替代实体库（可扩展）
ENTITY_REPLACEMENT_POOL = {
    "PER": ["Jeff Bezos", "Taylor Swift", "Steve Jobs", "Angela Merkel"],
    "LOC": ["Paris", "Tokyo", "Cairo", "Mexico City"],
    "ORG": ["Google", "UNESCO", "Greenpeace", "Netflix"],
    "GPE": ["Canada", "Brazil", "Sweden", "India"],
    "EVENT": ["Olympics", "World Cup", "Comic-Con", "Davos Forum"]
}

def find_counterfactual_entity(entity_type: str, original_entity: str) -> str:
    pool = ENTITY_REPLACEMENT_POOL.get(entity_type, [])
    if not pool:
        return None
    candidates = [e for e in pool if e.lower() != original_entity.lower()]
    if not candidates:
        return None
    return random.choice(candidates)

# 返回值是str  dict
def replace_entities_in_context(context: str, ner_results: List[Dict], min_score=0.85):
    new_context = context
    replaced = {}

    current_entity_tokens = []
    current_type = None

    def process_entity(entity_tokens, entity_type):
        entity_text = " ".join(entity_tokens)
        replacement = find_counterfactual_entity(entity_type, entity_text)
        if replacement:
            pattern = re.compile(rf'\b{re.escape(entity_text)}\b', flags=re.IGNORECASE)
            return entity_text, replacement, pattern
        return None, None, None

    for i, entry in enumerate(ner_results):
        entity_tag = entry["entity"]
        score = entry["score"]
        word = entry["word"]

        if score < min_score:
            continue

        prefix, label = entity_tag.split("-")  # e.g., 'B-PER' → ('B', 'PER')

        if prefix == "B":
            # 如果有正在处理的实体，先处理掉
            if current_entity_tokens:
                old_ent, new_ent, pattern = process_entity(current_entity_tokens, current_type)
                if old_ent and new_ent:
                    new_context = pattern.sub(new_ent, new_context)
                    replaced[old_ent] = new_ent
            # 开始新的实体
            current_entity_tokens = [word]
            current_type = label
        elif prefix == "I" and label == current_type:
            if word.startswith("##"):
                current_entity_tokens[-1] += word[2:]
            else:
                current_entity_tokens.append(word)
        else:
            # 不匹配或非法结构，强制结尾
            if current_entity_tokens:
                old_ent, new_ent, pattern = process_entity(current_entity_tokens, current_type)
                if old_ent and new_ent:
                    new_context = pattern.sub(new_ent, new_context)
                    replaced[old_ent] = new_ent
            current_entity_tokens = []
            current_type = None

    # 最后一个实体处理
    if current_entity_tokens:
        old_ent, new_ent, pattern = process_entity(current_entity_tokens, current_type)
        if old_ent and new_ent:
            new_context = pattern.sub(new_ent, new_context)
            replaced[old_ent] = new_ent

    return new_context, replaced


# 命名实体识别，找到key fact
def new_E(claim,webinfor):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("/data/bert-base-NER")

    # Create NER pipeline
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none")

    # Run NER
    ner_results = ner_pipeline(claim)
    
    new_context, replacements = replace_entities_in_context(webinfor, ner_results)
    
    return new_context, replacements 



# 计算直接因果效应
# causalgraph = {'First':xx, 'Second':xx, 'Third':xx, 'Forth':xx, 'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
def nde(E_star, claim, causalgraph):
    graph = causalgraph['graph']
    type = causalgraph['type']
    expression = causalgraph['expression']
    caculate = causalgraph['caculate']
    causal_answer = causalgraph['causal_answer'] # 选择的分类
    
    user_prompt = G.nde_template_2Classification.format(graph = graph,claim=claim, type = type, expression = expression,information=E_star )
    # prompt = {
    #         "prompt": [
    #             {
    #                 "role": "system",
    #                 "content": G.SYSTEM_PROMPT_ace
    #             },
    #             {
    #                 "role": "user",
    #                 "content": user_prompt
    #             },
    #         ]
    #     }
    prompt = f'{G.SYSTEM_PROMPT_ace}\n{user_prompt}'
    
    nderesponse = generate_response_satge2(prompt)
    ndeanswer,is_match = An.analysis_nde(nderesponse) # {'causal_answer':xx,'check_answer':xx,'explanation':xx}
    
    logger.info(f'nde answer: {ndeanswer}')
    nde = float(causal_answer) - float(ndeanswer['causal_answer'])
    
    return nde

# 计算间接因果效应，也就是根据新的E 重新生成graph和答案
# causalgraph = {'graph':xx, 'type':xx, 'expression':xx, 'caculate':xx, 'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
def tie(E_star, claim, causalgraph):
    graph = causalgraph['graph']
    type = causalgraph['type']
    expression = causalgraph['expression']
    caculate = causalgraph['caculate']
    causal_answer = causalgraph['causal_answer'] # 选择的分类
    
    causalprompt = An.generateprompt_causal(claim,E_star)
    causalanswer = generate_response_satge2(causalprompt)
    tieanswer, ismatch = An.analysis_answer_causal(causalanswer) # {'graph':xx, 'type':xx, 'expression':xx, 'caculate':xx, 'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
    
    tie = float(causal_answer) - float(tieanswer['causal_answer'])
    
    return tie


# 使用分析计算 选择最合适的causal_graph
# 输入{'graph':xx, 'type':xx, 'expression':xx, 'caculate':xx, 'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
def select_causalgraph(causalgraphs,webinfor,claim,claim_pub):
    causalgraph_wupian = dict()
    
    E_star,keyanti_fact = new_E(claim,webinfor)
    logger.info(f'select_causalgraph \nclaim={claim}\n(key_fact: anti_fact) = {keyanti_fact}\nwebinfor={webinfor}\nE_star= {E_star}')
    
    ACEs = -1
    for causalgraph in causalgraphs:
        NDE = nde(E_star, claim, causalgraph)
        TIE = tie(E_star, claim, causalgraph)
        ACE = NDE + TIE
        if ACE > ACEs:
            ACEs = ACE
            causalgraph_wupian = causalgraph
        logger.info(f'ACE = {ACE}, NDE= {NDE}, TIE={TIE}')
        
    logger.info(f'ACE_MAX = {ACEs}, causalgraph_wupian= {causalgraph_wupian}')
    
    return causalgraph_wupian

# -----------------causal--------------------------
# -------------------------------------------------
# -------------------------------------------------

if __name__ == "__main__":
    # 配置模型和调用的函数
    init_model("know4basellama3") # 可选项：knowcausal，know，casual
    
    # 配置日志
    now = datetime.now()
    logpath = f"/disk2/zhZH/mmfc/test_log_Averitec/TestProcess-NewReward-{CONFIG['model_name']}-{now.strftime('%Y-%m-%d-%H-%M')}.log"
    MyLogging.set_logger(print_level="INFO", logfile_level="DEBUG",log_file=logpath)
    write_answer_basepath = '/disk2/zhZH/mmfc/test_log_Averitec'
    write_answer_path = Path(write_answer_basepath)/f"TestAnswer-NewReward-{CONFIG['model_name']}-{now.strftime('%Y-%m-%d-%H-%M')}.json"
    
    testarticle_path = '/data2/zhZH/mmfc/data/AVERTIC/dev_CNSR10-id_prompt.json'
    testdatas = F.read_jsonlist_file(testarticle_path)
    testanswer = []
    querylist = []
    ismatch = False
    
    # 格式 test_id: , type: answerself, claim:"", explanation, confidence
    # 格式 test_id: ,type: search, claim:"", query:{stepnum: 步骤个数，sub_querylist: [检索的sub querys]}，answer:{option, explanation, confidence}
    for testarticle in testdatas[:50]:
        groundtruth = testarticle['ground_truth']
        test_ids = groundtruth['test_id'].split('-')
        test_id = int(test_ids[1])
        
        logger.info(f"========START test_id= {test_id}===========")
        claim_en = groundtruth['claim_en']
        claim = groundtruth['claim']
        claim_pub = claim_en.replace(claim,'')
        true_label = groundtruth['true_label']
        url_expalnation = groundtruth['expalnation_content']
        plan_question = An.generateprompt_answerself_4(claim_en)
        response = generate_response_satge1(plan_question) # 第一阶段的回复
        selfanswer = An.analysis_answer_self(response)
        logger.info(f'plan_question {plan_question}')
        attempt = 0
        while selfanswer['answer'] == "" and attempt <=3: # 回答格式错误
            two_plan_question = plan_question+ '''\nplease follow the required answer format.'
<think>your reasoning process</think>
<answer>choose A, B, C, D or E</answer>
<confidence>your confidence number</confidence>
<explanation>your explanation about current answer</explanation>'''
            response = generate_response_satge1(two_plan_question) # 第一阶段的回复
            selfanswer = An.analysis_answer_self(response) # 
            attempt += 1
        logger.info(f"plan_response:{response}  \n plan_analysis : {selfanswer}")
        # 自己回答的结果
        logger.info(f"===========self answer: {test_id}, ANSWER follows===========")
        
        # 计算准确率的，全部记录 之后再算吧。 selfanswer = {answer, expalnation,cofidence}
        now_save = datetime.now()
        
        save_time = now_save.strftime("%Y-%m-%d %H:%M:%S")
        testanswer.append({
        "test_id":test_id, 'type': 'answerself', "predict":selfanswer,"truthlabel":true_label, 'url_expalnation':url_expalnation, 'querylist':'','date':save_time
        })
            
            
        logger.info(f"test_id: {test_id}, type: answerself, predict:{selfanswer},truthlabel:{true_label}, url_expalnation:{url_expalnation},date:{save_time}")
        
        #  selfanswer = {answer, expalnation,cofidence}
        # 计算准确率的，要不先全部记录 之后再算准确率吧。
        
        # 记录答案
        
        logger.info(f"save: {testanswer}\npath:{write_answer_path}")
        F.writetestdata(testanswer,write_answer_path)
        testanswer = []
    F.writetestdata(testanswer,write_answer_path)