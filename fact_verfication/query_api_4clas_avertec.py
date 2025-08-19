'''
    读取 query_self的输出
    根据规划的querylist
    检索文档+causal推理
'''
import sys
import os
import random
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 必须放在torh 之前 选择机器
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
sys.path.append('/data2/zhZH/mmfc/stage2_byapi')  # add current terminal path to sys.path

import torch
import json
from datasets import Dataset
from transformers import TrainerCallback
from FlagEmbedding import FlagReranker
import math
from util import global_z as G
from util import filetool as F
from util import analysistool as An
from datetime import datetime
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
from transformers import pipeline
from typing import List, Dict
from util.filetool import MyLogging
from pathlib import Path
import re
from peft import  PeftModel
from stage2_byapi import apiHUOSHAN as huoshan
import numpy as np

# 可以改变地参数
Fact_num = 6 # 找到claim的实体个数
Paranum = 3 # 选取的web段落个数(历史方法使用)
Context_window = 5 # 选取的web段落个数(find_best_paragraph方法使用)
Causalgraph_Groupnum = 3  # 一次性生成8个因果图candidates
Candidat_num = 3
DEVICE="cuda" if torch.cuda.is_available() else "cpu"



# 配置日志
now = datetime.now()
modellsit = ['deepseek-r1HS','gpt-3.5-turbo','gpt-4o']
model_name = modellsit[2]

logpath = f"/disk2/zhZH/mmfc/test_log_Averitec/TestProcess-allcausalsub-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}.log"
MyLogging.set_logger(print_level="INFO", logfile_level="DEBUG",log_file=logpath)
write_answer_basepath = '/disk2/zhZH/mmfc/test_log_Averitec'
write_answer_path = Path(write_answer_basepath)/f"TestAnswer-allcausalsub-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}.json"
write_allanswer_path = Path(write_answer_basepath)/f"TestAnswerFinal-allcausalsub-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}.json"
write_graph_path = Path(write_answer_basepath)/f"TestGraph-allcausalsub-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}.json"


claim_enpath = '/data2/zhZH/mmfc/data/AVERTIC/dev_CNSR10-id_prompt.json'
test_claim_dict, test_label_dict, test_exp_dict = F.read_jsonlist_file4Search(claim_enpath) # {test_id : content}

# -----------------------------------------------------------
# -----------------------------------------------------------
# ----------------------查询相关语句----------------------------

# Avertic 在 An里面

# 将子query的列表组合成一段话webinfor= [[q,w],[q,w],...]
def comprise_webinfor(webinfor): 
    webarticle = ''
    for qw in webinfor:
        
        q = qw[0]
        w = qw[1]
        # if len(w)>100:
        #     continue
        webarticle= f'{webarticle}Question:{q}, Answer:{w} \n'    
    return webarticle

# -----------------------查询相关语句---------------------
# -----------------------------------------------------
# -----------------------------------------------------
    
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ----------------生成反事实实体，并验证合理性-------------------------------

# 替换实体
def merge_ner_entities(ner_results: List[Dict], min_score: float = 0.5) -> List[str]:
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

from util.analysistool import EntityTypeValidator
def generate_candidates( v_j, entity_type, max_attempts):
    # 结合LLM生成
    create_antiEntity_prompt = G.create_antiEntity_template.format(v_j=v_j, entity_type=entity_type,num = max_attempts)
    anticandidates_response = huoshan.getResponse_681(create_antiEntity_prompt, "Reply succinctly.",model_name)
    llm_candidates = An.analysis_anticandi_answer(anticandidates_response)
    return list(set(llm_candidates))[:max_attempts]  # 去重后取前5个


# 逻辑独立性验证
def check_logical_independence( v_j, cf):
    # 见前文check_logical_independence函数
    prompt = G.checkDuLi_antiEntity_template.format(v_j=v_j, v_j_star=cf)
    isDuLi = False
    response = huoshan.getResponse_681(prompt,"Reply succinctly.",model_name)
    isDuLi_option = An.analysis_isDuLi_response(response)
    if isDuLi_option == 'A': # 存在蕴含关系
        return False
    elif isDuLi_option == 'B':
        return True
    return isDuLi

# 逻辑独立性验证
def are_types_consistent( v_j, cf):
    # 见前文check_logical_independence函数
    prompt = G.checkSameType_antiEntity_template.format(v_j=v_j, v_j_star=cf)
    isSame = False
    response = huoshan.getResponse_681(prompt,"Reply succinctly.",model_name)
    isSame_option = An.analysis_isSameType_response(response)
    if isSame_option == 'A': # 相似
        return True
    elif isSame_option == 'B':
        return True
    return isSame

# entity_type:类型，v_j 原实体
def find_counterfactual_entity_dinamically(entity_type, v_j, Candi_num=Candidat_num, max_attempts=2):
    """生成符合类型一致、逻辑独立的反事实实体"""
    valid_cfs = []
    attempts = 0
    
    # while attempts <= max_attempts:
    #     # 1. 生成候选实体
    #     candidates = generate_candidates(v_j, entity_type, Candi_num)
    #     logger.info(f'find_counterfactual_entity_dinamically\nNo.{attempts}: {candidates}')
    #     for cf in candidates:
    #         # 2. 类型一致性校验
    #         is_SameType = are_types_consistent(v_j, cf) # 同类
    #         is_Duli = check_logical_independence(v_j, cf) # 独立
    #         if (not is_SameType) or not is_Duli :
    #             logger.info(f'find_counterfactual_entity_dinamically\n v_j={v_j}，cf={cf}，is_SameType: {is_SameType}，is_Duli={is_Duli}')
    #             continue
    #         valid_cfs.append(cf)
    #     attempts += 1
    candidates = generate_candidates(v_j, entity_type, Candi_num)
    for cf in candidates:
        valid_cfs.append(cf)
    candi_len = Candi_num if Candi_num < len(valid_cfs) else len(valid_cfs)
    logger.info(f'find_counterfactual_entity_dinamically\nfinal: {valid_cfs[:candi_len]}')
    return valid_cfs[:candi_len]

# context 原web信息，ner_results：关键实体
def replace_entities_in_context(context: str, ner_results: List[Dict], min_score=0.65) -> str:
    new_context = context
    replaced = {}

    current_entity_tokens = []
    current_type = None
    entity_text = ""

    def process_entity(entity_tokens, entity_type):
        entity_text = " ".join(entity_tokens)
        replacement = find_counterfactual_entity_dinamically(entity_type, entity_text,Candidat_num)
        pattern = re.compile(rf'\b{re.escape(entity_text)}\b', flags=re.IGNORECASE)
        if replacement:
            pattern = re.compile(rf'\b{re.escape(entity_text)}\b', flags=re.IGNORECASE)
            replacement_str = random.choice(replacement)
            logger.info(f'replace_entities_in_context\nreplacement= {replacement} → selected replacement: {replacement_str}')
            return entity_text, replacement_str, pattern
        elif not replacement and entity_type=='PER':
            replacement_str = f"Zhang san"
        elif not replacement and entity_type=='ORG':
            replacement_str = f"ALiGroup"
        elif not replacement and entity_type=='LOC':
            replacement_str = f"Danfeng Street"
        elif not replacement and entity_type=='GPE':
            replacement_str = f"BeiKing"
        elif not replacement and entity_type=='FAC':
            replacement_str = f"Wuxi Airport"
        elif not replacement and entity_type=='PRODUCT':
            replacement_str = f"Phoenix bicycle"
        elif not replacement and entity_type=='EVENT':
            replacement_str = f"Primary School Graduation Ceremony"
        elif not replacement and entity_type=='LAW':
            replacement_str = f"Law 311"
        elif not replacement and entity_type=='WORK_OF_ART':
            replacement_str = f"Shu embroidery"
        elif not replacement and entity_type=='LANGUAGE':
            replacement_str = f"Indonesian"
        else: # 数值类型 或空
            replacement_str = f"twenty two"
        logger.info(f'replace_entities_in_context\nreplacement= {replacement} → selected replacement: {replacement_str}')
        return entity_text, replacement_str, pattern

    for entry in ner_results:
        entity_tag = entry["entity"]
        word = entry["word"]
        score = entry["score"]
        if score < min_score:
            continue

        if "-" in entity_tag:
            prefix, tag_type = entity_tag.split("-")
        else:
            continue  # O 标签或非法标签
        index = 0
        if prefix == "B" and index < Fact_num:
            # 处理前一个实体
            if current_entity_tokens:
                old_ent, new_ent, pattern = process_entity(current_entity_tokens, current_type)
                if old_ent and new_ent:
                    new_context = pattern.sub(new_ent, new_context)
                    replaced[old_ent] = new_ent
            # 启动新实体
            current_entity_tokens = [word]
            current_type = tag_type
            index +=1
        elif prefix == "I" and tag_type == current_type:
            # 合并同一个实体
            if word.startswith("##"):
                current_entity_tokens[-1] += word[2:]
            else:
                current_entity_tokens.append(word)
        else:
            # 非连续实体块，结束当前，开始新的（或跳过）
            # if current_entity_tokens:
            #     old_ent, new_ent, pattern = process_entity(current_entity_tokens, current_type)
            #     if old_ent and new_ent:
            #         new_context = pattern.sub(new_ent, new_context)
            #         replaced[old_ent] = new_ent
            # current_entity_tokens = []
            # current_type = None
            continue
            
        
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

    # Merge entities
    entities = merge_ner_entities(ner_results)
    
    new_context, replacements = replace_entities_in_context(webinfor, ner_results)
    logger.info(f'new_E\nentities={entities},new_context={new_context}, replacements={replacements}')
    
    return entities,new_context, replacements 



# 计算直接因果效应
# causalgraph = {'First':xx, 'Second':xx, 'Third':xx, 'Forth':xx, 'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
def nde(E_star, claim, causalgraph):
    graph = causalgraph['graph']
    causal_answer = causalgraph['causal_answer'] # 选择的分类
    
    prompt = G.nde_template_avertic.format(graph = graph,claim=claim, information=E_star )
    logger.info(f'NDE prompt = {prompt}')
    index_true = 0
    while index_true < 3:
        nderesponse = huoshan.getResponse_681(G.SYSTEM_PROMPT_ace_ACERTIC, prompt,model_name)
        logger.info(f'NDE nderesponse = {nderesponse}')
        ndeanswer,is_match = An.analysis_nde(nderesponse) # {'causal_answer':xx,'check_answer':xx,'explanation':xx}
        logger.info(f'NDE is_match={is_match}，ndeanswer = {ndeanswer}')
        
        if is_match:
            break
        index_true+=1
    if index_true == 3:
        ndeanswer = {'causal_answer':'0.0','check_answer':'0.0','explanation':'no idea'}
    if An.is_float_string(causal_answer) == False:
        causal_answer ='0.0'
    
    if An.is_float_string(ndeanswer['causal_answer']) == False:
        ndeanswer['causal_answer'] = '0.0'
        
    nde = float(causal_answer) - float(ndeanswer['causal_answer'])
    
    return nde

# 计算间接因果效应，也就是根据新的E 重新生成graph和答案
# causalgraph = {'graph':xx, 'caculate':xx, 'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
def tie(E_star, claim, causalgraph):
    causal_answer = causalgraph['causal_answer'] # 选择的分类
    sys, causalprompt = An.generateprompt_causal_api4Averitec(claim,E_star)
    logger.info(f'causal_response= {causalprompt}')
    index= 0
    while index<3:
        causalanswer =  huoshan.getResponse_681(sys, causalprompt,model_name)
        logger.info(f'TIE causalanswer= {causalanswer}')
        tieanswer,ismatch = An.analysis_answer_causalNoType(causalanswer) # {'graph':xx, 'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
        logger.info(f'TIE ismatch={ismatch}。 DictAnswer= {tieanswer}, ')
        if ismatch:
            break
        index+=1
    if index == 3:
        tieanswer = causalgraph
    if An.is_float_string(causal_answer) == False:
        causal_answer ='0.0'
    if An.is_float_string(tieanswer['causal_answer']) == False:
        tieanswer['causal_answer'] = '0.0'
    tie = float(causal_answer) - float(tieanswer['causal_answer'])
    
    return tie


# 使用分析计算 选择最合适的causal_graph
# 输入{'graph':xx,  'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
def select_causalgraph(causalgraphs,webinfor,claim,test_10id):
    causalgraph_wupian = dict()
    
    key_fact,E_star,anti_fact = new_E(claim,webinfor) # 找反事实替换  关键事实，修改后的文本，反事实实体
    logger.info(f'select_causalgraph\nclaim={claim}, key_fact = {key_fact},anti_fact = {anti_fact}, webinfor={webinfor}, E_star= {E_star}')
    
    ACEs = -1
    for causalgraph in causalgraphs:
        NDE = nde(E_star, claim, causalgraph)
        TIE = tie(E_star, claim, causalgraph)
        ACE = NDE + TIE
        
        true_label = test_label_dict[test_10id]
        url_expalnation = test_exp_dict[test_10id]
        
        now_save = datetime.now()
        graphanswer =  {
                "test_id":test_id, "truthlabel":true_label, 'claim':claim,'url_expalnation':url_expalnation, 'causalgraph':causalgraph,"graphACE":{'NDE': NDE, 'TIE':TIE,'ACE':ACE,'key_fact':key_fact,'anti_fact':anti_fact,'E_star':E_star,'orgin_E':webinfor}, 'date':now_save.strftime("%Y-%m-%d %H:%M:%S") }
        F.writedata_onejson(graphanswer,write_graph_path)
        
        if ACE > ACEs:
            ACEs = ACE
            causalgraph_wupian = causalgraph
        logger.info(f'select_causalgraph\ncausalgraphs={causalgraphs} \n ACE = {ACE}, NDE= {NDE}, TIE={TIE}')
        
    logger.info(f'select_causalgraph\nACE_MAX = {ACEs}, causalgraph_wupian= {causalgraph_wupian}')
    
    return causalgraph_wupian



# ----------------生成反事实实体，并验证合理性-------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


stage1_path = '/disk2/zhZH/mmfc/test_log_Averitec/TestAnswer-SELFLora-know2qwen-2025-07-07-14-39.json'
query_plan = F.read_jsonlist_file(stage1_path)
have_answer = [] # 413有敏感词 
no_ids = [413] # 有敏感词的样例
weneed_answer1 = [404, 81, 100, 81, 457, 288, 366, 115, 143, 73, 389, 370, 43, 58, 496, 269, 482, 454, 360, 18, 60, 148, 448, 350, 423, 47, 33, 324, 259, 306, 303, 11, 427, 10, 9, 328, 220, 262, 197, 3, 72, 107, 417, 446, 106, 97, 167, 49, 441, 89, 61, 99, 443, 494, 129, 400, 333, 132, 124, 254, 487, 243, 483, 186, 198, 113, 156, 215, 214, 42, 133, 204, 308, 194]
weneed_answer = []
testanswers = []
for data in query_plan:
    # plan = data['predict']['answer']
    # if plan != 'Dont know':
    #     continue
    test_id = data['test_id']
    
    if test_id not in weneed_answer:
        continue
    
    test_10id = f'10-{test_id}'
    claim_en = test_claim_dict[test_10id]
    claim_pub = ''# fecerouse 没有 平台、作者信息
    claim= claim_en 
    logger.info(f"========START WEB test_id= {test_id}  {model_name}===========")
    # 这里也要换模型 22
    if model_name == modellsit[0]:
        querylist = An.generate_subquery_api(test_claim_dict[test_10id]) 
    else:
        querylist = An.generate_subquery_api681(test_claim_dict[test_10id],model_name)
        
    # subqweb_list = An.collectwebinfor_avertiec(querylist,test_id,Context_window) # 找到paranum条相关的web信息（要不要选定一个范围 前五和后五）
    
    subqweb_list_golden = An.collectwebinfor_avertiec_golden(querylist,test_id,Context_window)
    webcontentall = comprise_webinfor(subqweb_list_golden) # 将子query的列表组合成一段话webinfor= [[[q,w]]]
    
    
    webcontent = An.getAbstract(claim, webcontentall,model_name) # 将子query的列表 进行摘要
    
    sys,causalprompt = An.generateprompt_causal_api4Averitec(claim_en,webcontent)
    logger.info(f'causal_response= {causalprompt}')
    causalgraphs = []
    for i in range(Causalgraph_Groupnum): # 生成n个casual_graph
        index_true = 0
        while index_true < 3:
            causal_response = huoshan.getResponse_681(sys,causalprompt,model_name)
            logger.info(f'第{i}个  causal_response= {causal_response}')
            causal_answer, ismatch = An.analysis_answer_causalNoType(causal_response)
            if ismatch:
                break
            index_true += 1
        if index_true == 3:
            option = random.choice(['A','B','C','D'])
            causal_answer = {'graph':'', 'causal_answer':'', 'check_answer':option, 'explanation':''}
        
        causalgraphs.append(causal_answer)
        logger.info(f'第{i}个  analysis_causal_answer= {causal_answer}')
    # ACE评估casual_graph
    causalgraph_wupian = select_causalgraph(causalgraphs,webcontent,claim,test_10id)
    
    now_save = datetime.now()
    true_label = test_label_dict[test_10id]
    url_expalnation = test_exp_dict[test_10id]
    testanswer = {
        "test_id":test_id, 'type': 'search',"truthlabel":true_label, 'url_expalnation':url_expalnation, "query":{'stepnum': len(querylist), 'sub_querylist':querylist}, 'answer':causalgraph_wupian,'date':now_save.strftime("%Y-%m-%d %H:%M:%S") }
    testanswers.append(testanswer)
    logger.info(f'final answer = {testanswer}') 
    
    # 记录答案
    # if len(testanswer) /2 == 0:
    logger.info(f"save: {test_id}. path:{write_answer_path}")
    F.writedata_onejson(testanswer,write_answer_path)
    
    logger.info(f"========END WEB test_id= {test_id}  {model_name}===========")

F.wirteJsonlist(testanswers,write_allanswer_path)


