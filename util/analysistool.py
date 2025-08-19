'''
    查询问题过程中需要的工具
    1. 相似性判断

'''
import sys
import os
import re
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from util import global_z as G
import json


# 加载英文语言模型
nlp = spacy.load("en_core_web_sm")


# 从长文本中找相似性段落
# 使用BGE
def selectSimiliarPara(claim, longtxt):
    
    # 加载中文模型
    # model = SentenceTransformer('BAAI/bge-large-zh')
    # 加载英文模型
    model = SentenceTransformer('BAAI/bge-large-en')
    # long_paralist = longtxt.split('\n')
    
    long_paralist = re.split(r'[\n]', longtxt)
    # 获取文本向量（每个文本转换为768维向量）
    embeddings = model.encode(long_paralist)
    # 对查询文本进行编码
    query_embedding = model.encode(claim)
    # 计算查询与所有文本的相似度
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    # 获取最相似的文本索引
    top_indices = np.argsort(-similarities)[:3]  # 取前3个最相似的

# 提取实体
def getEntity(text):
    doc = nlp(text)

    # 提取并按实体类型分类
    entity_types = ['PERSON','ORG','GPE','DATE']
    entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],  # 地理政治实体
        "DATE": []
    }
     # 遍历文档中的所有实体
    # for ent in doc.ents:
    #     if ent.label_ in entity_types:
    #         chinese_type = entity_types[ent.label_]
    #         entities[chinese_type].append(ent.text)
    
    for ent in doc.ents:
        if ent.label_ in entity_types:
            entities[ent.label_].append(ent.text)
            
    # print(text)
    # print("Persons:", entities["PERSON"])
    # print("Organizations:", entities["ORG"])
    # print("Locations:", entities["GPE"])
    # print("Dates:", entities["DATE"])
    
    relevant_entities = [item for sublist in entities.values() for item in sublist]
    return relevant_entities


# 根据实体 找到对应的事实核查报告
def sameEntityParas_no(claim, longtxt):
    
    paragraphs = longtxt.split('\n')
    relevant_paragraphs = []
    entitys = getEntity(claim)
    for para in paragraphs:
        for entity in entitys:
            if entity in para:
                relevant_paragraphs.append(para)
    # for para in relevant_paragraphs:
    #     print(para)
    
    return '\n'.join(map(str, relevant_paragraphs))

from collections import defaultdict


"""
    从文档末尾向前查找相关段落
    
    参数:
    longdoc (str): 长文本
    query (str): 查询文本
    threshold (float): 相似度阈值（相对于最高相似度）
    
    返回:
    str: 提取的相关段落
"""
def sameEntityParas(longdoc, query, threshold=0.5):
    doc = nlp(longdoc)
    query_doc = nlp(query)
    sentences = list(doc.sents)
    
    if not sentences:
        return ""
    # 提取查询中的实体
    query_entities = set(ent.text for ent in query_doc.ents)
    
    # 计算每个句子的相似度分数
    sentence_scores = []
    for i, sent in enumerate(sentences):
        # 实体匹配分数：查询实体在句子中出现的数量
        entity_match_score = sum(1 for ent in sent.ents if ent.text in query_entities)
        
        # 语义相似度分数：基于词向量
        semantic_similarity = query_doc.similarity(sent)
        
        # 综合分数：实体匹配占主导，语义相似度辅助
        score = entity_match_score * 2 + semantic_similarity
        
        sentence_scores.append((i, score, semantic_similarity))
    
    # # 若没有找到匹配实体，仅使用语义相似度
    # if all(score == semantic_similarity for i,score,semantic_similarity in sentence_scores):
    #     sentence_scores = [(i, s, s) for i, s in enumerate(semantic_similarity)]
        
    # 找到最高相似度的句子A
    if not sentence_scores:
        return ""
    
    a_idx, max_score, max_semantic_sim = max(sentence_scores, key=lambda x: x[1])
    
    # 从文档末尾向前查找句子B
    b_idx = -1
    threshold_value = max_score * threshold
    
    for i in range(len(sentences) - 1, -1, -1):
        _, _, semantic_sim = sentence_scores[i]
        if semantic_sim >= threshold_value:
            b_idx = i
            break
    
    # 若未找到符合条件的句子B，默认取文档末尾
    if b_idx == -1:
        b_idx = len(sentences) - 1
    
    # 确定截取范围
    start_idx = min(a_idx, b_idx)
    end_idx = max(a_idx, b_idx)
    
    # 提取段落
    return " ".join([str(sentences[i]) for i in range(start_idx, end_idx + 1)])



#----------------------------------------------------------
#----------------------------------------------------------
#-------------------causal query--------------------------------

# 构建prompt，让大模型分解需要检索的子query
def generateprompt_sub_query(claim):
    # prompt = {
    #         "prompt": [
    #             {
    #                 "role": "system",
    #                 "content": G.SYSTEM_PROMPT_subquery
    #             },
    #             {
    #                 "role": "user",
    #                 "content": check_prompt
    #             },
    #         ]
    #     }
    
    prompt = G.SYSTEM_PROMPT_subquery.format(text=claim, n_queries=5)
    return prompt

# 构建prompt，让大模型分解需要检索的子query
def generateprompt_sub_query_api(claim):
    check_prompt = G.SYSTEM_PROMPT_subquery.format(text=claim, n_queries=5)
    
    return check_prompt, ''

# 构建prompt
def generateprompt_answerself(claim):
    check_prompt = G.check_template.format(text=claim)
    
    # prompt = {
    #         "prompt": [
    #             {
    #                 "role": "system",
    #                 "content": G.SYSTEM_PROMPT_answerself
    #             },
    #             {
    #                 "role": "user",
    #                 "content": check_prompt
    #             },
    #         ]
    #     }
    prompt = f'{G.SYSTEM_PROMPT_answerself}\n{check_prompt}'
    return prompt


# 构建prompt
def generateprompt_answerself_4(claim):
    check_prompt = G.check_template.format(text=claim)
    
    # prompt = {
    #         "prompt": [
    #             {
    #                 "role": "system",
    #                 "content": G.SYSTEM_PROMPT_answerself
    #             },
    #             {
    #                 "role": "user",
    #                 "content": check_prompt
    #             },
    #         ]
    #     }
    prompt = f'{G.SYSTEM_PROMPT_answerself_4}\n{check_prompt}'
    return prompt


# 构建prompt
def generateprompt_answerself_feverous(claim):
    check_prompt = G.check_template.format(text=claim)
    
    # prompt = {
    #         "prompt": [
    #             {
    #                 "role": "system",
    #                 "content": G.SYSTEM_PROMPT_answerself
    #             },
    #             {
    #                 "role": "user",
    #                 "content": check_prompt
    #             },
    #         ]
    #     }
    prompt = f'{G.SYSTEM_PROMPT_answerself_feverous}\n{check_prompt}'
    return prompt


# 构建prompt
def generateprompt_answerself_api(claim):
    check_prompt = G.check_template.format(text=claim)
    
    return G.SYSTEM_PROMPT_answerself, check_prompt

# 构建因果问答的prompt
def generateprompt_causal(claim, webinfor):
    check_prompt = G.causalcheck_template.format(text=claim, information = webinfor)
    # prompt = {
    #         "prompt": [
    #             {
    #                 "role": "system",
    #                 "content": G.SYSTEM_PROMPT_causalcheck_2Classification
    #             },
    #             {
    #                 "role": "user",
    #                 "content": check_prompt
    #             },
    #         ]
    #     }
    prompt = f'{G.SYSTEM_PROMPT_causalcheck_2Classification}\n{check_prompt}'
    return prompt

# 构建因果问答的prompt
def generateprompt_causal_api(claim, webinfor):
    check_prompt = G.causalcheck_template.format(text=claim, information = webinfor)
    
    return G.SYSTEM_PROMPT_causalcheck_2Classification, check_prompt


def generateprompt_causal_api4feverous(claim, webinfor):
    check_prompt = G.causalcheck_template.format(text=claim, information = webinfor)
    
    return G.SYSTEM_PROMPT_causalcheck_3ClassFeverous, check_prompt


def generateprompt_causalNoGraph_api4feverous(claim, webinfor):
    check_prompt = G.causalcheck_template.format(text=claim, information = webinfor)
    
    return G.SYSTEM_PROMPT_causalNoGraph_3ClassFeverous, check_prompt

def generateprompt_causal_api4Averitec(claim, webinfor):
    check_prompt = G.causalcheck_template.format(text=claim, information = webinfor)
    
    return G.SYSTEM_PROMPT_causalcheck_4Avertic, check_prompt
    


def generateprompt_causalNoGraph_api4Averitec(claim, webinfor):
    check_prompt = G.causalcheck_template.format(text=claim, information = webinfor)
    
    return G.SYSTEM_PROMPT_causalNoGraph_4Avertic, check_prompt

# 根据回复切分子query
# def analysis_answer_subquery(response):
#     '''answer的格式是<step_1>query one</step_1>
#         <step_2>query two</step_1>
#         ...
#         <step_n>query n</step_n>'''
#     pattern = r"((\d+))(.*?)"

#     # 使用 re.DOTALL 使 . 匹配包括换行符在内的所有字符
#     matches = re.findall(pattern, response, re.DOTALL)
#     ismatch = False
#     if matches:
#         # 提取内容并去除首尾空白
#         steps = [content.strip() for _, content in matches]
#         ismatch = True
#     else:
#         steps = []
#     return steps, ismatch

import ast
# 根据回复切分子query
def analysis_answer_subquery_test(response):
    pattern = r"\(\d\)\.\s*(.+)"
    matches = re.findall(pattern, response)
    querylist = []
    if not matches:
        match = re.search(r"```python\s*(\[[\s\S]*?\])\s*```", response)
        if match:
            list_str = match.group(1)
            try:
                # 安全转换为 Python 列表对象
                querylist= ast.literal_eval(list_str)
                for index in range(len(querylist)):
                    data = querylist[index]
                    if isinstance(data,str):
                        continue
                    elif isinstance(data,dict):
                        datai = str(data.items())
                        querylist[index]=datai
            except Exception as e:
                print("⚠️ 解析失败:", e)
        else: 
            querylist= []
    else:
       # 清洗每条 query
        querylist = [m.strip().strip('"').strip("'") for m in matches]
    
    ismatch = False
    if len(querylist)!= 0:
        ismatch = True
    
    return querylist,ismatch


def extract_confidence_number(text):
    # 匹配<confidence>标签内的数字（仅包含数字、空格和斜杠）
    pattern = r'<confidence>\s*([0-9\s/]+)\s*</confidence>'
    matches = re.findall(pattern, text)
    
    for match in matches:
        # 清理空格并检查是否为规范格式（如"5"或"5/10"）
        cleaned = match.strip()
        if cleaned.isdigit() or ("/" in cleaned and all(part.strip().isdigit() for part in cleaned.split("/"))):
            # 提取第一个数字部分（如"5/10"取5）
            return int(cleaned.split("/")[0])
    
    return None  # 未找到有效数字

# 根据回复 分析基于内部知识回复的答案
# 格式 {'answer':answer, 'confidence':confidence, 'explanation':''}
def analysis_answer_self(response):
    patterns = {
        "think": r"<think>(.*?)</think>",
        "answer": r"<answer>[\(\[]([ABCD])[\)\]]|[ \t]*([ABCD])[ \t]*[\.\)]?</answer>",
        "confidence": r"<confidence>(.*?)</confidence>",
        "explanation": r"<explanation>(.*?)(?=</explanation>|$)"
    }
    
    bad_sen = ['<think>\nyour reasoning process\n</think>','<answer>your option</answer>','<confidence>your confidence</confidence>','<explanation>\nyour explanation\n</explanation>']
    cleaned_response = re.sub(bad_sen[0],"",response)
    for bad in bad_sen[1:]:
        cleaned_response = re.sub(bad, "", cleaned_response)
    
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, cleaned_response, re.DOTALL)
        if match:
            if key == 'think':
                think = match.group(1).strip()
                cleaned_think = re.sub('<think>',"",think)
                cleaned_think = re.sub('</think>',"",cleaned_think)
                results[key] = cleaned_think
            elif key == 'answer':
                full_match = match.group(0)  # 获取完整匹配的字符串
                cleaned = re.sub('<answer>',"",full_match)
                cleaned = re.sub('</answer>',"",cleaned)
                if 'A' in cleaned:
                    answer = 'A'
                elif 'B' in cleaned:
                    answer = 'B'
                elif 'C' in cleaned:
                    answer = 'C'
                elif 'D' in cleaned:
                    answer = 'D'
                elif 'E' in cleaned:
                    answer = 'E'
                else:
                    answer = ""
                results[key] = answer
            elif key == 'confidence':
                confidencenum = extract_confidence_number(cleaned_response)
                results[key] = confidencenum
            elif key == 'explanation':
                explanation = match.group(1).strip()
                cleaned = re.sub('<explanation>',"",explanation)
                cleaned = re.sub('</explanation>',"",cleaned)
                results[key] = cleaned

        else:
            results[key] = ""

    return results

# 分析因果图回复
# 输出格式是{'graph':xx,  'causal_answer':xx, 'check_answer':xx, 'explanation':xx}
def analysis_answer_causal(response):
# 回复的格式是
    '''
    <First>The content of causal graph you extracted</First>
    <causal_answer>Caculate answer</causal_answer>
    <check_answer>Fact check option</check_answer>
    <explanation>Your explanation</explanation>
    '''
    patterns = {
        "graph": r"<First>(.*?)</First>",
        "causal_answer": r"<causal_answer>(.*?)</causal_answer>",
        "check_answer": r"<check_answer>(.*?)</check_answer>",
        "explanation": r"<explanation>(.*?)</explanation>"        
    }
    results = {}
    is_match = False
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        results[key] = match.group(1).strip() if match else None
        if key=='caculate':
            try:
                float(results['caculate'])  # 尝试转换为浮点数
                is_match = True
            except Exception as e: # value error, TYpeError
                is_match = False
                results[key] = '0.0'
        if key == 'check_answer':
            if results[key]:
                if 'A' in results[key]:
                    results[key] = 'A'
                elif 'B' in results[key]:
                    results[key] = 'B'
                elif 'C' in results[key]:
                    results[key] = 'C'
                elif 'D' in results[key]:
                    results[key] = 'D'
                elif 'E' in results[key]:
                    results[key] = 'E'
                else:
                    results[key] = random.choice(['A','B','C','D'])
                is_match = True
            else:
                is_match = False

    # ------报错，存在float causal_answer非数字的情况  
    
    return results, is_match
   # <check_answer>Fact check option. Choose A, B, C, or D.</check_answer>
   
def analysis_answer_causalNoType_flan(response):
# 回复的格式是
    '''
    <First>The content of causal graph you extracted</First>
    <causal_answer>Caculation, float number.</causal_answer>
    <check_answer>Fact check option. Choose A, B, C, or D.</check_answer>
    <explanation>Your explanation</explanation>
    '''
    
    badsentences = ['rst>The content of causal graph you extracted</Firs',
                    '<causal_answer>Calculation, float number.</causal_answer>',
'<check_answer>Fact check option. Choose A, B, or C.</check_answer>',
'<explanation>Your explanation</explanation>',
'<causal_answer>Calculation</causal_answer>',
'<check_answer></check_answer>'
'<explanation></explanation>',]
    
    for s in badsentences:
        response = response.replace(s,'')
    
    patterns = {
        "graph": r"First>(.*?)/First",
        "causal_answer": r"causal_answer>(.*?)/causal_answer>",
        "check_answer": r"check_answer>(.*?)check_answer>",
        "explanation": r"explanation>(.*?)/explanation>"        
    }
    results = {}
    is_match = False
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        results[key] = match.group(1).strip() if match else None
        if key=='causal_answer':
            try:
                float(results['causal_answer'])  # 尝试转换为浮点数
                is_match = True
            except Exception as e: # value error, TYpeError
                is_match = False
                results[key] = '0.0'
        if key == 'check_answer':
            if results[key]:
                if 'A' in results[key]:
                    results[key] = 'A'
                elif 'B' in results[key]:
                    results[key] = 'B'
                elif 'C' in results[key]:
                    results[key] = 'C'
                elif 'D' in results[key]:
                    results[key] = 'D'
                elif 'E' in results[key]:
                    results[key] = 'E'
                else:
                    results[key] =  random.choice(['A','B','C','D'])
                is_match = True
            else:
                is_match = False

    # ------报错，存在float causal_answer非数字的情况  
    
    return results, is_match
   
def analysis_answer_causalNoType(response):
# 回复的格式是
    '''
    <First>The content of causal graph you extracted</First>
    <causal_answer>Caculation, float number.</causal_answer>
    <check_answer>Fact check option. Choose A, B, C, or D.</check_answer>
    <explanation>Your explanation</explanation>
    '''
    
    badsentences = ['<First>The content of causal graph you extracted</First>',
                    '<causal_answer>Calculation, float number.</causal_answer>',
'<check_answer>Fact check option. Choose A, B, or C.</check_answer>',
'<explanation>Your explanation</explanation>',
'<causal_answer>Calculation</causal_answer>',
'<check_answer></check_answer>'
'<explanation></explanation>']
    
    for s in badsentences:
        response = response.replace(s,'')
    
    patterns = {
        "graph": r"<First>(.*?)</First>",
        "causal_answer": r"<causal_answer>(.*?)</causal_answer>",
        "check_answer": r"<check_answer>(.*?)</check_answer>",
        "explanation": r"<explanation>(.*?)</explanation>"        
    }
    results = {}
    is_match = False
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        results[key] = match.group(1).strip() if match else None
        if key=='caculate':
            try:
                float(results['caculate'])  # 尝试转换为浮点数
                is_match = True
            except Exception as e: # value error, TYpeError
                is_match = False
                results[key] = '0.0'
        if key == 'check_answer':
            if results[key]:
                if 'A' in results[key]:
                    results[key] = 'A'
                elif 'B' in results[key]:
                    results[key] = 'B'
                elif 'C' in results[key]:
                    results[key] = 'C'
                elif 'D' in results[key]:
                    results[key] = 'D'
                elif 'E' in results[key]:
                    results[key] = 'E'
                else:
                    results[key] =  random.choice(['A','B','C','D'])
                is_match = True
            else:
                is_match = False

    # ------报错，存在float causal_answer非数字的情况  
    
    return results, is_match

def analysis_answer_causalNoGraph(response):
# 回复的格式是
    '''
    <First>The content of causal graph you extracted</First>
    <causal_answer>Caculation, float number.</causal_answer>
    <check_answer>Fact check option. Choose A, B, C, or D.</check_answer>
    <explanation>Your explanation</explanation>
    '''
    
    badsentences = ['<CoT>Your Chain of Thought.</CoT>',
        '<confidence>Your confidence about your answer.</confidence>',
                    '<check_answer>Fact check option. Choose A, B, or C.</check_answer>',
                    'Choose',
'<check_answer>Fact check option. Choose A, B, C, or D.</check_answer>',
'<explanation>Your explanation</explanation>',
'<causal_answer>Calculation</causal_answer>',
'<check_answer></check_answer>'
'<explanation></explanation>']
    
    for s in badsentences:
        response = response.replace(s,'')
    
    patterns = {
        "graph": r"<CoT>(.*?)</CoT>",
        "check_answer": r"<check_answer>(.*?)</check_answer>",
        "causal_answer": r"<confidence>(.*?)</confidence>",
        "explanation": r"<explanation>(.*?)</explanation>"        
    }
    results = {}
    is_match = False
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        results[key] = match.group(1).strip() if match else None
        
        if key == 'check_answer':
            if results[key]:
                if 'A' in results[key]:
                    results[key] = 'A'
                elif 'B' in results[key]:
                    results[key] = 'B'
                elif 'C' in results[key]:
                    results[key] = 'C'
                elif 'D' in results[key]:
                    results[key] = 'D'
                elif 'E' in results[key]:
                    results[key] = 'E'
                else:
                    results[key] =  random.choice(['A','B','C','D'])
                is_match = True
            else:
                is_match = False

    # ------报错，存在float causal_answer非数字的情况  
    
    return results, is_match


def analysis_nde(response):
    '''
    <causal_answer>Caculate answer</causal_answer>
    <check_answer>Fact check option</check_answer>
    <explanation>Your explanation</explanation>
    '''
    patterns = {
        "causal_answer": r"causal_answer>(.*?)/causal_answer>",
        "check_answer": r"check_answer>(.*?)/check_answer>",
        "explanation": r"explanation>(.*?)/explanation>"
    }
    results = {}
    is_match = True
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL)
        results[key] = match.group(1).strip() if match else None
        if key=='caculate':
            try:
                float(results['caculate'])  # 尝试转换为浮点数
                is_match = True
            except ValueError:
                is_match = False
                results[key] = '0.0'
        if key == 'check_answer':
            if results[key]:
                if 'A' in results[key]:
                    results[key] = 'A'
                elif 'B' in results[key]:
                    results[key] = 'B'
                elif 'C' in results[key]:
                    results[key] = 'C'
                elif 'D' in results[key]:
                    results[key] = 'D'
                elif 'E' in results[key]:
                    results[key] = 'E'
                else:
                   results[key] = random.choice(['A','B','C','D'])
                is_match = True
            else:
                is_match = False


    return results,is_match # {'causal_answer':xx,'check_answer':xx,'explanation':xx}

def analysis_anticandi_answer(response):
    bad = '<entity>"Nanjing", "Beijing", "Nantong"</entity>'
    cleaned_response = re.sub(bad, "", response)
     # 匹配<entity>标签内的内容
    pattern = r'<entity>(.*?)</entity>' # r'\[\s*(.*?)\s*\]' #
    match = re.search(pattern, cleaned_response)
    
    if not match:
        return []
    
    # 提取内容并尝试JSON解析
    content = match.group(1).strip()
    try:
        # 确保内容是合法的JSON数组格式
        if content.startswith('[') and content.endswith(']'):
            return json.loads(content)
        else:
            # 手动添加方括号转换为JSON数组
            return json.loads(f"[{content}]")
    except json.JSONDecodeError:
        # 处理非标准格式（如缺少方括号）
        entities = []
        for item in content.split('", "'):
            clean_item = item.strip('" ')
            if clean_item:
                entities.append(clean_item)
        return entities

def analysis_isDuLi_response(response):
    pattern = r'<answer>\s*([AB])\s*</answer>'
    match = re.search(pattern, response)
    if match:
        return match.group(1)  # 返回A或B
    return None

def analysis_isSameType_response(response):
    pattern = r'<answer>\s*([AB])\s*</answer>'
    match = re.search(pattern, response)
    if match:
        return match.group(1)  # 返回A或B
    return None


# -----------------------------------------------------
# -----------------------------------------------------
# ------------判断两个实体是否是一个类型的 类【不准 丢弃】----------------------------

import spacy
from typing import List, Tuple, Optional

class EntityTypeValidator:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        初始化实体类型验证器
        
        Args:
            model_name: 使用的spaCy模型名称，默认为小型英语模型
        """
        self.nlp = spacy.load(model_name)
        # 定义常见实体类型映射，使输出更易理解
        self.entity_type_mapping = {
            "PERSON": "人物",
            "NORP": "民族/国家群体",
            "FAC": "建筑/设施",
            "ORG": "组织",
            "GPE": "地理政治实体",
            "LOC": "地理位置",
            "PRODUCT": "产品",
            "EVENT": "事件",
            "WORK_OF_ART": "艺术品",
            "LAW": "法律",
            "LANGUAGE": "语言",
            "DATE": "日期",
            "TIME": "时间",
            "PERCENT": "百分比",
            "MONEY": "金钱",
            "QUANTITY": "数量",
            "ORDINAL": "序数",
            "CARDINAL": "基数"
        }
    
    def get_entity_types(self, text: str) -> List[Tuple[str, str]]:
        """
        提取文本中所有实体及其类型
        
        Args:
            text: 输入文本
            
        Returns:
            包含(实体文本, 实体类型)的列表
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            # 获取标准化的实体类型名称
            entity_type = self.entity_type_mapping.get(ent.label_, ent.label_)
            entities.append((ent.text, entity_type))
        return entities
    
    def get_main_entity_type(self, text: str) -> Optional[str]:
        """
        获取文本中主要实体的类型（假设第一个实体为主要实体）
        
        Args:
            text: 输入文本
            
        Returns:
            主要实体的类型，若无实体则返回None
        """
        entities = self.get_entity_types(text)
        if entities:
            return entities[0][1]
        return None
    
    def are_types_consistent(self, original_entity: str, counterfactual_entity: str) -> bool:
        """
        验证原实体和反事实实体的类型是否一致
        
        Args:
            original_entity: 原实体文本
            counterfactual_entity: 反事实实体文本
            
        Returns:
            类型一致返回True，否则返回False
        """
        # 获取两个实体的类型
        original_type = self.get_main_entity_type(original_entity)
        counterfactual_type = self.get_main_entity_type(counterfactual_entity)
        
        # 打印类型信息用于调试
        print(f"原实体 '{original_entity}' 的类型: {original_type}")
        print(f"反事实实体 '{counterfactual_entity}' 的类型: {counterfactual_type}")
        
        # 验证类型是否一致
        return original_type == counterfactual_type


def is_float_string(s):
    if isinstance(s, str):
        try:
            float(s)  # 尝试转换
            return float(s) 
        except ValueError:
            return False
    return False


import stage2_byapi.apiHUOSHAN as huoshan 
# feverous 没有生成子query，使用api生成子query
def generate_subquery_api(claim):
    system_prompt = '''You are an intelligent assistant helping to fact-check a statement.
Given the following text, your task is to generate some concise, fact-checkable queries that can be used to search for evidence online.

Your queries should:
- Be in the form of natural language search questions
- Break down the main factual claims in the text
- Avoid vague or opinion-based questions

Return the queries as a Python list of strings.
    '''
    
    user_prompt = '''Text: {text}'''
    
    sys_sub_prompt = system_prompt
    sub_prompt = user_prompt.format(text=claim)
    answer = huoshan.generate_response_api(sys_sub_prompt, sub_prompt)
    
    querylist, ismatch = analysis_answer_subquery_test(answer)
    while True:
        answer = huoshan.generate_response_api( sys_sub_prompt, sub_prompt) # 生成第二阶段的回复
        querylist, ismatch = analysis_answer_subquery_test(answer)
        if ismatch:
            break
    return querylist




def generate_subquery_api681(claim,model_name):
    system_prompt = '''You are an intelligent assistant helping to fact-check a statement.
Given the following text, your task is to generate some concise, fact-checkable queries that can be used to search for evidence online.

Your queries should:
- Be in the form of natural language search questions
- Break down the main factual claims in the text
- Avoid vague or opinion-based questions

Return the queries as a Python list of strings.
    '''
    
    user_prompt = '''Text: {text}'''
    
    sys_sub_prompt = system_prompt
    sub_prompt = user_prompt.format(text=claim)
    answer = huoshan.getResponse_681(sys_sub_prompt, sub_prompt,model_name)
    
    '''
        回复样例：
        '1. How many main voice actors are featured in Family Guy?\n2. Which voice actor voices Carl in Family Guy and how many episodes has he appeared in?\n3. Who voices Mort Goldman in Family Guy and how many episodes has he appeared in?\n4. How many episodes has the character Horace the bartender appeared in on Family Guy?\n5. When did Ralph Garman start working with the Family Guy team and how many episodes has he appeared in?'

    '''
    
    querylist = answer.split('\n')
    return querylist

def generate_subquery_apimy(claim,model_name):
    system_prompt = '''You are an intelligent assistant helping to fact-check a statement.
Given the following text, your task is to generate some concise, fact-checkable queries that can be used to search for evidence online.

Your queries should:
- Be in the form of natural language search questions
- Break down the main factual claims in the text
- Avoid vague or opinion-based questions

Return the queries as a Python list of strings.
    '''
    
    user_prompt = '''Text: {text}'''
    
    sys_sub_prompt = system_prompt
    sub_prompt = user_prompt.format(text=claim)
    answer = huoshan.getResponse_my(sys_sub_prompt, sub_prompt,model_name)
    
    '''
        回复样例：
        '1. How many main voice actors are featured in Family Guy?\n2. Which voice actor voices Carl in Family Guy and how many episodes has he appeared in?\n3. Who voices Mort Goldman in Family Guy and how many episodes has he appeared in?\n4. How many episodes has the character Horace the bartender appeared in on Family Guy?\n5. When did Ralph Garman start working with the Family Guy team and how many episodes has he appeared in?'

    '''
    
    querylist = answer.split('\n')
    return querylist

from pathlib import Path
import filetool as F
from FlagEmbedding import FlagReranker
import random
def top_n_indices(arr, n):
    """返回浮点数组中前n个最大值的索引"""
    if n <= 0:
        return []
    # 获取排序后的索引（升序）
    indices = np.argsort(arr)
    # 返回最后n个索引（即最大的n个）
    return indices[-n:][::-1]  # [::-1] 用于反转顺序，使最大的在前

def find_gold(taget_id):
    filebase = '/data2/zhZH/mmfc/data/AVERTIC/Train1k_CNSR10-id_prompt.json'
    jsonlist = F.read_jsonlist_file(filebase)
    target_10id = f'10-{taget_id}'
    claim_label = 'We have verification the text "{claim}". The answer is {label}.'
    gold_lsit =[]
    for data in jsonlist:
        golden_truth = data['ground_truth']
        test_id = golden_truth['test_id']
        if test_id == target_10id:
            claim = golden_truth['claim_en']
            expalnation_content = golden_truth['expalnation_content']
            true_label = golden_truth['true_label']
            gold_ans = claim_label.format(claim=claim,label=true_label)
            gold_lsit.append(expalnation_content)
            gold_lsit.append(gold_ans)
            break
    return gold_lsit

def collectwebinfor_avertiec(querylist, test_id,paranum=1, noise_gold=1):
    reranker = FlagReranker('/data/bge-reranker-v2-m3', use_fp16=True)
    webbase_test_path = '/data2/zhZH/mmfc/data/AVERTIC/web/four_test'
    webtestpath = Path(webbase_test_path)/f'{test_id}.json'
    web_testdata = F.read_json_lines(webtestpath)# .json文件里面是巨大的json数据，key=has_query,item=[]
    web_hasquery = []
    gold_contenstr=''
    webcontent_hasquery,gold_content = [], []
    returnweblist = []
    for webdict in web_testdata:
        hasquery = webdict['query']
        webjsonlist = webdict['url2text']
        type = webdict['type']
        if type == 'gold':
            for onewebpage in webjsonlist:
                gold_contenstr = f'{gold_contenstr} {onewebpage}'
                if len(gold_contenstr)>200:
                    break
            if gold_contenstr !='':
                gold_content.append(gold_contenstr)
        webcontent = []
        if isinstance(webjsonlist,list):
            for onewebpage in webjsonlist:
                webcontent.append(onewebpage)# 所有paragraph都放到一个数组里
        else:
            webcontent.append(webjsonlist)
        if len(webcontent) != 0:
            webcontent_hasquery.extend(webcontent) 
            web_hasquery.append(hasquery)
    
    gold_content_label =[]
    gold_content_label = find_gold(test_id)
    gold_content.extend(gold_content_label)
    gold_len = len(gold_content)
    
    if gold_len == 0:
        gold_len = 2
    elif gold_len > 8:
        gold_content = gold_content[:8]
        gold_content.extend(gold_content_label)
        gold_len = len(gold_content)
    noise_len = noise_gold * gold_len
    random_noise = random.sample(webcontent_hasquery,noise_len)
    random_noise.extend(gold_content)
    
    # 分解出来的子query列表
    for index in range(len(querylist)):
        query = querylist[index]
        caculate = []
        # 和子query最相似的has_query
        for webdata in random_noise:
            onepair = [query, webdata]
            caculate.append(onepair)
        scores = reranker.compute_score(caculate)
        maxindexs = top_n_indices(scores,paranum)
        
        
        # 前Paranum个最相似的句子
        paragraph = ''
        for i in maxindexs:
            paragraph = f'{paragraph}{random_noise[i]}\n'# 和子query最相似的网页段落
        
        qw = [query,paragraph]
        returnweblist.append(qw)
    for onegold in gold_content:
        qw = ['gold',onegold]
        returnweblist.append(qw)
    
    return returnweblist


def collectwebinfor_avertiec_golden(querylist, test_id,paranum=1, noise_gold=1):
    reranker = FlagReranker('/data/bge-reranker-v2-m3', use_fp16=True)
    webbase_test_path = '/data2/zhZH/mmfc/data/AVERTIC/web/four_test'
    webtestpath = Path(webbase_test_path)/f'{test_id}.json'
    web_testdata = F.read_json_lines(webtestpath)# .json文件里面是巨大的json数据，key=has_query,item=[]
    web_hasquery = []
    gold_contenstr=''
    webcontent_hasquery,gold_content = [], []
    returnweblist = []
    for webdict in web_testdata:
        hasquery = webdict['query']
        webjsonlist = webdict['url2text']
        type = webdict['type']
        if type == 'gold':
            for onewebpage in webjsonlist:
                gold_contenstr = f'{gold_contenstr} {onewebpage}'
                
            if gold_contenstr !='':
                gold_content.append(gold_contenstr)
        
    
    gold_content_label =[]
    gold_content_label = find_gold(test_id)
    gold_content.extend(gold_content_label)
    gold_len = len(gold_content)
    
    if gold_len == 0:
        gold_len = 2
    elif gold_len > 8:
        gold_content = gold_content[:8]
        gold_content.extend(gold_content_label)
        gold_len = len(gold_content)
    noise_len = noise_gold * gold_len
    random_noise = []
    random_noise.extend(gold_content)
    
   
    for onegold in gold_content:
        qw = ['gold',onegold]
        returnweblist.append(qw)
    
    return returnweblist




def getAbstract(claim, allcontent,model_name):
    prompt = '''For evaluate this question 'Is what this text says true? Text:{claim}'. I collect this evidence:
    {text}.
    '''
    sys_sub_prompt = "Please give me the summerization of this evidenve less than 500 words. You don't have to use your own knowledge, just refer to the information given to you. Reply succinctly."
    sub_prompt = prompt.format(claim=claim, text=allcontent)
    print(f'sys_sub_prompt{sys_sub_prompt}\n sub_prompt:{sub_prompt}\n model: {model_name}')
    answer = huoshan.getResponse_my(sys_sub_prompt, sub_prompt,model_name)
    
    return answer

def getSelfKnow(claim,model_name):
    sub_prompt = G.interalknow_template.format(text=claim)
    answer = huoshan.getResponse_my(G.SYS_INTER_template, sub_prompt,model_name)
    
    return answer