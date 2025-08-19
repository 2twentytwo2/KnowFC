
import os
import sys
import json
from pprint import pprint
from datasets import Dataset
import torch
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
import re
from pathlib import Path
import sys
import os
import random
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from loguru import logger as logging
from pprint import pformat


from datetime import datetime


class MyLogging():
    @classmethod
    def set_logger(cls,print_level="INFO",
                   logfile_level="DEBUG",
                   log_file=None,
                   basepath='/data2/zhZH/mmfc/log'):

        now = datetime.now()
        # saved_model_name = saved_path / f"fcknow-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}"
        log_file = Path(basepath) / f"{Path(__file__).parent.stem}-{now.strftime('%Y-%m-%d-%H-%M')}.log" if not log_file else log_file

        logging.remove()
        logging.add(sys.stderr,level=print_level,backtrace=True,diagnose=True)
        logging.add( log_file,
                    level=logfile_level,
                    encoding="UTF-8",
                    rotation="1 day",
                    colorize=True,
                    backtrace=True,
                    diagnose=True,
                    format=
                    "{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}",
                )
        logging.level("NOTE", no=35, color="<cyan>")


def read_jsonlist_file(file_path):
    """
    读取JSON文件并返回内容列表
    参数:
        file_path (str): JSON文件路径
    返回:
        list: 包含JSON文件内容的列表  
    """
    filename = os.path.basename(file_path)
    data =[]
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 读取JSON文件内容  
    except Exception as e:
        print(f"读取文件 {filename} 出错: {str(e)}")    

    return data

def read_json_lines(file_path):
    """读取每行都是独立JSON的文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip() # 跳过空行
            if not stripped_line:
                continue
            try:
                data.append(json.loads(stripped_line))
            except json.JSONDecodeError as e:
                print(f"解析行出错: {e}\n问题行内容: {line[:100]}...")
    return data

def readonjson(file_path):
    try:
        # 打开并读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 直接解析文件对象
        
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except json.JSONDecodeError as e:
        print(f"错误：JSON 格式无效 - {e}")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return data

def read_jsonlist_file4Know(file_path:Path, bili:float):
    """
    读取JSON文件并返回内容列表
    参数:
        file_path (str): JSON文件路径
    返回:
        list: 包含JSON文件内容的列表  
    """
    filename = os.path.basename(file_path)
    datas =[]
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            datas = json.load(file)  # 读取JSON文件内容  
    except Exception as e:
        print(f"读取文件 {filename} 出错: {str(e)}")

    traindatas, testdatas=[],[]
    traindatas4know = []

    knowstage_label = ['Supported', 'Refuted']
    for data in datas:
        label = data['label']
        if label not in knowstage_label:
            continue
        traindatas4know.append(data)

    # 2. 打乱数据
    random.shuffle(traindatas4know)

    # 3. 按比例划分
    split_idx = int(bili * len(traindatas4know))  # 80%训练集
    traindatas = traindatas4know[:split_idx]
    testdatas = traindatas4know[split_idx:]

    return traindatas, testdatas

'''
    用于GRPO推理使用
'''
def read_jsonlist_file4Reason(trainpath:Path,devpath:Path, bili:float):
    filename = os.path.basename(trainpath)
    testfilename = os.path.basename(devpath)
    
    traindatas4reason,testdatas4reason = [], []
    traindatas, testdatas=[],[]

    try:
        with open(trainpath, 'r', encoding='utf-8') as file:
            traindatas4reason = json.load(file)  # 读取JSON文件内容  
    except Exception as e:
        print(f"读取文件 {filename} 出错: {str(e)}")

    try:
        with open(devpath, 'r', encoding='utf-8') as file:
            testdatas4reason = json.load(file)  # 读取JSON文件内容  
    except Exception as e:
        print(f"读取文件 {testfilename} 出错: {str(e)}")
        
    # 2. 打乱数据
    random.shuffle(traindatas4reason)
    random.shuffle(testdatas4reason)
    

    # 3. 按比例划分
    split_idx = int(bili * len(traindatas4reason))  # 80%训练集
    traindatas = traindatas4reason[:split_idx]
    testlen = len(testdatas4reason) * (1.0-bili)
    test_idx = split_idx if split_idx > testlen else testlen
    testdatas = testdatas4reason[:test_idx]

    return traindatas, testdatas



def read_jsonlist_file4Evaluate(devpath:Path, bili:float):
    traindatas4reason,testdatas4reason = [], []
    traindatas, testdatas=[],[]

    try:
        with open(devpath, 'r', encoding='utf-8') as file:
            datas = json.load(file)  # 读取JSON文件内容  
    except Exception as e:
        print(f"读取文件 {devpath} 出错: {str(e)}")

    traindatas4reason,testdatas4reason = datas,datas
    # 2. 打乱数据
    random.shuffle(traindatas4reason)
    random.shuffle(testdatas4reason)


    # 3. 按比例划分
    split_idx_train = int(bili * len(traindatas4reason))  # 80%训练集
    traindatas = traindatas4reason[:split_idx_train]

    split_idx_test = int((1-bili) * len(testdatas4reason))  # 20%测试集
    split_idx = split_idx_train if split_idx_train < split_idx_test else split_idx_test
    testdatas = testdatas4reason[:split_idx]

    return traindatas, testdatas


def read_jsonlist_file4Search(file_path):
    filename = os.path.basename(file_path)
    datalist =[]
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            datalist = json.load(file)  # 读取JSON文件内容  
    except Exception as e:
        print(f"读取文件 {filename} 出错: {str(e)}")    

    datadict = {}
    label_dict, exp_dict = {}, {}
    for data in datalist:
        ground_truth = data['ground_truth']
        true_label = ground_truth['true_label']
        exp = ground_truth['expalnation_content']
        claim_en = ground_truth['claim_en']
        test_id = ground_truth['test_id']
        datadict[test_id] = claim_en
        label_dict[test_id] = true_label
        exp_dict[test_id] = exp
        
    
    return datadict, label_dict, exp_dict


def read_jsonlist_file4Search_feverouse(file_path):
    filename = os.path.basename(file_path)
    datalist =[]
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            datalist = json.load(file)  # 读取JSON文件内容  
    except Exception as e:
        print(f"读取文件 {filename} 出错: {str(e)}")  

    datadict = {}
    label_dict, exp_dict = {}, {}
    for data in datalist:
        true_label = data['label']
        claim_en = data['claim']
        test_id = data['id']
        datadict[test_id] = claim_en
        label_dict[test_id] = true_label
        
        
        exps = data['evidence_content']
        evi = ''
        if isinstance(exps,dict):
            for evdict in exps:
                golden_evi = list(evdict.values())[0]
                evi += golden_evi
        label_format = '''The content of cliam: {claim}. This claim is {label}.'''
        label_answer = label_format.format(claim=claim_en, label=true_label)
        evi += label_answer
        exp_dict[test_id] = evi
            
    return datadict, label_dict, exp_dict



def wirteJsonlist(data,file_path):
    """
    将数据写入JSON文件
    参数:
        data (list): 要写入的列表数据
        file_path (str): JSON文件路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)  # 写入JSON文件内容  
        print(f"数据已成功写入 {file_path}")
    except Exception as e:
        print(f"写入文件 {file_path} 出错: {str(e)}")

def writedata_onejson(jsondata,file_path):
    """将 JSON 对象列表追加到文件，每行一个 JSON"""
    with open(file_path, 'a', encoding='utf-8') as f:
        # 将对象转换为 JSON 字符串并写入一行
        json_line = json.dumps(jsondata, ensure_ascii=False)
        f.write(json_line + ',\n')

def writetestdata(jsondatas,file_path):
    """将 JSON 对象列表追加到文件，每行一个 JSON"""
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in jsondatas:
            # 将对象转换为 JSON 字符串并写入一行
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + ',\n')


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

class MyLogging():
    @classmethod
    def set_logger(cls,print_level="INFO",
                   logfile_level="DEBUG",
                   log_file=None):

        now = datetime.now()
        # saved_model_name = saved_path / f"fcknow-{model_name}-{now.strftime('%Y-%m-%d-%H-%M')}"
        if not log_file:
            basepath = '/disk2/zhZH/mmfc/log'
            log_file = Path(basepath) / f"{Path(__file__).parent.stem}-{now.strftime('%Y-%m-%d-%H-%M')}.log"

        logging.remove()
        logging.add(sys.stderr,level=print_level,backtrace=True,diagnose=True)
        logging.add( log_file,
                    level=logfile_level,
                    encoding="UTF-8",
                    rotation="1 day",
                    colorize=True,
                    backtrace=True,
                    diagnose=True,
                    format=
                    "{time:YYYY-MM-DD HH:mm:ss} - {level} - {file} - {line} - {message}",
                )
        # logging.level("NOTE", no=35, color="<cyan>")
        
# 读现在已经有了的答案
def read_has_answer():
    # filepath = '/disk2/zhZH/mmfc/test_log_feverous/TestAnswer-allcausal-gpt-3.5-turbo-2025-07-18-00-45.json'
    file_aver='/disk2/zhZH/mmfc/test_log_Averitec/TestAnswer-allcausal-gpt-4o-2025-07-18-23-28.json'
    jsonlist = read_jsonlist_file(file_aver)
    datas = []
    for data in jsonlist:
        test_id = data['test_id']
        datas.append(test_id)
        
    print(datas)
    print(f'len = {len(datas)}')

# read_has_answer()