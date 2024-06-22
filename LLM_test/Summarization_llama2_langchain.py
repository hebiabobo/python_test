import torch
import transformers
# import lang_chain.lang_chain as lc
from transformers import AutoModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

access_token = "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"

config = PeftConfig.from_pretrained("awaisakhtar/llama-2-7b-summarization-finetuned-on-xsum-lora-adapter")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(model, "awaisakhtar/llama-2-7b-summarization-finetuned-on-xsum-lora-adapter")

# model_name = "summarization/t5-large"
# model = AutoModel.from_pretrained(model_name, token=access_token)
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, token=access_token)

tokenizer = AutoTokenizer.from_pretrained("awaisakhtar/llama-2-7b-summarization-finetuned-on-xsum-lora-adapter")

summary = pipeline("summarization", model="meta-llama/Llama-2-7b-hf", tokenizer=tokenizer)

import io, json, os, collections, pprint, time
import re
from string import punctuation
import unicodedata
import random


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def process_text(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub('\!+', '!', s)
    s = re.sub('\,+', ',', s)
    s = re.sub('\?+', '?', s)
    s = re.sub('\.+', '.', s)  # 将连续的句号替换为单个句号。
    s = re.sub("[^a-zA-Z.!?,'']+", ' ', s)  # 将所有非字母和特定标点符号的字符替换为空格。
    for p in punctuation:
        if p not in ["'", "[", "]"]:  # 将字符串 s 中的标点符号（除单引号、左方括号和右方括号之外）前后加上空格。
            # 使用 replace 方法在标点符号前后添加空格。
            s = s.replace(p, " " + p + " ")  # 这样做的目的是为了在处理文本时，可以更方便地分割和分析标点符号以及其周围的单词。
    s = re.sub(' +', ' ', s)  # 替换连续的空格为单个空格
    s = s.strip()  # 这行代码使用字符串方法 strip 去除字符串 s 首尾的所有空格字符。这包括空格、制表符、换行符等。
    return s


def filter_text(path):
    review_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    # f = io.open(path, encoding='utf-8')
    with open('example.txt', 'r') as f:
        lines = f.readlines()

    # 去除每行末尾的换行符，并将所有行连接成一个字符串，用空格分隔
    one_line_with_spaces = ' '.join(line.strip() for line in lines)
    clean_line = process_text(one_line_with_spaces)
    # 打印合并后的单行文本
    print("text_clean_line", clean_line)

    return clean_line


path1 = r"E:\pythonProject\dataset\DUC-2004-Dataset-master\DUC2004_Summarization_Documents\duc2004_testdata\tasks1and2\duc2004_tasks1and2_docs\docs\1\D1.txt"

input_text = filter_text(path1)  # 输入的原始文本
output_summary = summary(input_text, max_length=100, min_length=30, do_sample=False)
print(output_summary[0]['summary_text'])  # 输出摘要文本
