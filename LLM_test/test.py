# from transformers import pipeline
#
# translator = pipeline("translation_en_to_fr")
# print(translator("How old are you?"))


from transformers import pipeline
import re
from string import punctuation
import unicodedata


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
    with open(path, 'r') as f:
        lines = f.readlines()

    # 去除每行末尾的换行符，并将所有行连接成一个字符串，用空格分隔
    one_line_with_spaces = ' '.join(line.strip() for line in lines)
    clean_line = process_text(one_line_with_spaces)
    # 打印合并后的单行文本
    print("text_clean_line", clean_line)

    return clean_line


path1 = r"E:\pythonProject\dataset\DUC-2004-Dataset-master\DUC2004_Summarization_Documents\duc2004_testdata\tasks1and2\duc2004_tasks1and2_docs\docs\1\D1.txt"

input_text = filter_text(path1)  # 输入的原始文本


# 明确指定要使用的模型和版本
model_name = "google-t5/t5-base"
model_revision = "686f1db"
print("111")
# 创建总结管道
summarizer = pipeline("summarization", model=model_name, revision=model_revision)
print("222")
# 输入文本

print("333")
# 生成摘要
summary = summarizer(input_text)

print(summary)
