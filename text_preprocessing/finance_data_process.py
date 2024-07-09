import pandas as pd
import re


# 定义提取数字和小数点的函数
def extract_numbers(input_string):
    result = re.sub(r'[^\d.\n]', '', input_string)
    return result


# 定义保留前四个字符的函数
def keep_first_four_chars(input_string):
    # 使用正则表达式匹配前四个字符
    return input_string[:4]


# 定义去除前四个字符的函数
def remove_first_four_chars(input_string):
    # 使用正则表达式匹配前四个字符并替换为空字符串
    return input_string[4:]


df = pd.read_excel('finance_cap_test.xlsx')

for index, row in df.head(30).iterrows():
    processed_symbol = keep_first_four_chars(row.iloc[1])
    # print(processed_symbol)

    processed_corp_name = remove_first_four_chars(row.iloc[2])
    # print(processed_corp_name)

    processed_value = extract_numbers(row.iloc[4])
    # print(processed_value)

    # 保存
    df.at[index, df.columns[1]] = processed_symbol
    df.at[index, df.columns[2]] = processed_corp_name
    df.at[index, df.columns[4]] = processed_value

df.head(30).to_csv('finance.csv', index=False)
