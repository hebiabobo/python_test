total = ['item_one', 'item_two', 'item_three',
         'item_four', 'item_five']

# !/usr/bin/python3

str = '123456789'

print(str)  # 输出字符串
print(str[0:-1])  # 输出第一个到倒数第二个的所有字符
print(str[0])  # 输出字符串第一个字符
print(str[2:5])  # 输出从第三个开始到第六个的字符（不包含）
print(str[2:])  # 输出从第三个开始后的所有字符
print(str[1:5:2])  # 输出从第二个开始到第五个且每隔一个的字符（步长为2）
print(str * 2)  # 输出字符串两次
print(str + '你好')  # 连接字符串
print('------------------------------')
# total.sort()
# print(total)
print('------------------------------')
print(total[::-1])  # 反转列表
print(str[::-1])
print(str[-2::-2])

print('------------------------------')

print('hello\nrunoob')  # 使用反斜杠(\)+n转义特殊字符
print(r'hello\nrunoob')  # 在字符串前面添加一个 r，表示原始字符串，不会发生转义

print('------------------------------')

str1 = 'Share your videos with friends, family, and the world.'
liststr1 = str1.split()
print(liststr1)  # 输出字符串中以空格为分隔符的所有子字符串，返回列表
str2 = ' '.join(liststr1)
print(str2)  # 输出列表中所有元素以空格为分隔符连接成的字符串

print('------------------------------')
import re

p = re.compile('[\u0800-\u9fa5\uac00-\ud7a3]')
text = '''
这是简体中文,這是繁體中文
这是日文,これは日本語です
这是韩文,한국 사람입니다
'''
print(p.findall(text))


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


print(isEnglish(text))

# 导入所需的库
from langdetect import detect

# 获取一段文字

# 执行语言检测
language = detect(text)
# 判断结果
if language == 'ja':
    print("这段文字是日语。")
else:
    print("这段文字不是日语。")
