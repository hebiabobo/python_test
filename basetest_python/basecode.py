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

tuple0 = ('abcd', 786, 2.23, 'runoob', 70.2)
tinytuple0 = (123, 'runoob')

print(tuple0)  # 输出完整元组
print(tuple0[0])  # 输出元组的第一个元素
print(tuple0[1:3])  # 输出从第二个元素开始到第三个元素
print(tuple0[2:])  # 输出从第三个元素开始的所有元素
print(tinytuple0 * 2)  # 输出两次元组

print(tuple0 + tinytuple0)  # 连接元组

from collections import namedtuple

City = namedtuple("City", "name country population coordinates")
tokyo = City('Tokyo', 'JP', 36.933, (35.689722, 139.691667))
print(tokyo)
# City(name='Tokyo', country='JP', population=36.933, coordinates=(35.689722, 139.691667))
print(tokyo.name)
# Tokyo


sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}

print(sites)  # 输出集合，重复的元素被自动去掉

# 成员测试
if 'Runoob' in sites:
    print('Runoob 在集合中')
else:
    print('Runoob 不在集合中')

# set可以进行集合运算
a = set('abracadabra')
b = set('alacazam')

print(a)

print(a - b)  # a 和 b 的差集

print(a | b)  # a 和 b 的并集

print(a & b)  # a 和 b 的交集

print(a ^ b)  # a 和 b 中不同时存在的元素

sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}

print(sites)  # 输出集合，重复的元素被自动去掉

# 成员测试
if 'Runoob' in sites:
    print('Runoob 在集合中')
else:
    print('Runoob 不在集合中')

# set可以进行集合运算
a = set('abracadabra')
b = set('alacazam')

print(a)

print(a - b)  # a 和 b 的差集

print(a | b)  # a 和 b 的并集

print(a & b)  # a 和 b 的交集

print(a ^ b)  # a 和 b 中不同时存在的元素

dict = {}
dict['one'] = "1 - 菜鸟教程"
dict[2] = "2 - 菜鸟工具"

tinydict = {'name': 'runoob', 'code': 1, 'site': 'www.runoob.com'}

print(dict['one'])  # 输出键为 'one' 的值
print(dict[2])  # 输出键为 2 的值
print(dict)
print(tinydict)  # 输出完整的字典
print(tinydict.keys())  # 输出所有键
print(tinydict.values())  # 输出所有值

x = b"hello"
y = x[1:3]  # 切片操作，得到 b"el"
z = x + b"world"  # 拼接操作，得到 b"helloworld"

print(x)
print(y)
print(z)

num_int = 123
num_flo = 1.23

num_new = num_int + num_flo

print("num_int 数据类型为:", type(num_int))
print("num_flo 数据类型为:", type(num_flo))

print("num_new 值为:", num_new)
print("num_new 数据类型为:", type(num_new))

# num_int = 123
# num_str = "456"
#
# print("num_int 数据类型为:",type(num_int))
# print("num_str 数据类型为:",type(num_str))
#
# print(num_int+num_str)
#
# # Traceback (most recent call last):
# #   File "E:\pythonProject\python_test\basetest_python\basecode.py", line 184, in <module>
# #     print(num_int+num_str)
# # TypeError: unsupported operand type(s) for +: 'int' and 'str'


x = int(1)  # x 输出结果为 1
y = int(2.8)  # y 输出结果为 2
z = int("3")  # z 输出结果为 3

print(x)
print(y)
print(z)

x = float(1)  # x 输出结果为 1.0
y = float(2.8)  # y 输出结果为 2.8
z = float("3")  # z 输出结果为 3.0
w = float("4.2")  # w 输出结果为 4.2

num_int = 123
num_str = "456"

print("num_int 数据类型为:", type(num_int))
print("类型转换前，num_str 数据类型为:", type(num_str))

num_str = int(num_str)  # 强制转换为整型
print("类型转换后，num_str 数据类型为:", type(num_str))

num_sum = num_int + num_str

print("num_int 与 num_str 相加结果为:", num_sum)
print("sum 数据类型为:", type(num_sum))

# s = 'RUNOOB'
# print(s)
# print(type(s))
# s1 = repr(s)
# print(s1)
# print(type(s1))

s = "物品\t单价\t数量\n包子\t1\t2"
print(s)
print(repr(s))

list1 = ['Google', 'Taobao', 'Runoob', 'Baidu']
tuple1 = tuple(list1)
print(tuple1)

a = 'wwe'
b = tuple(a)
print(b)

a = {'www': 123, 'aaa': 234}
b = tuple(a)
print(b)

a = set('abcd')
print(a)
b = tuple(a)
print(b)
a = frozenset('abcd')
print(a)
