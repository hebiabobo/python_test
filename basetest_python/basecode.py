import math
import random

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

a = 60  # 60 = 0011 1100
b = 13  # 13 = 0000 1101

c = a & b  # 12 = 0000 1100
print("1 - c 的值为：", c)

c = a | b  # 61 = 0011 1101
print("2 - c 的值为：", c)

c = a ^ b  # 49 = 0011 0001
print("3 - c 的值为：", c)

c = ~a  # -61 = 1100 0011
print("4 - c 的值为：", c)

c = a << 2  # 240 = 1111 0000
print("5 - c 的值为：", c)

c = a >> 2  # 15 = 0000 1111
print("6 - c 的值为：", c)

a = 10
b = 20
list3 = [1, 2, 3, 4, 5]

if a in list3:
    print("1 - 变量 a 在给定的列表中 list 中")
else:
    print("1 - 变量 a 不在给定的列表中 list 中")

if b not in list3:
    print("2 - 变量 b 不在给定的列表中 list 中")
else:
    print("2 - 变量 b 在给定的列表中 list 中")

# 修改变量 a 的值
a = 2
if a in list3:
    print("3 - 变量 a 在给定的列表中 list 中")
else:
    print("3 - 变量 a 不在给定的列表中 list 中")

a = 20
b = 20

if a is b:
    print("1 - a 和 b 有相同的标识")
else:
    print("1 - a 和 b 没有相同的标识")

if id(a) == id(b):
    print("2 - a 和 b 有相同的标识")
else:
    print("2 - a 和 b 没有相同的标识")

# 修改变量 b 的值
b = 30
if a is b:
    print("3 - a 和 b 有相同的标识")
else:
    print("3 - a 和 b 没有相同的标识")

if a is not b:
    print("4 - a 和 b 没有相同的标识")
else:
    print("4 - a 和 b 有相同的标识")

a = [1, 2, 3]
b = a
print(b is a)
print(b == a)

b = a[::]
print(b)

print(b is a)
print(b == a)

a = 1
print(complex(a))

a = -1.2
print(math.ceil(a))
print(math.floor(a))
print(math.exp(a))
# print(math.abs(a))
print(math.fabs(a))
print(math.log(math.e))  # e
print(math.log10(1000))
print(math.modf(a))  # 返回x的整数部分与小数部分，两部分的数值符号与x相同，整数部分以浮点型表示。
print(round(a, 3))
print(round(3.1415926535, 3))  # 3.142
print(math.sqrt(2))
print(random.choice(range(10)))
print(random.randrange(1, 10, 2))
print(random.random())
print(random.uniform(1, 10))

var1 = 'Hello World!'
var2 = "Runoob"

print("var1[0]: ", var1[0])
print("var2[1:5]: ", var2[1:5])

print('\'Hello, world!\'')  # 输出：'Hello, world!'

print("Hello, world!\nHow are you?")  # 输出：Hello, world!
#       How are you?

print("Hello, world!\tHow are you?")  # 输出：Hello, world!    How are you?

print("Hello,\b world!")  # 输出：Hello world!

print("Hello,\f world!")  # 输出：
# Hello,
#  world!

print("A 对应的 ASCII 值为：", ord('A'))  # 输出：A 对应的 ASCII 值为： 65

print("\x41 为 A 的 ASCII 代码")  # 输出：A 为 A 的 ASCII 代码

decimal_number = 42
binary_number = bin(decimal_number)  # 十进制转换为二进制
print('转换为二进制:', binary_number)  # 转换为二进制: 0b101010

octal_number = oct(decimal_number)  # 十进制转换为八进制
print('转换为八进制:', octal_number)  # 转换为八进制: 0o52

hexadecimal_number = hex(decimal_number)  # 十进制转换为十六进制
print('转换为十六进制:', hexadecimal_number)  # 转换为十六进制: 0x2a

# !/usr/bin/python3

a = "Hello"
b = "Python"

print("a + b 输出结果：", a + b)
print("a * 2 输出结果：", a * 2)
print("a[1] 输出结果：", a[1])
print("a[1:4] 输出结果：", a[1:4])

if ("H" in a):
    print("H 在变量 a 中")
else:
    print("H 不在变量 a 中")

if ("M" not in a):
    print("M 不在变量 a 中")
else:
    print("M 在变量 a 中")

print(r'\n')
print(R'\n')

# !/usr/bin/python3

print("我叫 %s 今年 %d 岁!" % ('小明', 10))

# !/usr/bin/python3

para_str = """这是一个多行字符串的实例
多行字符串可以使用制表符
TAB ( \t )。
也可以使用换行符 [ \n ]。
也可以使用换行符 [ r"\n" ]。 # 没用
"""
print(para_str)

name = 'Runoob'
print(f'Hello {name}')
print(f'{1 + 2}')

w = {'name': 'Runoob', 'url': 'www.runoob.com'}
print(f'{w["name"]}: {w["url"]}')

x = 1
print(f'{x + 1}')  # Python 3.6
print(f'{x+1=}')  # Python 3.8

list1 = ['Google', 'Runoob', "Zhihu", "Taobao", "Wiki"]

# 读取第二位
print("list[1]: ", list1[1])
# 从第二位开始（包含）截取到倒数第二位（不包含）
print("list[1:-2]: ", list1[1:-2])

# !/usr/bin/python3

list1 = ['Google', 'Runoob', 1997, 2000]

print("第三个元素为 : ", list1[2])
list1[2] = 2001
print("更新后的第三个元素为 : ", list1[2])

list1 = ['Google', 'Runoob', 'Taobao']
list1.append('Baidu')
print("更新后的列表 : ", list1)
list1.insert(1, 'Facebook')
print("更新后的列表 : ", list1)
list1.reverse()
print("翻转后的列表 : ", list1)
list1.sort()
print("排序后的列表 : ", list1)
print(list1.pop(-2))  # 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
print("移除列表中的一个元素 : ", list1)

list1 = ['Google', 'Runoob', 1997, 2000]

print("原始列表 : ", list1)
del list1[2]
print("删除第三个元素 : ", list1)

for x in [1, 2, 3]: print(x, end=" ")

squares = [1, 4, 9, 16, 121]
squares += [36, 49, 64, 81, 100]
print(squares)

a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
print(x)
print(x[0])
print(x[0][1])

# 导入 operator 模块
import operator

a = [1, 2]
b = [2, 3]
c = [2, 3]
d = [3, 2]
print("operator.eq(a,b): ", operator.eq(a, b))  # False
print("operator.eq(c,b): ", operator.eq(c, b))  # True
print("operator.eq(c,b): ", operator.eq(c, d))  # False

list_2d = [[0 for i in range(5)] for i in range(5)]
list_2d[0].append(3)
list_2d[0].append(5)
list_2d[2].append(7)
print(list_2d)  # [[0, 0, 0, 0, 0, 3, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

# 元组
tup1 = (50)  # 元组中只包含一个元素时，需要在元素后面添加逗号 , ，否则括号会被当作运算符使用：
print(type(tup1))
tup1 = (50,)
print(type(tup1))

tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')

# 以下修改元组元素操作是非法的。
# tup1[0] = 100

# 创建一个新的元组
tup3 = tup1 + tup2
print(tup3)

# 使用大括号 {} 来创建空字典
emptyDict = {}

# 打印字典
print(emptyDict)

# 查看字典的数量
print("Length:", len(emptyDict))

# 查看类型
print(type(emptyDict))

tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}

print("tinydict['Name']: ", tinydict['Name'])
print("tinydict['Age']: ", tinydict['Age'])

tinydict['Age'] = 8  # 更新 Age
tinydict['School'] = "1212"  # 添加信息

print("tinydict['Age']: ", tinydict['Age'])
print("tinydict['School']: ", tinydict['School'])

# tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
#
# del tinydict['Name']  # 删除键 'Name'
# tinydict.clear()  # 清空字典
# del tinydict  # 删除字典
#
# print("tinydict['Age']: ", tinydict['Age'])
# print("tinydict['School']: ", tinydict['School'])


print(tinydict.items())
dict12 = tinydict.fromkeys(tup1)
print(dict12)


def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case 401 | 403 | 404:  # 一个 case 也可以设置多个匹配条件，条件使用 ｜ 隔开，例如：
            return "Not allowed"
        case _:
            return "Something's wrong with the internet"


mystatus = 400
print(http_error(400))

names = ['Bob', 'Tom', 'alice', 'Jerry', 'Wendy', 'Smith']
new_names = [name.upper() for name in names if len(name) > 3]
print(new_names)  # ['ALICE', 'JERRY', 'WENDY', 'SMITH']

multiples = [i for i in range(30) if i % 3 == 0]
print(multiples)  # [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

listdemo = ['Google', 'Runoob', 'Taobao']
# 将列表中各字符串值为键，各字符串的长度为值，组成键值对
new_dict = {i: len(i) for i in listdemo}
print(new_dict)

dic = {i: math.pow(i, 2) for i in [1, 2, 3, 4, 5]}
print(dic)
set1 = {i ** 2 for i in (1, 2, 3, 4, 5)}
print(set1)
a = (i ** 3 for i in range(1, 6))
print(a)  # 返回的是生成器对象  <generator object <genexpr> at 0x7faf6ee20a50>
print(type(a))

print(tuple(a))  # 使用 tuple() 函数，可以直接将生成器对象转换成元组
print(a)
print(tuple(a))
print(list(a))
# 生成器在 Python 中是一种一次性使用的迭代器。转换生成器对象为元组后，生成器中的所有值都被消耗，
# 生成器变为空，不能再生成值。因此，不能再将其转换为元组。如果需要多次访问生成器的值，应该在第一次转换时保存结果，或者每次需要时重新创建生成器实例。


# list2 = [1, 2, 3, 4]
# it = iter(list2)
# for x in it:
#     print(x)
# for x in list2:
#     print(x)
#
# # !/usr/bin/python3
#
# import sys  # 引入 sys 模块
#
# list = [1, 2, 3, 4]
# it = iter(list)  # 创建迭代器对象

# while True:
#     try:
#         print(next(it))
#     except StopIteration:
#         sys.exit()


# def countdown(n):
#     while n > 0:
#         yield n
#         n -= 1
#
#
# # 创建生成器对象
# generator = countdown(5)
#
# # 通过迭代生成器获取值
# print(next(generator))  # 输出: 5
# print(next(generator))  # 输出: 4
# print(next(generator))  # 输出: 3

# # 使用 for 循环迭代生成器
# for value in generator:
#     print(value)  # 输出: 2 1

f = lambda: "Hello, world!"
print(f())  # 输出: Hello, world!

x = lambda a: a + 10
print(x(5))
x = lambda a, b: a * b
print(x(5, 6))
x = lambda a, b, c: a + b + c
print(x(5, 6, 2))


def square(x):
    return x * x


numbers = [1, 2, 3, 4, 5]
squared_numbers = map(square, numbers)
print(squared_numbers)  # <map object at 0x0000015E8785BD00>
print(list(squared_numbers))  # 输出: [1, 4, 9, 16, 25]

numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x * x, numbers)

print(list(squared_numbers))  # 输出: [1, 4, 9, 16, 25]


def add(x, y):
    return x + y


a = [1, 2, 3]
b = [4, 5, 6]
sums = map(add, a, b)

print(list(sums))  # 输出: [5, 7, 9]

strings = ['1', '2', '3', '4', '5']
numbers = map(int, strings)

print(list(numbers))  # 输出: [1, 2, 3, 4, 5]

from functools import reduce


def add(x, y):
    return x + y


numbers = [1, 2, 3, 4, 5]
sum_result = reduce(add, numbers)
print(sum_result)  # 输出: 15

numbers = [1, 2, 3, 4, 5]
sum_result = reduce(lambda x, y: x + y, numbers)
print(sum_result)  # 输出: 15

from functools import reduce


def multiply(x, y):
    return x * y


numbers = [1, 2, 3, 4, 5]
product_result = reduce(multiply, numbers)
print(product_result)  # 输出: 120

numbers = [1, 2, 3, 4, 5]
sum_result = reduce(lambda x, y: x + y, numbers, 10)  # 初始值 10 会首先与序列中的第一个元素进行运算，然后再进行后续的累积计算。
print(sum_result)  # 输出: 25


class Human:
    def __init__(self):
        # self.不能丢
        self.arms = 2
        self.legs = 2
        self.hair = '各种颜色的头发'

    def walk(self):
        print('直立行走')

    def speak(self):
        print('说着各式各样的语言')


human = Human()
print(human.hair)  # 输出：各种颜色的头发


class Students(object):
    def __init__(self, *args):
        self.names = args

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx]


ss = Students('Bob', 'Alice', 'Tim')
print(len(ss))  # 3
print(ss[1])  # Alice

iterator = iter([1, 2, 3, 4])

'''
next() 函数是 Python 中用于从迭代器中获取下一个元素的内置函数。迭代器是一种可以记住遍历位置的对象，它使用 __iter__() 和 __next__() 方法来实现迭代。next() 函数提供了一种方便的方式来获取迭代器的下一个元素。

用法和参数

next(iterator[, default])
iterator：一个迭代器对象。
default（可选）：如果提供了这个参数，当迭代器没有更多元素时，会返回这个默认值，而不会抛出 StopIteration 异常。
'''

# 使用 next() 获取元素
print(next(iterator))  # 输出: 1
print(next(iterator))  # 输出: 2
print(next(iterator))  # 输出: 3
print(next(iterator))  # 输出: 4


# print(next(iterator))  # 如果继续调用，会抛出 StopIteration 异常


def func(a, b, c):
    print(a, b, c)


func(**{"a": 1, "b": 2, "c": 3})

from rouge import Rouge

generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"

rouge = Rouge()

'''
基于准确率和召回率来计算 F1 值。
ROUGE-1 度量 uni-grams 的重合情况，ROUGE-2 度量 bi-grams 的重合情况，
而 ROUGE-L 则通过在生成摘要和参考摘要中寻找最长公共子串来度量最长的单词匹配序列
'''

scores = rouge.get_scores(
    hyps=[generated_summary], refs=[reference_summary]
)[0]
print(scores)

'''
rouge 库默认使用空格进行分词，因此无法处理中文、日文等语言，最简单的办法是按字进行切分，
当然也可以使用分词器分词后再进行计算，否则会计算出不正确的 ROUGE 值:
'''
generated_summary = "我在苏州大学学习计算机，苏州大学很美丽。"
reference_summary = "我在环境优美的苏州大学学习计算机。"

rouge = Rouge()

TOKENIZE_CHINESE = lambda x: ' '.join(x)  # 按字进行切分

# from transformers import AutoTokenizer
# model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# TOKENIZE_CHINESE = lambda x: ' '.join(
#     tokenizer.convert_ids_to_tokens(tokenizer(x).input_ids, skip_special_tokens=True)
# )

scores = rouge.get_scores(
    hyps=[TOKENIZE_CHINESE(generated_summary)],
    refs=[TOKENIZE_CHINESE(reference_summary)]
)[0]
print('ROUGE:', scores)
scores = rouge.get_scores(
    hyps=[generated_summary],
    refs=[reference_summary]
)[0]
print('wrong ROUGE:', scores)
print(TOKENIZE_CHINESE(generated_summary))




