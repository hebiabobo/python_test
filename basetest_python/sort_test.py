a = [8, 5, 6, 4, 3, 7, 10, 2]


def selection_sort(alist):
    n = len(alist)
    for j in range(n - 1):
        min_index = j
        for i in range(j + 1, n):
            if alist[min_index] > alist[i]:
                min_index = i
            alist[j], alist[min_index] = alist[min_index], alist[j]

        print(alist)


def bubble_sort(alist):
    n = len(alist)
    for j in range(0, n - 1):
        count = 0
        for i in range(0, n - 1 - j):
            if alist[i] > alist[i + 1]:
                alist[i], alist[i + 1] = alist[i + 1], alist[i]
            count += 1
        if 0 == count:
            break

        print(alist)


def quicksort(alist):
    if len(alist) <= 1:
        return alist
    else:
        pivot = alist[len(alist) // 2]  # 选择中间元素作为基准
        left = [x for x in alist if x < pivot]  # 小于基准的元素
        middle = [x for x in alist if x == pivot]  # 等于基准的元素
        right = [x for x in alist if x > pivot]  # 大于基准的元素
        return quicksort(left) + middle + quicksort(right)


# 示例
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = quicksort(arr)
print(sorted_arr)

# selection_sort(a)
# bubble_sort(a)
