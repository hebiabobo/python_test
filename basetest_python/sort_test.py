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


def shell_sort(alist):
    n = len(alist)
    gap = n // 2
    while gap:
        for i in range(gap, n):  # 每一个gap里面的元素比较
            while i - gap >= 0 and alist[i - gap] > alist[i]:
                alist[i - gap], alist[i] = alist[i], alist[i - gap]
                print("i:", i)
                i -= gap
                print("i = i - gap:", i)
                print(alist)
            print("222")
        gap //= 2
        print("333")
    return alist


def heapify(arr, n, i):
    largest = i  # 初始化最大值为根节点
    left = 2 * i + 1  # 左子节点
    right = 2 * i + 2  # 右子节点

    # 如果左子节点存在且大于根节点
    if left < n and arr[i] < arr[left]:
        largest = left

    # 如果右子节点存在且大于最大值
    if right < n and arr[largest] < arr[right]:
        largest = right

    # 如果最大值不是根节点
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # 交换
        heapify(arr, n, largest)  # 递归堆化子树


def heap_sort(arr):
    n = len(arr)

    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 一个个交换元素
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # 交换
        heapify(arr, i, 0)  # 堆化根节点

    return arr


arr = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# arr = [3, 6, 8, 10, 1, 2, 1, 15, 4, 7, 9, 11]
# sorted_arr = quicksort(arr)
# sorted_arr = shell_sort(arr)
sorted_arr = heap_sort(arr)
print(sorted_arr)

# selection_sort(a)
# bubble_sort(a)
