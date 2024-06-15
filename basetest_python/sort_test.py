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


# selection_sort(a)
bubble_sort(a)
