import math
import random
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
p2 = 0.3
LEN = 10000
p = 1 - p2
n = 1000
k = 3
m = k - 2
p1 = 1 - p
values = []
d1 = {}


def sample_mean(d1):
    summ = sum([d1[key] for key in d1])
    sm = (1/summ)*sum([(d1[key]+ 1) * (key+1) for key in d1])
    return sm


def choice_dispertion(d1): # выборочная дисперсия
    v = sum([d1[key] for key in d1])
    x = sample_mean(d1)
    c_d = (1 / v) * sum([(key + 1 - x) * (key + 1 - x) * d1[key] for key in d1])
    return c_d


def median(d1):
    v = sum([d1[key] for key in d1])
    w = list()
    for key in d1:
        for s in range(d1[key]):
            w.append(key)
    # w = [int(x) for x in list(''.join([str(key)*d1[key] for key in d1]))]
    if(v % 2 == 0):
        m = (1 / 2) * (w[int(v / 2)] + w[int(v / 2) - 1])
        return m
    else:
        return w[int(v / 2)]


def sweep(d1): # размах выборки
    # w = [int(x) for x in list(''.join([str(key) * d1[key] for key in d1]))]
    w = list()
    for key in d1:
        for s in range(d1[key]):
            w.append(key)
    return w[len(w) - 1] - w[0]


def distribution_function(x, d1):
    if x <= 0:
        return 0
    else:
        val = 0
        for key in d1:
            if x > key:
                val += d1[key]
            else:
                break
        val /= n
        return val


def th_func(x):  # Теоретическая функция распределения
    return 1 - (1-p2)**x


def geom_probability(x):  # Подсчет геометрической вероятности для q_j
    return p2*(1 - p2)**(x - 1)


def draw_plot(d1):
    x1 = np.linspace(0, max(d1), 100000)
    x2 = np.linspace(0, max(d1), max(d1) + 1)
    plt.plot(x1, [distribution_function(i, d1) for i in x1], color='r', label='Эмпирическая функция распределения')
    plt.step(x2, [th_func(i) for i in x2], where='pre', color='b', label='Теоретическая функция распределения')
    plt.legend(loc='lower right', frameon=False, prop={'size': 6})
    # plt.grid()
    plt.show()


def D(d1): # Мера расхождения теоретической и выборочной функций распределения
    x = np.linspace(0, max(d1), max(d1) + 1)
    d = 0
    for i in x:
        if abs(distribution_function(i, d1) - th_func(i)) > d:
            d = abs(distribution_function(i, d1) - th_func(i))
    return round(d, 4)


def max_deviation(d1):
    maxd = 0
    for key in d1:
        if abs(p * (1 - p) ** key - d1[key] / n) > maxd:
            maxd = abs(p * (1 - p) ** key - d1[key] / n)
    return round(maxd, 4)


def density(x, r):
    if x <= 0:
        return 0
    else:
        return 2 ** (-r / 2) * (1 / math.gamma(r / 2)) * x ** (r / 2 - 1) * math.exp(-x / 2)


def integral(a, b, r):
    n = 1000
    t = (b - a) / n
    res = 0
    for i in range(n):
        res += (density(a + t * i, r) + density(a + t * (i + 1), r)) * t / 2
    return res


def Function(R_0, r):
    return 1 - integral(0, R_0, r)


def nj(n, splitting, d2):
    result = 0
    for i in range(splitting[n], splitting[n + 1]):
        if i in d2:
            result += d2[i]
    return result


def R0(splitting, d2, q):
    r0 = 0
    for i in range(k):
        if q[i] != 0:
            r0 += ((nj(i, splitting, d2) - n * q[i])**2)/(n * q[i])
    return r0


for i in range(LEN):
    d1[i] = 0
# Получение выборки
flag = True
for i in range(n):
    while flag:
        a = random.random()
        if a > p:
            flag = False
        else:
            values.append(a)
    d1[len(values)] += 1
    values.clear()
    flag = True
for i in range(LEN):
    if d1[i] == 0:
        d1.pop(i)

# Печать таблицы для лабораторной 1
table_lab1 = PrettyTable()
table_lab1.header = False
table_lab1.add_row([key + 1 for key in d1])
table_lab1.add_row([d1[key] for key in d1])
table_lab1.add_row([round(d1[key]/n, 4) for key in d1])
table_lab1.add_column('', ['η‎(i)', 'n(i)', 'η‎(i)/n'])
print('')
print(table_lab1)
# Находим значения выборочных величин для таблицы
sampl_m = sample_mean(d1)
med = median(d1)
disp2 = choice_dispertion(d1)
sw = sweep(d1)

draw_plot(d1)  # Строим графики теоретич. и эмпирич. функций распределения

# Таблица 1 для лабораторной 2
table_lab2_part1 = PrettyTable()
table_lab2_part1.field_names = ['Eη', 'x', '|Eη − x|', 'Dη', 'S^2', '|Dη − S^2|', 'Me', 'R']
table_lab2_part1.add_rows(
    [
        [round(x, 4) if isinstance(x, float) else x for x in [1/p2, sampl_m, abs(1/p2 - sampl_m), (1-p2)/(p2**2), disp2, abs((1-p2)/(p2**2) - disp2), med, sw]]

    ]
)
# Таблица 2 для лабораторной 2
table_lab2_part2 = PrettyTable()
table_lab2_part2.add_column('y', [key + 1 for key in d1])
table_lab2_part2.add_column('P', [round(p2 * (1 - p2) ** key, 25) for key in d1])
table_lab2_part2.add_column('η/n', [round(d1[key] / n, 4) for key in d1])
print('')
print(table_lab2_part1)
print('')
print(table_lab2_part2)
print('')
print('D =', D(d1))
print('Maxd = ', max_deviation(d1))

# Lab3
d2 = {key + 1: d1[key] for key in d1}
left_b = max(d2) # Максимальное значение выборки
splitting = [0]*(k+1)  # Разбиение
splitting[0] = 1
print('Введите интервалы разбиения')
for z in range(k - 1):
    print(f'z{z + 1} = ')
    splitting[z + 1] = int(input())
splitting[k] = 10000  # Берем большую правую границу, чтобы точно не дойти до нее
q = [0]*k  # Вероятности для каждого интервала разбиение
index = 0
for b in range(1, len(splitting)):

    for i in range(left_b):
        if i < splitting[b]:
            if i >= splitting[b - 1]:
                q[index] += geom_probability(i)
        else:
            index += 1
            break
q[index] = 1 - sum(q[:k - 1])
print('q = ', q)
r0 = R0(splitting, d2, q)
f = Function(r0, k - 1)
print('r0 = ', r0)
print('f = ', f)
alpha = 0.5  # Для уровня значимости 0.5 количество 'Принимаем' и 'Отвергаем' должно быть примерно поровну
if f > alpha:
    print('Принимаем')
else:
    print('Отвергаем')
