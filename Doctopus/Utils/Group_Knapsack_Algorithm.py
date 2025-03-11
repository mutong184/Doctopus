def group_knap_sack(n, m, items):
    items.sort(key=lambda x: x[0])
    groups = []
    current_group = -1
    for g, c, w in items:
        if g != current_group:
            groups.append([])
            current_group = g
        groups[-1].append((c, w, g))  # 添加组号以便追踪

    t = len(groups)
    dp = [[-float('inf')] * (m + 1) for _ in range(t + 1)]
    dp[0][0] = 0

    # 记录选取路径
    choice = [[[] for _ in range(m + 1)] for _ in range(t + 1)]

    for k in range(1, t + 1):
        for j in range(m + 1):
            dp[k][j] = dp[k - 1][j]
            choice[k][j] = choice[k - 1][j][:]  # 继承之前的选择路径
            for c, w, g in groups[k - 1]:
                if j >= c and dp[k - 1][j - c] + w > dp[k][j]:
                    dp[k][j] = dp[k - 1][j - c] + w
                    choice[k][j] = choice[k - 1][j - c] + [(g, c, w)]

    max_value = max(dp[t])
    max_value_index = dp[t].index(max_value)
    selected_items = choice[t][max_value_index]
    return max_value, selected_items

# python 代码示例
 
def group_knap_sack2(n, m, items) :
    items.sort(key = lambda x : x[0])
    groups = []
    current_group = -1
    for g, c, w in items :
        if g != current_group :
            groups.append([])
            current_group = g
        groups[-1].append((c, w))
    t = len(groups)
    dp = [ [-float('inf')] * (m + 1) for _ in range(t + 1)]
    dp[0][0] = 0
    for k in range(1, t + 1) :
        for j in range(1, m + 1) :
            dp[k][j] = dp[k - 1][j]
            for c, w in groups[k - 1] :
                if j >= c :
                    dp[k][j] = max(dp[k][j], dp[k - 1][j - c] + w)
    max_value = max(dp[t])
    return max_value
 
# n = 7
# m = 10
# items = [
#     (1, 2, 3),  # 组号1, 容量2, 价值3
#     (1, 3, 4),  # 组号1, 容量3, 价值4
#     (2, 4, 5),  # 组号2, 容量4, 价值5
#     (3, 1, 2),  # 组号3, 容量1, 价值2
#     (3, 2, 2),  # 组号3, 容量2, 价值2
#     (4, 5, 10),  # 组号4, 容量5, 价值10
#     (5, 3, 6)  # 组号5, 容量3, 价值6
# ]
 
# print(group_knap_sack(n, m, items))  # 输出最大价值

# def group_knap_sack(n, m, items):
#     items.sort(key=lambda x: x[0])
#     groups = []
#     current_group = -1
#     for g, c, w in items:
#         if g != current_group:
#             groups.append([])
#             current_group = g
#         groups[-1].append((c, w))

#     dp = [0] * (m + 1)  # 维持一维数组以保持简洁

#     for group in groups:
#         current_max = [0] * (m + 1)  # 记录当前组可达到的最大价值
#         for c, w in group:
#             for j in range(m, c - 1, -1):  # 只更新能够放下当前物品的容量
#                 current_max[j] = max(current_max[j], dp[j - c] + w)
#         for j in range(m + 1):  # 将当前组的最大价值更新到dp中
#             dp[j] = max(dp[j], current_max[j])

#     return max(dp)

# 示例调用代码
n = 7
m = 10
items = [
    (1, 2, 3),  # 组号1, 容量2, 价值3
    (1, 3, 4),  # 组号1, 容量3, 价值4
    (2, 4, 5),  # 组号2, 容量4, 价值5
    (3, 1, 2),  # 组号3, 容量1, 价值2
    (3, 2, 2),  # 组号3, 容量2, 价值2
    (4, 5, 10), # 组号4, 容量5, 价值10
    (5, 3, 6)   # 组号5, 容量3, 价值6
]

print(group_knap_sack(n, m, items))  # 输出最大价值

def group_knap_sack2(n, m, items):
    items.sort(key=lambda x: x[0])  # 按组排序
    groups = []
    current_group = -1
    for g, c, w in items:
        if g != current_group:
            groups.append([])
            current_group = g
        groups[-1].append((c, w))

    t = len(groups)
    dp = [[0] * (m + 1) for _ in range(t + 1)]  # 初始化DP表为0

    for k in range(1, t + 1):
        for j in range(m + 1):
            dp[k][j] = dp[k - 1][j]  # 继承之前的最大值
            for c, w in groups[k - 1]:
                if j >= c and dp[k - 1][j - c] + w > dp[k][j]:
                    dp[k][j] = dp[k - 1][j - c] + w

    max_value = max(dp[t])  # 获取最大价值
    return max_value



# num = 7
# contain = 10
# items = [
#     (1, 2, 3),  # 组号1, 容量2, 价值3
#     (1, 0, 4),  # 组号1, 容量3, 价值4
#     (2, 4, 5),  # 组号2, 容量4, 价值5
#     (3, 1, 2),  # 组号3, 容量1, 价值2
#     (3, 2, 2),  # 组号3, 容量2, 价值2
#     (4, 5, 10),  # 组号4, 容量5, 价值10
#     (5, 0, 6)   # 组号5, 容量3, 价值6
# ]

# max_value, selected_items = group_knap_sack(num, contain, items)
# print("最大价值:", max_value)
# print("选择的物品:", selected_items)
