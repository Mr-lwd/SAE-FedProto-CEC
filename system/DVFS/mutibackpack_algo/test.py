def knapsack_multiple(N, M, weights, values):
    # dp[i][j] 表示选择 j 个物品，且总重量不超过 i 的最小总价值
    dp = [[float('inf')] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0  # 初始状态，0重量，选择0个物品，总价值为0
    
    # 用来记录每种状态下选择的物品及其次数
    item_count = [[[] for _ in range(M + 1)] for _ in range(N + 1)]
    
    # 动态规划，更新 dp 表和 item_count
    for i in range(len(weights)):  # 遍历每个物品
        weight = weights[i]
        value = values[i]
        
        for w in range(weight, N + 1):  # 从背包容量为 weight 到 N 进行遍历（完全背包）
            for k in range(1, M + 1):  # 遍历选择物品的数量
                if dp[w - weight][k - 1] + value < dp[w][k]:
                    dp[w][k] = dp[w - weight][k - 1] + value
                    item_count[w][k] = item_count[w - weight][k - 1] + [(i, 1)]  # 记录选择的物品索引和次数
    
    # 找到最小总价值，并记录选择的物品及其次数
    min_value = float('inf')
    min_weight = -1
    for w in range(N + 1):
        if dp[w][M] < min_value:
            min_value = dp[w][M]
            min_weight = w
    
    # 输出结果
    print(f"最小总价值: {min_value}")
    print(f"背包容量: {min_weight}")
    print("选择的物品及其次数:")
    for item in item_count[min_weight][M]:
        print(f"物品 {item[0]}，选择次数: {item[1]}")

# 示例输入
weights = [2, 3, 4]  # 物品的重量
values = [3, 4, 5]   # 物品的价值
N = 3  # 背包容量
M = 2   # 选择物品的数量

# 调用函数
knapsack_multiple(N, M, weights, values)
