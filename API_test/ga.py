import numpy as np
import matplotlib.pyplot as plt

# ===================== 解决中文显示问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# ===================== 1. 问题定义：目标函数 + 变量约束 =====================
# 火力覆盖效能目标函数（函数值越大越好）
# x1: 火力点A横向坐标，x2: 火力点B纵向坐标
def fitness_function(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 2

# 变量约束（坐标范围，单位：千米）
X1_MIN, X1_MAX = -2, 2
X2_MIN, X2_MAX = -2, 2

# ===================== 2. 遗传算法核心参数设置 =====================
POPULATION_SIZE = 50          # 种群大小（20~100，取50平衡性能）
CHROMOSOME_LEN = 16           # 单个变量二进制编码长度（精度1e-4）
CROSSOVER_RATE = 0.8          # 交叉概率（0.6~0.9，取0.8）
MUTATION_RATE = 0.005         # 变异概率（0.001~0.01，取0.005）
GENERATIONS = 100             # 迭代次数（50~200，取100）
ELITE_SIZE = 1                # 精英保留数量

# ===================== 3. 遗传算法核心函数 =====================
def binary_to_real(binary_chrom, min_val, max_val, chrom_len):
    """二进制编码 → 实数坐标解码"""
    # 将二进制列表转换为整数
    integer = 0
    for bit in binary_chrom:
        integer = (integer << 1) | bit
    # 映射到实数区间
    max_int = (1 << chrom_len) - 1
    real_val = min_val + (integer / max_int) * (max_val - min_val)
    return real_val

def init_population():
    """初始化种群：每个个体 = [x1二进制编码, x2二进制编码]"""
    population = []
    for _ in range(POPULATION_SIZE):
        # 生成x1的二进制串（长度CHROMOSOME_LEN）
        x1_chrom = [np.random.randint(0, 2) for _ in range(CHROMOSOME_LEN)]
        x2_chrom = [np.random.randint(0, 2) for _ in range(CHROMOSOME_LEN)]
        individual = x1_chrom + x2_chrom   # 总长度 2*CHROMOSOME_LEN
        population.append(individual)
    return population

def get_fitness(population):
    """计算种群所有个体的适应度（火力效能值）"""
    fitness = []
    for chrom in population:
        # 拆分x1、x2的染色体
        x1_chrom = chrom[:CHROMOSOME_LEN]
        x2_chrom = chrom[CHROMOSOME_LEN:]
        # 解码为真实坐标
        x1 = binary_to_real(x1_chrom, X1_MIN, X1_MAX, CHROMOSOME_LEN)
        x2 = binary_to_real(x2_chrom, X2_MIN, X2_MAX, CHROMOSOME_LEN)
        # 计算适应度
        fit = fitness_function(x1, x2)
        fitness.append(fit)
    return np.array(fitness)

def select(population, fitness):
    """选择操作：轮盘赌法（适应度越高被选中概率越大）"""
    # 处理负适应度：平移使最小值非负
    min_fit = np.min(fitness)
    if min_fit < 0:
        fitness = fitness - min_fit + 1e-6
    # 计算概率
    total_fit = np.sum(fitness)
    probs = fitness / total_fit
    # 轮盘赌选择种群大小个个体（有放回）
    selected_indices = np.random.choice(len(population), size=POPULATION_SIZE, p=probs)
    new_population = [population[i] for i in selected_indices]
    return new_population

def crossover(population):
    """交叉操作：单点交叉，产生新个体"""
    new_population = []
    # 精英保留：直接保留最优个体（在主流程中单独处理，这里保留所有个体便于后续处理）
    # 实际交叉时，对相邻个体按概率交叉
    for i in range(0, POPULATION_SIZE, 2):
        parent1 = population[i]
        parent2 = population[i+1] if i+1 < POPULATION_SIZE else population[0]
        if np.random.rand() < CROSSOVER_RATE:
            # 随机选择交叉点（1到总长度-1之间）
            cross_point = np.random.randint(1, len(parent1))
            child1 = parent1[:cross_point] + parent2[cross_point:]
            child2 = parent2[:cross_point] + parent1[cross_point:]
            new_population.extend([child1, child2])
        else:
            new_population.extend([parent1, parent2])
    # 如果因为奇数导致多一个，截断到种群大小
    return new_population[:POPULATION_SIZE]

def mutate(population):
    """变异操作：随机位翻转"""
    for i in range(len(population)):
        for j in range(len(population[i])):
            if np.random.rand() < MUTATION_RATE:
                # 翻转该位
                population[i][j] = 1 - population[i][j]
    return population

# ===================== 4. 主进化流程 =====================
def genetic_algorithm():
    # 初始化
    pop = init_population()
    best_fitness_history = []      # 记录每代最优适应度
    avg_fitness_history = []       # 记录每代平均适应度

    # 迭代进化
    for gen in range(GENERATIONS):
        # 计算适应度
        fit = get_fitness(pop)
        best_fit = np.max(fit)
        avg_fit = np.mean(fit)
        best_fitness_history.append(best_fit)
        avg_fitness_history.append(avg_fit)

        # 精英保留：找出当前最优个体
        elite_idx = np.argmax(fit)
        elite = pop[elite_idx].copy()

        # 进化操作
        pop = select(pop, fit)
        pop = crossover(pop)
        pop = mutate(pop)

        # 精英替换：用最优个体替换最差个体（保证精英不丢失）
        worst_idx = np.argmin(get_fitness(pop))
        pop[worst_idx] = elite

        # 每10代打印一次进度
        if (gen + 1) % 10 == 0:
            print(f"第{gen + 1}代 | 最优火力覆盖效能：{best_fit:.6f} | 平均效能：{avg_fit:.6f}")

    # 最终最优个体
    final_fit = get_fitness(pop)
    best_idx = np.argmax(final_fit)
    best_chrom = pop[best_idx]

    # 解码最优坐标
    best_x1 = binary_to_real(best_chrom[:CHROMOSOME_LEN], X1_MIN, X1_MAX, CHROMOSOME_LEN)
    best_x2 = binary_to_real(best_chrom[CHROMOSOME_LEN:], X2_MIN, X2_MAX, CHROMOSOME_LEN)
    best_value = final_fit[best_idx]

    # 最终种群的统计信息
    final_avg_fit = np.mean(final_fit)
    final_best_fit = best_value

    return best_x1, best_x2, best_value, final_avg_fit, best_fitness_history, avg_fitness_history

# ===================== 5. 运行算法 + 结果输出 =====================
if __name__ == "__main__":
    # 执行遗传算法
    x1_opt, x2_opt, max_efficiency, avg_efficiency, best_history, avg_history = genetic_algorithm()

    # 打印最终优化结果
    print("\n" + "=" * 60)
    print("军事火力点部署最优方案")
    print(f"火力点A最优横向坐标：x1 = {x1_opt:.6f} 千米")
    print(f"火力点B最优纵向坐标：x2 = {x2_opt:.6f} 千米")
    print(f"最大火力覆盖综合效能：{max_efficiency:.6f}")
    print(f"最终种群平均效能：{avg_efficiency:.6f}")
    print("=" * 60)

    # 可视化迭代收敛曲线（最优效能 + 平均效能）
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, GENERATIONS + 1), best_history, 'b-', linewidth=2, label='每代最优效能')
    plt.plot(range(1, GENERATIONS + 1), avg_history, 'r--', linewidth=2, label='每代平均效能')
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('火力覆盖效能值', fontsize=12)
    plt.title('遗传算法优化火力部署收敛曲线', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()