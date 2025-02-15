import math
from pyomo.environ import *
import itertools

model = ConcreteModel()

CaptureSites = ['CaptureSite1', 'CaptureSite2', 'CaptureSite3', 'CaptureSite4', 'CaptureSite5', 'CaptureSite6', 'CaptureSite8', 'CaptureSite9', 'CaptureSite10']
TransportHubs = []
StorageSites = ['StorageSite1', 'StorageSite2']
nodes = StorageSites + CaptureSites + TransportHubs
model.N = Set(initialize=CaptureSites + StorageSites + TransportHubs)  # 包含所有站点
model.L = Set(initialize=['truck', 'rail', 'pipeline'])  # 运输方式集合 {truck, rail, pipeline}
rail_connections = [('CaptureSite1', 'CaptureSite3'), ('CaptureSite3', 'CaptureSite1'),
                    ('CaptureSite3', 'CaptureSite4'), ('CaptureSite4', 'CaptureSite3')]
# truck_connections = [
#     ('StorageSite1', 'CaptureSite1'), ('StorageSite1', 'CaptureSite2'),
#     ('StorageSite1', 'CaptureSite3'), ('StorageSite1', 'CaptureSite4'),
#     ('StorageSite1', 'CaptureSite5'), ('StorageSite1', 'CaptureSite6'),
#     ('StorageSite1', 'CaptureSite8'),  ('StorageSite1', 'CaptureSite9'),
#     ('StorageSite1', 'CaptureSite10'),
#     ('CaptureSite1', 'CaptureSite2'), ('CaptureSite1', 'CaptureSite3'),
#     ('CaptureSite1', 'CaptureSite4'), ('CaptureSite1', 'CaptureSite5'),
#     ('CaptureSite2', 'CaptureSite3'), ('CaptureSite2', 'CaptureSite4'),
#     ('CaptureSite2', 'CaptureSite5'),
#     ('CaptureSite3', 'CaptureSite4'), ('CaptureSite3', 'CaptureSite5'),
#     ('CaptureSite4', 'CaptureSite5'),
#     ('StorageSite2', 'CaptureSite1'), ('StorageSite2', 'CaptureSite2'),
#     ('StorageSite2', 'CaptureSite3'), ('StorageSite2', 'CaptureSite4'),
#     ('StorageSite2', 'CaptureSite5'), ('StorageSite2', 'CaptureSite6'),
#     ('StorageSite2', 'CaptureSite8'), ('StorageSite2', 'CaptureSite9'),
#     ('StorageSite2', 'CaptureSite10'),
# ]
# truck_connections = truck_connections + [(j, i) for (i, j) in truck_connections]
truck_connections = [
    ('StorageSite1', 'CaptureSite1'), ('StorageSite1', 'CaptureSite2'),
    ('StorageSite1', 'CaptureSite3'), ('StorageSite1', 'CaptureSite4'),
    ('StorageSite1', 'CaptureSite5'), ('StorageSite1', 'CaptureSite6'),
    ('StorageSite1', 'CaptureSite8'),  ('StorageSite1', 'CaptureSite9'),
    ('StorageSite1', 'CaptureSite10'),
    ('StorageSite2', 'CaptureSite1'), ('StorageSite2', 'CaptureSite2'),
    ('StorageSite2', 'CaptureSite3'), ('StorageSite2', 'CaptureSite4'),
    ('StorageSite2', 'CaptureSite5'), ('StorageSite2', 'CaptureSite6'),
    ('StorageSite2', 'CaptureSite8'), ('StorageSite2', 'CaptureSite9'),
    ('StorageSite2', 'CaptureSite10'),
]

# 定义站点编号
sites = [f'CaptureSite{i}' for i in [1, 2, 3, 4, 5, 6, 8, 9, 10]]

# 生成所有有序的两两组合（即包括相反方向）
pairs = list(itertools.permutations(sites, 2))

truck_connections = truck_connections + pairs
# 定义关键管道连接（示例）
pipeline_connections = [
    # StorageSite1 到所有捕获站点
    ('StorageSite1', 'CaptureSite1'),
    ('StorageSite1', 'CaptureSite2'),
    ('StorageSite1', 'CaptureSite3'),
    ('StorageSite1', 'CaptureSite4'),
    ('StorageSite1', 'CaptureSite5'),
    ('StorageSite1', 'CaptureSite6'),
    ('StorageSite1', 'CaptureSite8'),
    ('StorageSite1', 'CaptureSite9'),
    ('StorageSite1', 'CaptureSite10'),

    # StorageSite2 到所有捕获站点（注意可用时间）
    ('StorageSite2', 'CaptureSite1'),
    ('StorageSite2', 'CaptureSite2'),
    ('StorageSite2', 'CaptureSite3'),
    ('StorageSite2', 'CaptureSite4'),
    ('StorageSite2', 'CaptureSite5'),
    ('StorageSite2', 'CaptureSite6'),
    ('StorageSite2', 'CaptureSite8'),
    ('StorageSite2', 'CaptureSite9'),
    ('StorageSite2', 'CaptureSite10'),

    # # 捕获站点之间的关键连接
    # ('CaptureSite1', 'CaptureSite3'),
    # ('CaptureSite2', 'CaptureSite3'),
    # ('CaptureSite3', 'CaptureSite4'),
    # ('CaptureSite4', 'CaptureSite5'),
    # ('CaptureSite4', 'CaptureSite6'),
    # ('CaptureSite4', 'CaptureSite8'),
    # ('CaptureSite4', 'CaptureSite9'),
    # ('CaptureSite4', 'CaptureSite10'),
    #
]
pipeline_connections = pipeline_connections + pairs
# pipeline_connections = pipeline_connections + [(j, i) for (i, j) in pipeline_connections]
# 生成所有可能的节点对 (i,j)，其中 i < j 避免重复
# 允许双向连接但统一管理
# pipeline_connections = [
#     (i, j) for i in nodes for j in nodes if i != j
# ]

# 绑定到模型集合
model.P = Set(dimen=2, initialize=pipeline_connections)

model.A = Set(dimen=2, initialize=rail_connections + truck_connections + pipeline_connections)  # 运输连接集合 二维

model.T = RangeSet(0, 25)  # 规划期 0~25年

E_values = {
    'CaptureSite1': 200,
    'CaptureSite2': 250,
    'CaptureSite3': 400,
    'CaptureSite4': 200,
    'CaptureSite5': 350,
    'CaptureSite6': 50,
    'CaptureSite8': 65,
    'CaptureSite9': 90,
    'CaptureSite10': 45,
    'StorageSite1': 0,  # 存储站点不会排放 CO₂
    'StorageSite2': 0  # 存储站点不会排放 CO₂
}

model.E = Param(model.N, initialize=E_values, within=NonNegativeReals)  # 每个站点的 CO₂ 排放量


Q_init_value = sum(E_values[i] for i in E_values)
model.Q_init = Param(initialize=Q_init_value, within=NonNegativeReals)  # 初始总排放量
Q_T_value = Q_init_value * 0.2  # 80% 减排目标(这里是我根据欧盟2050年计划设定的 可以你们再找找看)
model.Q_T = Param(initialize=Q_T_value, within=NonNegativeReals)  # w目标排放量

beta_1 = {'truck': 5, 'rail': 1.6, 'pipeline': 0.08}
beta_2 = {'truck': 44, 'rail': 31, 'pipeline': 0}

model.alpha_1 = Param(initialize=0.0005, within=NonNegativeReals, mutable=True)
model.alpha_2 = Param(initialize=3000, within=NonNegativeReals, mutable=True)

################################################
# 是否有铁路连接
w_rail_values = {(i, j): 1 if (i, j) in rail_connections else 0 for (i, j) in model.A}
model.w_rail = Param(model.A, initialize=w_rail_values, within=Binary)

################################################
# 运输距离
node_coords = {
    'CaptureSite1': (9, 10),
    'CaptureSite2': (5, 7),
    'CaptureSite3': (10, 4),
    'CaptureSite4': (5, 2),
    'CaptureSite5': (0, 0),
    'CaptureSite6': (4.5, 1),
    'CaptureSite8': (8.5, 0),
    'CaptureSite9': (2, 6),
    'CaptureSite10': (7, 8),
    'StorageSite1': (0, 10),
    'StorageSite2': (6, 4)
}
L_ij_values = {(i, j): math.sqrt((node_coords[i][0] - node_coords[j][0]) ** 2 +
                                 (node_coords[i][1] - node_coords[j][1]) ** 2)
               for (i, j) in model.A}
model.L_ij = Param(model.A, initialize=L_ij_values, within=NonNegativeReals)


################################################
# 运输排放系数
gamma_values = {'truck': 0.0007, 'rail': 0.0001, 'pipeline': 0}
model.gamma = Param(model.L, initialize=gamma_values, within=NonNegativeReals)
# 储存站点的可用性
storage_availability_t = {
    'StorageSite1': 0,  # 从 t=0 开始可用
    'StorageSite2': 14  # 从 t=14 开始可用
}
####################################################################
# 3. Variables (决策变量)
####################################################################

# 是否在 t 年安装碳捕集设备
# x[i, t] 表示在时间 t，站点 i 对于碳捕捉设备  0: 不安装 1: 安装
model.x = Var(model.N, model.T, within=Binary)

# CO2运输流量
# f[i, j, l, t] 表示在时间 t，使用 l 运输方式（卡车、铁路、管道），从 站点 i 到 j 运输的 CO2 流量
model.f = Var(model.A, model.L, model.T, within=NonNegativeReals, bounds=(0, 1e6))
# 是否建造管道
# s[i, j, t] 表示在时间 t，从站点 i 到 j 是否建造管道  0: 不建造 1: 建造
model.s = Var(model.P, model.T, within=Binary)
# 管道容量应与时间和连接关联
model.S = Var(model.P, model.T, within=NonNegativeReals, bounds=(0, 5000))

################################################################
# 4. Objective Function (目标函数：最小化总成本)
################################################################
def total_cost(model):
    transport_cost = sum((beta_1[l] * model.f[i, j, l, t] * model.L_ij[i, j] + beta_2[l] * model.f[i, j, l, t])
                         for (i, j) in model.A for l in model.L for t in model.T)
    pipeline_cost = sum(
        (model.alpha_1 * model.S[i, j, t] + model.alpha_2 * model.L_ij[i, j]) * model.s[i, j, t]
        for (i, j) in model.A  # 遍历所有连接
        for t in model.T  # 显式遍历时间维度
    )
    return transport_cost + pipeline_cost


model.obj = Objective(rule=total_cost, sense=minimize)


################################################################
# 5. Constraints (约束)
################################################################
# 流平衡约束
# 进入站点的流量 - 离开站点的流量 = 站点剩余排放量
# 修改后的流平衡约束
def flow_balance(model, i, t):
    if i in CaptureSites:
        # 对捕获站点，我们在 t=0 时刻跳过流平衡约束（初始状态不运输）
        if t == 0:
            return Constraint.Skip
        else:
            # 计算该时段的入流和出流
            inflow = sum(model.f[j, i, l, t] for (j, i) in model.A for l in model.L)
            outflow = sum(model.f[i, j, l, t] for (i, j) in model.A for l in model.L)
            # 捕获站点需要将排放的 CO₂（扣除90%捕获率）输送出去
            return outflow - inflow == model.E[i] * (1 - 0.9 * model.x[i, t])
    elif i in StorageSites:
        # 对于存储站点：如果站点暂时不可用，则禁止流入；否则允许流入
        if (i == 'StorageSite2' and t < storage_availability_t[i]):
            return sum(model.f[j, i, l, t] for (j, i) in model.A for l in model.L) == 0
        else:
            return sum(model.f[j, i, l, t] for (j, i) in model.A for l in model.L) >= 0
    else:
        return Constraint.Skip

# 将修改后的流平衡约束绑定到模型
model.flow_balance_constraint = Constraint(model.N, model.T, rule=flow_balance)


# 管道建设决策与容量绑定（避免容量变量独立于建设时间）
def pipeline_capacity_link(model, i, j, t):
    return model.S[i,j,t] <= 1e6 * model.s[i,j,t]  # 容量仅在建设时生效
model.pipe_cap_link = Constraint(model.A, model.T, rule=pipeline_capacity_link)

# 正确容量约束：流量不超过已建设管道的总容量
def pipeline_capacity(model, i, j, t):
    return model.f[i,j,'pipeline',t] <= sum(model.S[i,j,tau] for tau in range(t+1))
model.pipe_capacity_constraint = Constraint(model.A, model.T, rule=pipeline_capacity)



# 铁路运输方式约束
def rail_availability(model, i, j, t):
    return model.f[i, j, 'rail', t] <= model.w_rail[i, j] * 1e6


model.rail_constraint = Constraint(model.A, model.T, rule=rail_availability)


#####################################################################

# def max_transport_capacity(model, i, j, l, t):
#     return model.f[i, j, l, t] <= 1000  # 限制单个连接上的运输量

# model.max_transport_capacity_constraint = Constraint(model.A, model.L, model.T, rule=max_transport_capacity)

def max_transport_capacity(model, i, j, l, t):
    if l == 'pipeline':
        return model.f[i, j, l, t] <= model.S[i, j] * model.s[i, j, t]
    else:
        return Constraint.Skip

    # 确保总排放量每年降低


model.Q = Var(model.T, within=NonNegativeReals, bounds=(0, 1e6))  # 记录每年的总排放量

def emission_reduction(model, t):
    # 总排放 = 未捕获排放 + 运输排放
    capture_emissions = sum(model.E[i]*(1-0.9*model.x[i,t]) for i in CaptureSites)
    transport_emissions = sum(model.gamma[l] * model.f[i,j,l,t] * model.L_ij[i,j]
                         for (i,j) in model.A for l in model.L)
    return model.Q[t] == capture_emissions + transport_emissions

model.emission_calc_constraint = Constraint(model.T, rule=emission_reduction)

# 管道容量不可逆性约束
def pipeline_capacity_fix(model, i, j, t):
    if t >= 1:
        return model.S[i,j,t] >= model.S[i,j,t-1]  # 容量只能保持或增加
    else:
        return Constraint.Skip
model.pipeline_capacity_fix = Constraint(model.A, model.T, rule=pipeline_capacity_fix)

# 存储站点可用性绑定
def storage_pipeline_link(model, i, j, t):
    if j in StorageSites and t < storage_availability_t[j]:
        return model.s[i,j,t] == 0  # 存储站点不可用时禁止连接
    else:
        return Constraint.Skip
model.storage_pipeline_link = Constraint(model.A, model.T, rule=storage_pipeline_link)

def emission_limit(model, t):
    return model.Q[t] <= model.Q_init - t / 25 * (model.Q_init - model.Q_T) + 100

model.emission_limit_constraint = Constraint(model.T, rule=emission_limit)
# 示例：最小管道容量约束
def min_pipeline_capacity_constraint(model, i, j, t):
    if (i, j) in model.P:
        return model.S[i,j,t] >= 50 * model.s[i,j,t]
    else:
        return Constraint.Skip

model.min_pipeline_capacity_constraint = Constraint(model.P, model.T, rule=min_pipeline_capacity_constraint)


def total_storage_capacity(model, t):
    # 找出所有目的节点为存储站点的连接列表
    relevant_arcs = [(i, j, l) for (i, j) in model.A for l in model.L if j in StorageSites]
    if not relevant_arcs:
        return Constraint.Skip  # 如果没有这样的连接，则跳过该约束
    rhs = model.Q_init * (1 - t / 25) * 0.4
    return sum(model.f[i, j, l, t] for (i, j, l) in relevant_arcs) >= max(0, rhs)


# def limited_pipeline_construction(model, i, j):
#     return sum(model.s[i, j, t] for t in model.T) <= 3  # 限制最多建造 3 条管道
# model.limited_pipeline_construction_constraint = Constraint(model.A, rule=limited_pipeline_construction)
# 储存点的可用性
def storage_avail(model, i, t):
    if i == "StorageSite2" and t < 14:
        return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) == 0
    elif i in StorageSites:
        return Constraint.Feasible  # 这里表示该约束在该索引下始终满足
    else:
        return Constraint.Skip

def pipeline_continuity(model, i, j, t):
    if t > 0:
        return model.s[i, j, t] >= model.s[i, j, t-1]  # 只增不减
    else:
        return Constraint.Feasible

model.pipeline_continuity_constraint = Constraint(model.A, model.T, rule=pipeline_continuity)
def pipeline_capacity(model, i, j, t):
    return model.f[i, j, 'pipeline', t] <= model.S[i, j, t] * sum(model.s[i, j, tau] for tau in range(t+1))
model.pipe_capacity_constraint = Constraint(model.A, model.T, rule=pipeline_capacity)
def pipeline_one_time_build(model, i, j):
    return sum(model.s[i, j, t] for t in model.T) <= 1  # 只允许建造一次

model.pipeline_one_time_build_constraint = Constraint(model.A, rule=pipeline_one_time_build)

solver = SolverFactory('gurobi')

results = solver.solve(model, tee=True, options={
    'ResultFile': 'gurobi_solution.sol',
    'DualReductions': 0,
    'Presolve': 2,
    'NumericFocus': 1,
    'MIPGap': 0.03,
    'TimeLimit': 1200,
    'Threads': 6,
    'Cuts': 3,
    'Heuristics': 0.15,
    'MIPFocus': 1
})  # 生成 Gurobi 诊断文件

if results.solver.termination_condition == TerminationCondition.infeasible:
    print('Model is infeasible')
    model.iis = SolverFactory('gurobi')
    model.iis.solve(model, tee=True, options={'ResultFile': 'gurobi_iis.ilp'})

    with open('gurobi_iis.ilp', 'r') as f:
        print(f.read())


