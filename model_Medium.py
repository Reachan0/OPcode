import math
from pyomo.environ import *

model = ConcreteModel()

CaptureSites = ['CaptureSite1', 'CaptureSite2', 'CaptureSite3', 'CaptureSite4', 'CaptureSite5', 'CaptureSite6', 'CaptureSite8', 'CaptureSite9', 'CaptureSite10']
TransportHubs = []
StorageSites = ['StorageSite1', 'StorageSite2']
nodes = StorageSites + CaptureSites + TransportHubs
# model.N = Set(initialize=['CaptureSite1', 'CaptureSite2', 'CaptureSite3',
#                           'CaptureSite4', 'CaptureSite5', 'StorageSite1'])  # 站点集合（instance1包括 N_C=5, N_S=1）
model.N = Set(initialize=CaptureSites + StorageSites)  # 包含所有站点

model.L = Set(initialize=['truck', 'rail', 'pipeline'])  # 运输方式集合 {truck, rail, pipeline}
rail_connections = [('CaptureSite1', 'CaptureSite3'), ('CaptureSite3', 'CaptureSite1'),
                    ('CaptureSite3', 'CaptureSite4'), ('CaptureSite4', 'CaptureSite3')]
truck_connections = [
    ('StorageSite1', 'CaptureSite1'), ('StorageSite1', 'CaptureSite2'),
    ('StorageSite1', 'CaptureSite3'), ('StorageSite1', 'CaptureSite4'),
    ('StorageSite1', 'CaptureSite5'),
    ('CaptureSite1', 'CaptureSite2'), ('CaptureSite1', 'CaptureSite3'),
    ('CaptureSite1', 'CaptureSite4'), ('CaptureSite1', 'CaptureSite5'),
    ('CaptureSite2', 'CaptureSite3'), ('CaptureSite2', 'CaptureSite4'),
    ('CaptureSite2', 'CaptureSite5'),
    ('CaptureSite3', 'CaptureSite4'), ('CaptureSite3', 'CaptureSite5'),
    ('CaptureSite4', 'CaptureSite5'),
    ('StorageSite2', 'CaptureSite1'), ('StorageSite2', 'CaptureSite2'),
    ('StorageSite2', 'CaptureSite3'), ('StorageSite2', 'CaptureSite4'),
    ('StorageSite2', 'CaptureSite5'), ('StorageSite2', 'CaptureSite6'),
    ('StorageSite2', 'CaptureSite8'), ('StorageSite2', 'CaptureSite9'),
    ('StorageSite2', 'CaptureSite10'),
    ('StorageSite1', 'CaptureSite6'), ('CaptureSite6', 'StorageSite1'),
    ('StorageSite1', 'CaptureSite8'), ('CaptureSite8', 'StorageSite1'),
    ('StorageSite1', 'CaptureSite9'), ('CaptureSite9', 'StorageSite1'),
    ('StorageSite1', 'CaptureSite10'), ('CaptureSite10', 'StorageSite1')
]
truck_connections = truck_connections + [(j, i) for (i, j) in truck_connections]

pipeline_connections = [
    ('StorageSite1', 'CaptureSite1'), ('StorageSite1', 'CaptureSite2'),
    ('StorageSite1', 'CaptureSite3'), ('StorageSite1', 'CaptureSite4'),
    ('StorageSite1', 'CaptureSite5'), ('StorageSite1', 'StorageSite2'),
    ('CaptureSite2', 'CaptureSite3'), ('CaptureSite3', 'CaptureSite4'),
    ('CaptureSite4', 'CaptureSite5')
]
pipeline_connections = pipeline_connections + [(j, i) for (i, j) in pipeline_connections]


model.A = Set(dimen=2, initialize=rail_connections + truck_connections + pipeline_connections)  # 运输连接集合 二维
# **检查 `StorageSite2` 是否有输入路径**
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
model.s = Var(model.A, model.T, within=Binary)

# 管道容量
model.S = Var(model.A, within=NonNegativeReals, bounds=(0, 5000))  # 下界为0



################################################################
# 4. Objective Function (目标函数：最小化总成本)
################################################################
def total_cost(model):
    transport_cost = sum((beta_1[l] * model.f[i, j, l, t] * model.L_ij[i, j] + beta_2[l] * model.f[i, j, l, t])
                         for (i, j) in model.A for l in model.L for t in model.T)

    pipeline_cost = sum((model.alpha_1 * model.S[i, j] + model.alpha_2 * model.L_ij[i, j]) *
                        sum(model.s[i, j, t] for t in model.T)  # 每年都支付管道维护费用
                        for (i, j) in model.A)

    return transport_cost + pipeline_cost


model.obj = Objective(rule=total_cost, sense=minimize)


################################################################
# 5. Constraints (约束)
################################################################
# 流平衡约束
# 进入站点的流量 - 离开站点的流量 = 站点剩余排放量
def flow_balance(model, i, t):
    if i in CaptureSites:
        # 正确公式：E[i] * (1 - 0.9 x[i,t])
        return sum(model.f[j,i,l,t] for (j,i) in model.A for l in model.L) - \
               sum(model.f[i,j,l,t] for (i,j) in model.A for l in model.L) == \
               model.E[i] * (1 - 0.9 * model.x[i, t])  # ✅ 关键修正

    # **如果 `i` 是存储站 (Storage Site)**
    elif i in StorageSites:
        if t < storage_availability_t[i]:  # 🚨 `StorageSite2` 在 `t<14` 时不可用
            return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) == 0
        else:
            return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) - \
                   sum(model.f[i, j, l, t] for j in model.N for l in model.L if (i, j) in model.A) == 0

    # **如果 `i` 是运输枢纽 (Transport Hub)**
    elif i in TransportHubs:
        return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) - \
               sum(model.f[i, j, l, t] for j in model.N for l in model.L if (i, j) in model.A) == 0

    else:
        return Constraint.Skip  # ✅ 避免 Pyomo 解析 `True/False`

# 绑定到 Pyomo 模型
model.flow_balance_constraint = Constraint(model.N, model.T, rule=flow_balance)

# 铁路运输方式约束
def rail_availability(model, i, j, t):
    return model.f[i, j, 'rail', t] <= model.w_rail[i, j] * 1e6


model.rail_constraint = Constraint(model.A, model.T, rule=rail_availability)


#####################################################################
def pipeline_capacity(model, i, j, t):
    return model.f[i, j, 'pipeline', t] <= model.S[i, j] * model.s[i, j, t] + 0.05 * model.S[i, j]


model.pipe_capacity_constraint = Constraint(model.A, model.T, rule=pipeline_capacity)


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
    if t == 0:
        return model.Q[t] == model.Q_init
    else:
        return model.Q[t] == sum(model.E[i] * (1 - 0.9 * model.x[i, t]) for i in model.N) + \
            sum(model.gamma[l] * model.f[i, j, l, t] * model.L_ij[i, j] for (i, j) in model.A for l in model.L)


model.emission_calc_constraint = Constraint(model.T, rule=emission_reduction)


def emission_limit(model, t):
    return model.Q[t] <= model.Q_init - t / 25 * (model.Q_init - model.Q_T) + 100


model.emission_limit_constraint = Constraint(model.T, rule=emission_limit)


def min_pipeline_capacity(model, i, j):
    return model.S[i,j] >= 50 * sum(model.s[i,j,t] for t in model.T)  # 仅建造时生效


model.min_pipeline_capacity_constraint = Constraint(model.A, rule=min_pipeline_capacity)


def total_storage_capacity(model, t):
    rhs = model.Q_init * (1 - t / 25) * 0.4
    return sum(model.f[i,j,l,t] for (i,j) in model.A for l in model.L if j in StorageSites) >= max(0, rhs)


model.total_storage_capacity_constraint = Constraint(model.T, rule=total_storage_capacity)

# def limited_pipeline_construction(model, i, j):
#     return sum(model.s[i, j, t] for t in model.T) <= 3  # 限制最多建造 3 条管道
# model.limited_pipeline_construction_constraint = Constraint(model.A, rule=limited_pipeline_construction)
# 储存点的可用性
def storage_avail(model, i, t):
    if i == "StorageSite2" and t < 14:
        return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) == 0
    elif i in StorageSites:
        return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) >= 0
    else:
        return Constraint.Skip

model.storage_avail_constraint = Constraint(model.N, model.T, rule=storage_avail)

def pipeline_continuity(model, i, j, t):
    if t > 0:
        return model.s[i, j, t] >= model.s[i, j, t-1]  # 只增不减
    else:
        return Constraint.Feasible

model.pipeline_continuity_constraint = Constraint(model.A, model.T, rule=pipeline_continuity)
def pipeline_capacity(model, i, j, t):
    return model.f[i, j, 'pipeline', t] <= model.S[i, j] * sum(model.s[i, j, tau] for tau in range(t+1))

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


