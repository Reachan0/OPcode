import math
from pyomo.environ import *

model = ConcreteModel()

CaptureSites = ['CaptureSite1', 'CaptureSite2', 'CaptureSite3', 'CaptureSite4', 'CaptureSite5']
TransportHubs = []
StorageSites = ['StorageSite1']

model.N = Set(initialize = ['CaptureSite1', 'CaptureSite2', 'CaptureSite3',
                            'CaptureSite4', 'CaptureSite5', 'StorageSite1'])    # 站点集合（instance1包括 N_C=5, N_S=1）
model.L = Set(initialize = ['truck', 'rail', 'pipeline'])                       # 运输方式集合 {truck, rail, pipeline}
rail_connections = [('CaptureSite1', 'CaptureSite3'), ('CaptureSite3', 'CaptureSite1'),
                    ('CaptureSite3', 'CaptureSite4'), ('CaptureSite4', 'CaptureSite3')]
truck_connections = [('StorageSite1', 'CaptureSite1'), ('StorageSite1', 'CaptureSite2'),
                     ('StorageSite1', 'CaptureSite3'), ('StorageSite1', 'CaptureSite4'), ('StorageSite1', 'CaptureSite5'),
                     ('CaptureSite1', 'CaptureSite2'), ('CaptureSite1', 'CaptureSite3'), ('CaptureSite1', 'CaptureSite4'), ('CaptureSite1', 'CaptureSite5'),
                     ('CaptureSite2', 'CaptureSite3'), ('CaptureSite2', 'CaptureSite4'), ('CaptureSite2', 'CaptureSite5'),
                     ('CaptureSite3', 'CaptureSite4'), ('CaptureSite3', 'CaptureSite5'),
                     ('CaptureSite4', 'CaptureSite5')]
pipeline_connections = [('StorageSite1', 'CaptureSite3'), ('CaptureSite3', 'StorageSite1'),
                        ('CaptureSite3', 'CaptureSite5'), ('CaptureSite5', 'CaptureSite3'),
                        ('CaptureSite5', 'StorageSite1'), ('StorageSite1', 'CaptureSite5')]
model.A = Set(dimen=2, initialize=rail_connections + truck_connections + pipeline_connections)  # 运输连接集合 二维
model.T = RangeSet(0, 25)  # 规划期 0~25年

####################################################################
# 2. Parameters (参数)
####################################################################
E_values = {
    'CaptureSite1': 200,
    'CaptureSite2': 250,
    'CaptureSite3': 400,
    'CaptureSite4': 200,
    'CaptureSite5': 350,
    'StorageSite1': 0  # 存储站点不会排放 CO₂
}

model.E = Param(model.N, initialize=E_values, within=NonNegativeReals)  # 每个站点的 CO₂ 排放量

####################################
# 计算初始排放总量和目标排放总量
Q_init_value = sum(E_values[i] for i in E_values)  
model.Q_init = Param(initialize = Q_init_value, within=NonNegativeReals)  # 初始总排放量
Q_T_value = Q_init_value * 0.2  # 80% 减排目标(这里是我根据欧盟2050年计划设定的 可以你们再找找看)
model.Q_T = Param(initialize=Q_T_value, within=NonNegativeReals)       # 目标排放量

##########################################
# 运输成本参数
beta_1 = {'truck': 1000000, 'rail': 1, 'pipeline': 0.05}
beta_2 = {'truck': 0, 'rail': 0.5, 'pipeline': 0}
# C_trans_values = {(i, j, l): beta_1[l] * model.L_ij[i, j] + beta_2[l] for (i, j) in model.A for l in model.L}
# model.C_trans = Param(model.A, model.L, initialize=C_trans_values, within=NonNegativeReals)

##############################################
# 管道建设成本
alpha_1 = 0.3
alpha_2 = 27
# C_pipe_values = {(i, j, t): (alpha_1 * model.S[i, j] + alpha_2 * model.L_ij[i, j]) * model.s[i, j, t] 
#                  for (i, j) in model.A for t in model.T}
# model.C_pipe = Param(model.A, initialize=C_pipe_values, within=NonNegativeReals)

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
    'StorageSite1': (0, 10)
}
L_ij_values = {(i, j): math.sqrt((node_coords[i][0] - node_coords[j][0])**2 +
                                 (node_coords[i][1] - node_coords[j][1])**2)
               for (i, j) in model.A}
model.L_ij = Param(model.A, initialize=L_ij_values, within=NonNegativeReals)

################################################
# 运输排放系数
gamma_values = {'truck': 0, 'rail': 0.003, 'pipeline': 0}
model.gamma = Param(model.L, initialize=gamma_values, within=NonNegativeReals)

####################################################################
# 3. Variables (决策变量)
####################################################################

# 是否在 t 年安装碳捕集设备 
# x[i, t] 表示在时间 t，站点 i 对于碳捕捉设备  0: 不安装 1: 安装
model.x = Var(model.N, model.T, within=Binary)  

# CO2运输流量
# f[i, j, l, t] 表示在时间 t，使用 l 运输方式（卡车、铁路、管道），从 站点 i 到 j 运输的 CO2 流量
model.f = Var(model.A, model.L, model.T, within=NonNegativeReals)  

# 是否建造管道
# s[i, j, t] 表示在时间 t，从站点 i 到 j 是否建造管道  0: 不建造 1: 建造
model.s = Var(model.A, model.T, within=Binary)    

# 管道容量
model.S = Var(model.A, within=NonNegativeReals)  

################################################################
# 4. Objective Function (目标函数：最小化总成本)
################################################################
def total_cost(model):
    transport_cost = sum((beta_1[l] * model.f[i, j, l, t] * model.L_ij[i, j] + beta_2[l] * model.f[i, j, l, t])
                            for (i, j) in model.A  for l in model.L  for t in model.T)
    
    pipeline_cost = sum((alpha_1 * model.S[i, j] + alpha_2 * model.L_ij[i, j]) * model.s[i, j, t]
                         for (i, j) in model.A for t in model.T)
    
    return transport_cost + pipeline_cost

model.obj = Objective(rule=total_cost, sense=minimize)

################################################################
# 5. Constraints (约束)
################################################################
# 流平衡约束
# 进入站点的流量 - 离开站点的流量 = 站点剩余排放量
def flow_balance(model, i, t):
    if i in CaptureSites:
        return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) - \
                sum(model.f[i, j, l, t] for j in model.N for l in model.L if (i, j) in model.A) == \
                model.E[i] * (1 - 0.9 * model.x[i, t]) - 10
    
    elif i in StorageSites:
        return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) >= 0  
    
    elif i in TransportHubs:
        return sum(model.f[j, i, l, t] for j in model.N for l in model.L if (j, i) in model.A) - \
                sum(model.f[i, j, l, t] for j in model.N for l in model.L if (i, j) in model.A) == 0

model.flow_balance_constraint = Constraint(model.N, model.T, rule=flow_balance)

# 铁路运输方式约束
def rail_availability(model, i, j, t):
    return model.f[i, j, 'rail', t] <= model.w_rail[i, j] * 1e6  

model.rail_constraint = Constraint(model.A, model.T, rule=rail_availability)

# 管道运输不能超过管道容量 s[i, j, t] = 1 即建造了管道时 才可以使用管道运输
def pipeline_capacity(model, i, j, t):
    return model.f[i, j, 'pipeline', t] <= sum(model.s[i, j, tau] for tau in range(0, t+1)) * model.S[i, j]
    
model.pipe_capacity_constraint = Constraint(model.A, model.T, rule=pipeline_capacity)

# 对于碳捕集设备 只能安装一次
def single_installation(model, i):
    return sum(model.x[i, t] for t in model.T) <= 3

model.single_installation_constraint = Constraint(model.N, rule=single_installation)

def installation_persistence(model, i, t):
    if t > 0:
        return model.x[i, t] >= 0 # 允许设备在任何年份自由开关
    else:
        return Constraint.Skip
    
model.installation_persistence_constraint = Constraint(model.N, model.T, rule=installation_persistence)

def capture_effectiveness(model, i, t):
    return sum(model.f[i, j, l, t] for j in model.N for l in model.L if (i, j) in model.A) <= model.E[i] \
        * (1 - 0.9 * model.x[i, t])

model.capture_effectiveness_constraint = Constraint(model.N, model.T, rule=capture_effectiveness)

# 确保总排放量每年降低
model.Q = Var(model.T, within=NonNegativeReals)  # 记录每年的总排放量
def emission_reduction(model, t):
    return model.Q[t] == sum(model.E[i] * (1 - 0.9 * model.x[i, t]) for i in model.N)
model.emission_calc_constraint = Constraint(model.T, rule=emission_reduction)

# def emission_limit(model, t):
#     return model.Q[t] <= model.Q_init - t / 25 * (model.Q_init - model.Q_T)
# def emission_limit(model, t):
#     return model.Q[t] <= model.Q_init - (t / 25) * (model.Q_init - model.Q_T) + 0.05 * model.Q_init
def emission_limit(model, t):
    return model.Q[t] <= model.Q_init * (1 - (t / 25)**1.1)  # 允许前期排放慢点减少


model.emission_limit_constraint = Constraint(model.T, rule=emission_limit)

def enforce_pipeline_construction(model, i, j):
    return sum(model.s[i, j, t] for t in model.T) <=5  # 至少建造一次

model.pipeline_construction_constraint = Constraint(model.A, rule=enforce_pipeline_construction)

solver = SolverFactory('gurobi')  
#results = solver.solve(model, tee=True, options={'DualReductions': 0})  # 关闭对偶约减，避免 Gurobi 误判

results = solver.solve(model, tee=True, options={'ResultFile': 'gurobi_infeasible.sol'})  # 生成 Gurobi 诊断文件


# 检查碳捕集设备安装变量 x[i, t]
for i in model.N:
    for t in model.T:
        print(f"x[{i}, {t}] = {model.x[i, t].value}")

# 检查 CO₂ 运输流量 f[i, j, l, t]
for (i, j) in model.A:
    for l in model.L:
        for t in model.T:
            print(f"f[{i}, {j}, {l}, {t}] = {model.f[i, j, l, t].value}")

# 检查年排放总量 Q[t]
for t in model.T:
    print(f"Q[{t}] = {model.Q[t].value}")


# print("Solver status:", results.solver.status)
# print("Termination condition:", results.solver.termination_condition)


