import geatpy as ea
import numpy as np

Q = 0
W = 0 
# 构建问题
r = 1  # 目标函数需要用到的额外数据
@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）
    global Q
    loga = Vars[0]
    b    = Vars[1]
    
    Qhat = (np.log(W) - loga) / b
    f = np.sum((Q - Qhat) ** 2)  # 计算目标函数值

    CV = np.array(np.sum(1/b)**2)  # 计算违反约束程度

    return f, CV


problem = ea.Problem(name='AMHG',
                        M=1,  # 目标维数
                        maxormins=[-1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                        Dim=2,  # 决策变量维数
                        varTypes=[0, 0],  # 决策变量的类型列表，0：实数；1：整数
                        lb=[-999,-999],  # 决策变量下界
                        ub=[999, 999],  # 决策变量上界
                        evalVars=evalVars)
# 构建算法
algorithm = ea.soea_SEGA_templet(problem,
                                    ea.Population(Encoding='RI', NIND=20),
                                    MAXGEN=50,  # 最大进化代数。
                                    logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                    trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                    maxTrappedCount=10)  # 进化停滞计数器最大上限值。

## 求解
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, 
                  outputMsg=True, drawLog=False, saveFlag=True, 
                  dirName='result')

help(ea.problem)