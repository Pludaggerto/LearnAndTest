import geatpy as ea
import numpy as np

Q = 0
W = 0 
# ��������
r = 1  # Ŀ�꺯����Ҫ�õ��Ķ�������
@ea.Problem.single
def evalVars(Vars):  # ����Ŀ�꺯������Լ����
    global Q
    loga = Vars[0]
    b    = Vars[1]
    
    Qhat = (np.log(W) - loga) / b
    f = np.sum((Q - Qhat) ** 2)  # ����Ŀ�꺯��ֵ

    CV = np.array(np.sum(1/b)**2)  # ����Υ��Լ���̶�

    return f, CV


problem = ea.Problem(name='AMHG',
                        M=1,  # Ŀ��ά��
                        maxormins=[-1],  # Ŀ����С��󻯱���б�1����С����Ŀ�ꣻ-1����󻯸�Ŀ��
                        Dim=2,  # ���߱���ά��
                        varTypes=[0, 0],  # ���߱����������б�0��ʵ����1������
                        lb=[-999,-999],  # ���߱����½�
                        ub=[999, 999],  # ���߱����Ͻ�
                        evalVars=evalVars)
# �����㷨
algorithm = ea.soea_SEGA_templet(problem,
                                    ea.Population(Encoding='RI', NIND=20),
                                    MAXGEN=50,  # ������������
                                    logTras=1,  # ��ʾÿ�����ٴ���¼һ����־��Ϣ��0��ʾ����¼��
                                    trappedValue=1e-6,  # ��Ŀ���Ż�����ͣ�͵��ж���ֵ��
                                    maxTrappedCount=10)  # ����ͣ�ͼ������������ֵ��

## ���
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, 
                  outputMsg=True, drawLog=False, saveFlag=True, 
                  dirName='result')

help(ea.problem)