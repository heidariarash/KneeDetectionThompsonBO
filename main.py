import matplotlib.pyplot  as plt

from solver   import solver
from config   import Setting



rhvs = []
rd2l = []
while(len(rhvs) < 1):
    try:
        print(len(rhvs))
        regret_d2l, regret_hv, model, dataset = solver(Setting.problem, Setting.method, reps = 80)
        rhvs.append(regret_hv)
        rd2l.append(regret_d2l)
    except KeyboardInterrupt:
        break