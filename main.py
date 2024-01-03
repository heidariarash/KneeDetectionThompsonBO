import numpy as np

from solver import solver
from config import Setting

rhvs = []
rd2l = []
while(len(rhvs) < Setting.repetitions):
    try:
        print(len(rhvs))
        regret_d2l, regret_hv, model, dataset = solver(Setting.problem, Setting.method, reps = 80)
        rhvs.append(regret_hv)
        rd2l.append(regret_d2l)
    except:
        pass

np.save(f"./HV Regret of {Setting.problem} with {Setting.knees} knee(s) and input dimension of {Setting.n_var}", rhvs)
np.save(f"./Distance Regret of {Setting.problem} with {Setting.knees} knee(s) and input dimension of {Setting.n_var}", rd2l)