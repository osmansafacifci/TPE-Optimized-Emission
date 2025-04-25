#Below code optimizes the materials selection for core and shell in addition to
#core radius and shell thickness to minimize absorption-to-extinction ratio and
#scattering asymmetry parameter. It utilizes TPE, GA, PSO, and GPBO algorithms
#The materials database is curated from refractiveindex.info website
#Lowest achieved value and evaluation speed for each algorithm is given at the end.
#Currently, it optimizes for the case of polymeric medium and if vacuum medium
#optimization is needed the only thing to change in the code is nMedium.
#Requires Optuna and pymoo libraries to run.

import numpy as np
import optuna
import pandas as pd
from scattnlay import scattnlay
from statistics import mean
import torch
import time
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

# Create a custom callback to store evaluations
class HistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data = []
        self.best_so_far = float('inf')
        self.best_so_far_history = []
        
    def notify(self, algorithm):
        # Store a copy of each evaluation
        for ind in algorithm.pop:
            if ind.F is not None:  # Make sure the individual has been evaluated
                val = ind.F[0]
                self.data.append((ind.X.copy(), val))
                self.best_so_far = min(self.best_so_far, val)
                self.best_so_far_history.append(self.best_so_far)


def load_material_data():
    material_params = {}
    material_files = [
        ('nk_interpolation/Al2O3.csv', 'al2o3'),
        ('nk_interpolation/Cu2O.csv', 'cu2o'),
        ('nk_interpolation/Fe2O3.csv', 'fe2o3'),
        ('nk_interpolation/HfO2.csv', 'hfo2'),
        ('nk_interpolation/SiO2.csv', 'sio2'),
        ('nk_interpolation/TiO2.csv', 'tio2'),
        ('nk_interpolation/ZnO.csv', 'zno'),
        ('nk_interpolation/BaF2_nk.csv', 'baf2'),
        ('nk_interpolation/SiC_nk.csv', 'sic'),
        ('nk_interpolation/AlN_nk.csv', 'aln'),
        ('nk_interpolation/Cu_nk.csv', 'cu'),
        ('nk_interpolation/Al_nk.csv', 'al'),
        ('nk_interpolation/Ni_nk.csv', 'ni'),
        ('nk_interpolation/Au_nk.csv', 'au'),
        ('nk_interpolation/Ag_nk.csv', 'ag')
    ]

    for i, (file_path, material_name) in enumerate(material_files):
        df = pd.read_csv(file_path)
        df.drop(df.columns[[0]], axis=1, inplace=True)
        n = df['n'].tolist()
        k = df['k'].tolist()
        material_params[i] = tuple(n + k)
    return material_params

material_params = load_material_data()
lambdas = np.linspace(7000, 14000, 71)
nMedium = 1.53

# Core evaluation function (used by all optimization methods)
def evaluate_design(c_m, s1_m, c_t, s1_t):
    c_index = material_params[c_m]
    s1_index = material_params[s1_m]

    core_r = c_t / 2
    total_r = core_r + s1_t

    g_array = []
    abs_ext_array = []

    i = 0
    j = 71
    for wavelength in lambdas:
        mCore = complex(c_index[i], c_index[j])
        mShell = complex(s1_index[i], s1_index[j])
        x = 2.0 * np.pi * np.array([core_r, total_r], dtype=np.float64) / wavelength
        m = np.array((mCore, mShell), dtype=np.complex128) / nMedium
        terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(np.array(x), np.array(m), mp=False)
        abs_ext_array.append(Qabs / Qext)
        g_array.append(g)

        i += 1
        j += 1

    g_averaged = mean(g_array)
    abs_ext = mean(abs_ext_array)
    result = 0.25 * (g_averaged + 1) + 0.5 * (abs_ext)

    print(f'Wavelength averaged asymmetry parameter is {g_averaged}')
    print(f'Absorption-to-extinction ratio is {abs_ext}')
    print(f'Result is {result} for {c_m}, {c_t}, {s1_m}, {s1_t}.')

    return result

# Objective function for Optuna
def objective(trial):
    c_m = trial.suggest_categorical('c_m', list(range(10)))
    s1_m = trial.suggest_categorical('s1_m', list(range(15)))
    c_t = trial.suggest_float('c_t', 500.0, 4000.0)
    s1_t = trial.suggest_float('s1_t', 5.0, 100.0)

    result = evaluate_design(c_m, s1_m, c_t, s1_t)

    result = torch.tensor([result])
    return result.item()

# PyMOO Problem definition for GA and PSO
class NanoparticleOptProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, 
                         n_obj=1, 
                         n_constr=0, 
                         xl=np.array([0, 0, 500.0, 5.0]), 
                         xu=np.array([9, 14, 4000.0, 100.0]),
                         vtype=np.array(['int', 'int', 'real', 'real']))
    
    def _evaluate(self, x, out, *args, **kwargs):
        f = np.zeros(len(x))
        for i, design in enumerate(x):
            c_m = int(design[0])
            s1_m = int(design[1])
            c_t = design[2]
            s1_t = design[3]
            f[i] = evaluate_design(c_m, s1_m, c_t, s1_t)
        out["F"] = f

# Samplers from Optuna
samplers = {
    "TPE": optuna.samplers.TPESampler(seed=42),
    "GP": optuna.samplers.GPSampler(n_startup_trials=10, seed=42)
}

# Run optimization with all methods
results = {}
wall_times = {}
max_evals = 200  # Limit for fair comparison

# Run Optuna optimization for each sampler
for sampler_name, sampler in samplers.items():
    study = optuna.create_study(direction='minimize', sampler=sampler)
    start_time = time.time()
    study.optimize(objective, n_trials=max_evals)
    end_time = time.time()
    results[sampler_name] = study.trials
    wall_times[sampler_name] = end_time - start_time

# Run GA and PSO optimizations using PyMOO
problem = NanoparticleOptProblem()

# GA optimization
start_time = time.time()
ga_algorithm = GA()

ga_callback = HistoryCallback()
ga_res = minimize(problem, 
                 ga_algorithm, 
                 termination=('n_eval', max_evals), 
                 seed=42, 
                 verbose=True,
                 callback=ga_callback)
end_time = time.time()

# Convert GA results to format comparable with Optuna
ga_trials = []
for i, (x, val) in enumerate(ga_callback.data):
    if i >= max_evals:
        break
    trial = optuna.trial.FrozenTrial(
        state=optuna.trial.TrialState.COMPLETE,
        value=val,
        params={'c_m': int(x[0]), 's1_m': int(x[1]), 'c_t': x[2], 's1_t': x[3]},
        datetime_start=None,
        datetime_complete=None,
        number=i,
        user_attrs={},
        system_attrs={},
        distributions=None,
        intermediate_values={},
        trial_id=10000+i
    )
    ga_trials.append(trial)

results["GA"] = ga_trials
wall_times["GA"] = end_time - start_time

# PSO optimization
start_time = time.time()
pso_algorithm = PSO() #Using the default one


pso_callback = HistoryCallback()
pso_res = minimize(problem, 
                  pso_algorithm, 
                  termination=('n_eval', max_evals), 
                  seed=42, 
                  verbose=True,
                  callback=pso_callback)
end_time = time.time()

# Convert PSO results to format comparable with Optuna
pso_trials = []
for i, (x, val) in enumerate(pso_callback.data):
    if i >= max_evals:
        break
    trial = optuna.trial.FrozenTrial(
        state=optuna.trial.TrialState.COMPLETE,
        value=val,
        params={'c_m': int(x[0]), 's1_m': int(x[1]), 'c_t': x[2], 's1_t': x[3]},
        datetime_start=None,
        datetime_complete=None,
        number=i,
        user_attrs={},
        system_attrs={},
        distributions=None,
        intermediate_values={},
        trial_id=20000+i
    )
    pso_trials.append(trial)

results["PSO"] = pso_trials
wall_times["PSO"] = end_time - start_time


for sampler_name, trials in results.items():
    if sampler_name in ["GA", "PSO"]:
        # For GA/PSO, use the callback's history
        callback = ga_callback if sampler_name == "GA" else pso_callback
        best_values = callback.best_so_far_history[:max_evals]
        trial_numbers = range(1, len(best_values) + 1)
    else:
        # For Optuna methods, use your existing code
        values = [trial.value for trial in trials if trial.value is not None]
        best_values = []
        current_best = float('inf')
        for val in values:
            if val is not None:
                current_best = min(current_best, val)
            best_values.append(current_best)
        trial_numbers = range(1, len(best_values) + 1)

    # Save best values to individual CSV files
    filename = f"best_values_vacuum_{sampler_name}.csv"
    pd.Series(best_values, name='best_value').to_csv(filename, index_label='iteration')
    
methods = list(wall_times.keys())
times = [wall_times[method] for method in methods]

# Calculate function evaluations per second
evals_per_second = {}
for method in methods:
    completed_trials = len([t for t in results[method] if t.value is not None])
    evals_per_second[method] = completed_trials / wall_times[method]

efficiency_values = [evals_per_second[method] for method in methods]

# Print best results and wall times for each method
print("\n=== OPTIMIZATION RESULTS ===")
for method_name, trials in results.items():
    best_idx = np.argmin([t.value for t in trials if t.value is not None])
    best_trial = trials[best_idx]
    
    print(f"\nBest results for {method_name}:")
    print(f"  Best parameters: {best_trial.params}")
    print(f"  Best value: {best_trial.value:.6f}")
    print(f"  Wall time: {wall_times[method_name]:.2f} seconds")
    print(f"  Evaluations per second: {evals_per_second[method_name]:.4f}")
    
