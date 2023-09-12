import threading

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from prune import ModelPruner


class PruningProblem(ElementwiseProblem):

    def __init__(self, mp: ModelPruner):
        super().__init__(n_var=mp.prunable_layer_num, n_obj=1,
                         xl=0., xu=1.0)
        self._mp = mp

    def _evaluate(self, x, out, *args, **kwargs):
        pruned_model = self._mp.prune_model(x)
        f, _ = self._mp.get_fitness_score(pruned_model)
        out["F"] = f


if __name__ == '__main__':

    # TODO: arg parse
    model_name = 'resnet18'
    dataset = 'MNIST'
    pruning_method = 'by_parameter'
    es_n_iter = 10
    device = 'cpu'


    mp = ModelPruner(model_name, dataset, pruning_method)
    print('baseline:', mp.baseline)

    problem = PruningProblem(mp)
    algorithm = CMAES(x0=np.random.random(problem.n_var))
    res = minimize(problem,
                   algorithm,                
                   ('n_iter', es_n_iter),
                   seed=1,
                   verbose=True)

    print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
    print(mp.get_fitness_score(mp.cached_model, verbose=True))

    pruned_model_saving_path = f"{mp.config['model_weights_root_path']}pruned/pruned_{model_name}_{dataset}.model"
    torch.save(mp.cached_model, pruned_model_saving_path)
    
    # Load pruned model
    # pruned_model_saving_path = f"{mp.config['model_weights_root_path']}pruned/pruned_{model_name}_{dataset}.model"
    # model = torch.load(pruned_model_saving_path, map_location=torch.device(device))
    # model = model.to(device)
    # print(mp.get_fitness_score(model))