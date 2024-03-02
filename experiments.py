import os
import pickle
import argparse

from jax.config import config

from EnergyFunctions import *
from PHIZY import PHIZY, OPTPHIZY
from Jacobians import JacobiansX, JacobiansZ
from DataOpt import *
from util import *

config.update("jax_enable_x64", True)


def preDataOpts(args):
    dataset = Dataset(args)
    if args.inner_function == 'ridge_regression':
        inner_eng = RidgeRegression(dataset, args)
    elif args.inner_function == 'logistic_regression':
        inner_eng = LogisticRegression(dataset, args)
    if args.outer_function == 'linear':
        outer_eng = Linear(dataset, args)
    elif args.outer_function == 'quadratic':
        outer_eng = Quadratic(dataset, args)
    elif args.outer_function == 'logistic_regression':
        outer_eng = Logistic(dataset, args)
    x = dataset.x_init
    y = dataset.y_init

    return inner_eng, outer_eng, x, y


def solveProblem(inner_eng: EnergyFunctions, x, y, args):
    x_opt = inner_eng.computeOptX(x, y)
    # this is to make sure the initial x is far away from x_opt to get a good plot
    x = x + 10 ** (args.max_plot) * (x - x_opt)
    xs = inner_eng.GradDesc(x, y)
    return xs, x_opt


def changeVar(inner_eng: EnergyFunctions, x_opt):
    exp_options = {'phi': 'expz',
                   'Hess': inner_eng.Gradx}
    precond_options = {'phi': 'precond',
                       'Hess': inner_eng.Gradx}
    opt_options = {'Grad': inner_eng.Grad,
                   'Hess': inner_eng.Gradx}
    exp_phi = PHIZY(exp_options)
    Pz_phi = PHIZY(precond_options)
    opt_phi = OPTPHIZY(opt_options, inner_eng.dataset, x_opt)
    return exp_phi, Pz_phi, opt_phi


def compareResults(x_opt, xs, y, inner_eng, outer_eng, exp_phi, Pz_phi, opt_phi, jacx, jacz, args):
    results = computeDistVSHyperGrad(x_opt, xs, y, inner_eng, outer_eng, exp_phi, Pz_phi, opt_phi,
                                    jacx, jacz, args)
    return results


def runExps(args):
    inner_eng, outer_eng, x, y = preDataOpts(args)
    xs, x_opt = solveProblem(inner_eng, x, y, args)
    exp_phi, Pz_phi, opt_phi = changeVar(inner_eng, x_opt)
    jacx = JacobiansX()
    jacz = JacobiansZ()
    results = compareResults(x_opt, xs, y, inner_eng, outer_eng, exp_phi, Pz_phi, opt_phi, jacx, jacz,
                             args)
    return results

def paserConfig():
    parser = argparse.ArgumentParser()

    parser.add_argument('--figure', type=str, default='3b', help='Reproduce the figure in the paper')
    parser.add_argument('--num_exp', type=int, default=2)
    parser.add_argument('--inner_function', type=str, default='ridge_regression', 
                        choices=['ridge_regression', 'logistic_regression'], help='Inner function of bilevel optimization')
    parser.add_argument('--outer_function', type=str, default='linear',
                         choices=['linear', 'quadratic', 'logistic'], help='Outer function of bilevel optimization')
    parser.add_argument('--dataset', type=str, default='mpg_scale', help='Dataset to use')
    parser.add_argument('--max_iter', type=int, default=150, help='Maximum number of iterations for inner solver')
    parser.add_argument('--y_scale', type=str, default='3_6', help='Scale of regularization parameter y')
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--min_plot', type=float, default=-3.5)
    parser.add_argument('--max_plot', type=float, default=1.5)
    parser.add_argument('--num_plot', type=int, default=20)
    parser.add_argument('--demo', type=bool, default=True, help='Run the demo')
    args = parser.parse_args()
    if args.demo == False:
        args.num_plot = 150
        args.max_iter = 50000
        args.num_exp = 10
    if args.figure == '1a':
        args.methods = ['Newton', 'OptPhi']
        args.y_scale = '3_6'
        args.inner_function='ridge_regression'
        args.outer_function='linear'
    elif args.figure == '1b':
        args.methods = ['Newton', 'OptPhi']
        args.y_scale = '3_6'
        args.inner_function='ridge_regression'
        args.outer_function='quadratic'
    elif args.figure == '2b':
        args.methods = ['Precond', 'ExpPhi', 'PzPhi', 'Vanilla']
        args.y_scale = '3_6'
        args.inner_function='ridge_regression'
        args.outer_function='quadratic'
    elif args.figure == '3a':
        args.methods = ['Precond', 'ExpPhi', 'PzPhi', 'Vanilla', 'Newton']
        args.y_scale = '-1_1'
        args.inner_function='logistic_regression'
        args.outer_function='logistic_regression'
        args.dataset='liver_scale'
    elif args.figure == '3b':
        args.methods = ['Precond', 'ExpPhi', 'PzPhi', 'Vanilla', 'Newton']
        args.y_scale = '3_6'
        args.inner_function='logistic_regression'
        args.outer_function='logistic_regression'
        args.dataset='liver_scale'
    return args

if __name__ == "__main__":
    args = paserConfig()
    results = []
    for i in range(args.num_exp):
        np.random.seed(i)
        result = runExps(args)
        results.append(result)
        
    plotResults(results, args)
