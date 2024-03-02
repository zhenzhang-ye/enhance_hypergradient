import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def computeDistVSHyperGrad(x_opt, xs, y, inner_eng, outer_eng, exp_phi, Pz_phi, opt_phi, jacx, jacz, args):
    xdists = []
    basicdists = []
    newtondists = []
    preconddists = []
    Pzdists = []
    expdists = []
    optdists = []
    hyper_opt = jacx.HyperGrad(x_opt, y, inner_eng, outer_eng)
    if args.inner_function == "ridge_regression":
        hess = inner_eng.Gradx(x_opt, y)
        hess_inv = jnp.linalg.inv(inner_eng.Gradx(x_opt, y))
        P_inv = jnp.diag(1. / jnp.diag(hess))
        hess_inv_prev = hess_inv
        P_inv_prev = P_inv

    plotxs = np.logspace(args.max_plot, args.min_plot, args.num_plot)
    plotind = 0
    hyper = jacx.HyperGrad(xs[0], y, inner_eng, outer_eng)
    for i in range(1, len(xs)):
        x = xs[i]
        x_prev = xs[i-1]
        dist_cur = jnp.linalg.norm(x_opt - x)
        dist_prev = jnp.linalg.norm(x_opt - x_prev)
        xdists.append(jnp.linalg.norm(x - x_opt))
        if dist_prev >plotxs[plotind] and dist_cur < plotxs[plotind]:
            while (dist_cur < plotxs[plotind]):
                alpha = (plotxs[plotind] - dist_cur) / (dist_prev - dist_cur)
                plotind += 1
                if plotind == args.num_plot:
                    break
                hyper = jacx.HyperGrad(x, y, inner_eng, outer_eng)
                hyper_prev = jacx.HyperGrad(x_prev, y, inner_eng, outer_eng)
                hyper = (1 - alpha) * hyper + alpha * hyper_prev

            if 'Vanilla' in args.methods:
                basicdists.append(jnp.linalg.norm(hyper_opt - hyper))

            # newton
            if args.inner_function == "logistic_regression":
                hess = inner_eng.Gradx(x, y)
                hess_inv = jnp.linalg.inv(hess)
                P_inv = jnp.diag(1. / jnp.diag(hess))
                hess_prev = inner_eng.Gradx(x_prev, y)
                hess_inv_prev = jnp.linalg.inv(hess_prev)
                P_inv_prev = jnp.diag(1. / jnp.diag(hess_prev))

            if 'Newton' in args.methods:
                x_hat = x - hess_inv @ inner_eng.Grad(x, y)
                hyper = jacx.HyperGrad(x_hat, y, inner_eng, outer_eng)
                x_hat_prev = x_prev - hess_inv_prev @ inner_eng.Grad(x_prev, y)
                hyper_prev = jacx.HyperGrad(x_hat_prev, y, inner_eng, outer_eng)
                hyper = (1 - alpha) * hyper + alpha * hyper_prev
                newtondists.append(jnp.linalg.norm(hyper_opt - hyper))

            # precond
            if 'Precond' in args.methods:
                x_hat = x - P_inv @ inner_eng.Grad(x, y)
                hyper = jacx.HyperGrad(x_hat, y, inner_eng, outer_eng)
                x_hat_prev = x_prev - P_inv_prev @ inner_eng.Grad(x_prev, y)
                hyper_prev = jacx.HyperGrad(x_hat_prev, y, inner_eng, outer_eng)
                hyper = (1 - alpha) * hyper + alpha * hyper_prev
                preconddists.append(jnp.linalg.norm(hyper_opt - hyper))

            # Pz
            if 'PzPhi' in args.methods:
                Pz_phi.updateConst(x, y)
                hyper = jacz.HyperGrad(x, y, inner_eng, outer_eng, Pz_phi)
                Pz_phi.updateConst(x_prev, y)
                hyper_prev = jacz.HyperGrad(x_prev, y, inner_eng, outer_eng, Pz_phi)
                hyper = (1 - alpha) * hyper + alpha * hyper_prev
                Pzdists.append(jnp.linalg.norm(hyper_opt - hyper))

            # exp
            if 'ExpPhi' in args.methods:
                exp_phi.updateConst(x, y)
                hyper = jacz.HyperGrad(x, y, inner_eng, outer_eng, exp_phi)
                exp_phi.updateConst(x_prev, y)
                hyper_prev = jacz.HyperGrad(x_prev, y, inner_eng, outer_eng, exp_phi)
                hyper = (1 - alpha) * hyper + alpha * hyper_prev
                expdists.append(jnp.linalg.norm(hyper_opt - hyper))

            # ideal
            if args.inner_function ==  'ridge_regression' and 'OptPhi' in args.methods:
                opt_phi.updateConst(x, y)
                hyper = jacz.HyperGrad(x, y, inner_eng, outer_eng, opt_phi)
                opt_phi.updateConst(x_prev, y)
                hyper_prev = jacz.HyperGrad(x_prev, y, inner_eng, outer_eng, opt_phi)
                hyper = (1 - alpha) * hyper + alpha * hyper_prev
                optdists.append(jnp.linalg.norm(hyper_opt - hyper))
        else:
            continue
        if plotind == args.num_plot:
            break

    results = {}
    results['plotxs'] = plotxs
    results['xdists'] = np.array(xdists)
    if 'Vanilla' in args.methods:
        results['Vanilla'] = np.array(basicdists)
    if 'Newton' in args.methods:
        results['Newton'] = np.array(newtondists)
    if 'Precond' in args.methods:
        results['Precond'] = np.array(preconddists)
    if 'PzPhi' in args.methods:
        results['PzPhi'] = np.array(Pzdists)
    if 'ExpPhi' in args.methods:
        results['ExpPhi'] = np.array(expdists)
    if 'OptPhi' in args.methods:
        results['OptPhi'] = np.array(optdists)
    return results

def plotResults(results, args):
    def getcolors():
        colors = {'Vanilla': [1., 95. / 255., 1.],
                'Newton': [208. / 255., 40. / 255., 28. / 255.],
                'Precond': [80. / 255., 180. / 255., 79. / 255.],
                'PzPhi': [1., 156. / 255., 70. / 255.],
                'ExpPhi': [42. / 255., 43. / 255., 192. / 255.],
                'OptPhi': [66. / 255., 156. / 255., 185. / 255.]}
        return colors
    
    def computeMeanVar(results, method):
        stack_results = np.zeros((len(results), args.num_plot))
        min_len = args.num_plot
        for i, result in enumerate(results):
            cur_len = len(result[method])
            stack_results[i, :cur_len] = result[method]
            if cur_len < min_len:
                min_len = cur_len
        stack_results = stack_results[:, :min_len]
        mean_result = np.mean(stack_results, axis=0)
        var_result = np.var(stack_results, axis=0)
        return mean_result, var_result
    
    colors = getcolors()
    for key in results[0].keys():
        if key == 'xdists' or key == 'plotxs':
            continue
        mean_result, var_result = computeMeanVar(results, key)
        xs = results[0]['plotxs'][:mean_result.shape[0]]
        if key == "PzPhi":
            plt.plot(xs, mean_result, label=key, color=colors[key], linestyle='dashed',
                    linewidth=2.0)
        else:
            plt.plot(xs, mean_result, label=key, color=colors[key],linewidth=2.0)
        plt.fill_between(xs, mean_result - np.sqrt(var_result), mean_result + np.sqrt(var_result),
                         alpha=0.1, color=colors[key])
    plt.yscale('log')
    plt.xscale('log')
    x_axis = plt.xlabel(r'$\|x - x^*(y)\|$')
    y_axis = plt.ylabel('Hypergradient Error')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()