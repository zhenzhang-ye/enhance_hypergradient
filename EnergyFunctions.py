import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize


class EnergyFunctions:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.step_size = 1.0
        self.x_opt = None

    def computeOptX(self, x, y):
        xs = self.BFGS(x, y)
        self.x_opt = xs[-1]
        return self.x_opt

    def Eng(self):
        raise NotImplementedError

    def Grad(self, x, y):
        return jax.jacfwd(self.Eng)(x, y)

    def GradOny(self, x, y):
        return jax.jacfwd(self.Eng, argnums=1)(x, y)

    def Gradx(self, x, y):
        return jax.jacfwd(self.Grad)(x, y)

    def Gradxx(self, x, y):
        return jax.jacfwd(self.Gradx)(x, y)

    def Grady(self, x, y):
        return jax.jacfwd(self.Grad, argnums=1)(x, y)

    def Gradxy(self, x, y):
        return jax.jacfwd(self.Gradx, argnums=1)(x, y)

    def GradDesc(self, x, y, x_stop=1.0):
        xs = np.zeros((self.args.max_iter + 1, *x.shape)) #recording
        xs[0] = x
        for i in range(self.args.max_iter):
            if self.args.inner_function == 'logistic_regression':
                x_prev = x
                x -= self.step_size * self.Grad(x, y)
                # Backtracking line search
                while (self.Eng(x, y) > self.Eng(x_prev, y)):
                    self.step_size = self.step_size * 0.5
                    x = x_prev - self.step_size * self.Grad(x_prev, y)
            else:
                x -= self.step_size * self.Grad(x, y)
            xs[i + 1] = x.reshape(-1)
        return xs

    def BFGS(self, x, y):
        result = minimize(self.Eng, x, (y,),
                          method='BFGS',
                          options={'gtol': 1e-13, 'norm': 2})
        x_opt = result.x
        return [x_opt]

    def GenerateA(self):
        raise NotImplementedError

    def Generateb(self):
        raise NotImplementedError

    def setHess(self, hess_inv):
        self.hess_inv = hess_inv


class LogisticRegression(EnergyFunctions):
    def Eng(self, x, y):
        return jnp.log(1 + jnp.exp(-self.dataset.label * (self.dataset.feat @ x))).mean() + (
                0.5 * jnp.mean(jnp.exp(y) * x * x))


class RidgeRegression(EnergyFunctions):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.computeStepsize(dataset.y_init)

    def Eng(self, x, y):
        return 0.5 * jnp.linalg.norm(self.dataset.feat @ x - self.dataset.label) ** 2 / self.dataset.feat.shape[0] + (
                0.5 * jnp.sum(jnp.exp(y) * x * x)) / x.shape[0]

    def computeOptX(self, x, y):
        lhs = self.dataset.feat.T @ self.dataset.feat / self.dataset.feat.shape[0]
        lhs += jnp.diag(jnp.exp(y)) / x.shape[0]
        rhs = self.dataset.feat.T @ self.dataset.label / self.dataset.feat.shape[0]
        self.x_opt = jnp.linalg.solve(lhs, rhs)
        return self.x_opt

    def computeStepsize(self, y):
        Hess = self.dataset.feat.T @ self.dataset.feat / self.dataset.feat.shape[0] \
                + jnp.diag(jnp.exp(y)) / y.shape[0]
        self.step_size = 1.8 / jnp.linalg.eigh(Hess)[0].max()


class Linear(EnergyFunctions):
    def Eng(self, x, y):
        return self.dataset.b_out @ x + jnp.linalg.norm(y)**2


class Quadratic(EnergyFunctions):
    def Eng(self, x, y):
        return 0.5 * jnp.linalg.norm(self.dataset.A_out @ x - self.dataset.b_out) ** 2

class Logistic(EnergyFunctions):
    def Eng(self, x, y):
        return jnp.log(1 + jnp.exp(-self.dataset.label_t * (self.dataset.feat_t @ x))).mean()