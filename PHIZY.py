import jax.numpy as jnp
import jax

class ChangeVar:
    def __init__(self, options):
        self.options = options

    def phi(self, z, y):
        raise NotImplementedError

    def phiinv(self, x, y):
        raise NotImplementedError

    def phiz(self, z, y):
        return jax.jacfwd(self.phi)(z, y)

    def phiy(self, z, y):
        return jax.jacfwd(self.phi, argnums=1)(z, y)

    def phizy(self, z, y):
        return jax.jacfwd(self.phiz, argnums=1)(z, y)


class PHIZY(ChangeVar):
    def __init__(self, options):
        super().__init__(options)
        self.x_copy = 0.
        self.y_copy = 0.
        self.x_max = 0.

    def updateConst(self, x, y):
        self.x_copy = jax.lax.stop_gradient(x)
        self.y_copy = jax.lax.stop_gradient(y)
        self.x_max = jnp.max(jnp.abs(self.x_copy)) * 1.01

    def Py_inv(self, y):
        if self.options['phi'] == "precond":
            hess = self.options['Hess'](self.x_copy, y)
            P = jnp.diag(1. / jnp.diag(hess))
            return P
        elif self.options['phi'] == "expz":
            return jnp.eye(y.shape[0])

    def Py(self, y):
        if self.options['phi'] == "precond":
            hess = self.options['Hess'](self.x_copy, y)
            P = jnp.diag(jnp.diag(hess))
            return P
        elif self.options['phi'] == "expz":
            return jnp.eye(y.shape[0])

    def Qz(self, z):
        if self.options['phi'] == "precond":
            return z
        elif self.options['phi'] == "expz":
            return jnp.sign(self.x_copy) * jnp.exp(z)

    def Qz_inv(self, x):
        if self.options['phi'] == "precond":
            return x
        elif self.options['phi'] == "expz":
            return jnp.log(jnp.abs(x))

    def phi(self, z, y):
        P = self.Py_inv(y)
        Q = self.Qz(z)
        return P @ Q

    def phiinv(self, x, y):
        P = self.Py(y)
        return self.Qz_inv(P @ x)

class OPTPHIZY(ChangeVar):
    def __init__(self, options, dataset, x_opt):
        super().__init__(options)
        self.x_copy = 0.
        self.y_copy = 0.
        self.Ab = dataset.feat.T @ dataset.label / dataset.feat.shape[0]
        self.const = x_opt

    def updateConst(self, x, y):
        self.x_copy = jax.lax.stop_gradient(x)
        self.y_copy = jax.lax.stop_gradient(y)
        self.const = self.x_copy

    def Py_inv(self, y):
        hess = self.options['Hess'](self.x_copy, y)
        P = jnp.linalg.inv(hess)
        return P

    def Py(self, y):
        hess = self.options['Hess'](self.x_copy, y)
        return hess

    def Qz(self, z):
        return self.options['Grad'](z, self.y_copy)

    def Qz_inv(self, x):
        hess = self.options['Hess'](x, self.y_copy)
        return jnp.linalg.inv(hess) @ (x + self.Ab)

    def phi(self, z, y):
        P = self.Py_inv(y)
        Q = self.Qz(z)
        return P @ Q + self.const

    def phiinv(self, x, y):
        x = x - self.const
        P = self.Py(y)
        return self.Qz_inv(P @ x)

