import jax
import jax.numpy as jnp
from EnergyFunctions import EnergyFunctions
from PHIZY import ChangeVar


class Jacobians:
    def __init__(self):
        return

    def JacXy(self):
        raise NotImplementedError

    def DJacXy(self):
        raise NotImplementedError


class JacobiansX(Jacobians):
    def __init__(self, hess=None):
        super().__init__()
        if hess is not None:
            self.Fx_inv = jnp.linalg.inv(hess)
        else:
            self.Fx_inv = None
    def JacXy(self, x, y, func_eng: EnergyFunctions):
        if self.Fx_inv == None:
            Fx = func_eng.Gradx(x, y)
            Fy = func_eng.Grady(x, y)
            return -Fy.T @ jnp.linalg.inv(Fx)
        else:
            Fy = func_eng.Grady(x, y)
            return -Fy.T @ self.Fx_inv

    def DJacXy(self, x, y, func_eng: EnergyFunctions):
        return jax.jacfwd(self.JacXy)(x, y, func_eng)

    def HyperGrad(self, x, y, inner_eng: EnergyFunctions, outer_eng: EnergyFunctions):
        return self.JacXy(x, y, inner_eng) @ outer_eng.Grad(x, y) + outer_eng.GradOny(x, y)

    def DHyperGrad(self, x, y, inner_eng, outer_eng):
        return jax.jacfwd(self.HyperGrad)(x, y, inner_eng, outer_eng)

    def HyperGradConst(self, x, y, inner_eng, outer_eng, a):
        return self.HyperGrad(x, y, inner_eng, outer_eng) / a

    def DHyperGradConst(self, x, y, inner_eng, outer_eng, a):
        return jax.jacfwd(self.HyperGradConst)(x, y, inner_eng, outer_eng, a)

class JacobiansZ(Jacobians):
    def EnergyZ(self, z, y, func_eng: EnergyFunctions, func_var: ChangeVar):
        x = func_var.phi(z, y)
        return func_eng.Eng(x, y)

    def F(self, z, y, func_eng: EnergyFunctions, func_var: ChangeVar):
        return jax.jacfwd(self.EnergyZ)(z, y, func_eng, func_var)

    def Fz(self, z, y, func_eng: EnergyFunctions, func_var: ChangeVar):
        return jax.jacfwd(self.F)(z, y, func_eng, func_var)

    def Fy(self, z, y, func_eng: EnergyFunctions, func_var: ChangeVar):
        return jax.jacfwd(self.F, argnums=1)(z, y, func_eng, func_var)

    def JacZy(self, z, y, func_eng: EnergyFunctions, func_var: ChangeVar):
        Fz = self.Fz(z, y, func_eng, func_var)
        Fy = self.Fy(z, y, func_eng, func_var)
        return -Fy.T @ jnp.linalg.inv(Fz)

    def JacZyToXy(self, z, y, Jz, func_var: ChangeVar):
        return Jz @ func_var.phiz(z, y).T + func_var.phiy(z, y).T

    def JacXy(self, x, y, func_eng: EnergyFunctions, func_var: ChangeVar):
        z = func_var.phiinv(x, y)
        Jz = self.JacZy(z, y, func_eng, func_var)
        return self.JacZyToXy(z, y, Jz, func_var)

    def DJacXy(self, x, y, func_eng, func_var):
        return jax.jacfwd(self.JacXy)(x, y, func_eng, func_var)

    def HyperGrad(self, x, y, inner_eng: EnergyFunctions, outer_eng: EnergyFunctions, func_var: ChangeVar):
        return self.JacXy(x, y, inner_eng, func_var) @ outer_eng.Grad(x, y) + outer_eng.GradOny(x, y)

    def DHyperGrad(self, x, y, inner_eng, outer_eng, func_var):
        return jax.jacfwd(self.HyperGrad)(x, y, inner_eng, outer_eng, func_var)