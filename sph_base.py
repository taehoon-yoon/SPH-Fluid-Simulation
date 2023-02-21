import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = self.ps.config['gravitation']
        self.dt = ti.field(ti.f32, shape=())
        self.dt[None] = self.ps.config['dt']

    @ti.func
    def cubic_spline_kernel(self, r_norm):
        h = self.ps.support_length  # 4r
        coeff = 8 / np.pi if self.ps.dim == 3 else 40 / 7 / np.pi
        coeff /= (h ** self.ps.dim)
        q = r_norm / h
        kernel_val = 0.0
        if q <= 1.0:
            if q <= 0.5:
                kernel_val = coeff * (1 - 6 * (q ** 2) + 6 * (q ** 3))
            else:
                kernel_val = coeff * (2 * (1 - q) ** 3)
        return kernel_val

    @ti.func
    def cubic_spline_kernel_derivative(self, r):
        h = self.ps.support_length
        coeff = 16 / np.pi if self.ps.dim == 3 else 80 / 7 / np.pi
        coeff /= (h ** (self.ps.dim + 1))
        derivative = ti.Vector.zero(ti.f32, self.ps.dim)
        r_norm = r.norm()
        r_hat = r / (r_norm + 1e-6)
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                derivative = coeff * (9 * q ** 2 - 6 * q) * r_hat
            else:
                derivative = coeff * (-3 * (1 - q) ** 2) * r_hat
        return derivative
