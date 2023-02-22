import taichi as ti
import numpy as np
import sph_base


class WCSPHSolver(sph_base.SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.gamma = self.ps.config['gamma']
        self.B = self.ps.config['B']
        self.surface_tension = 0.01

    @ti.func
    def update_density_task(self, p_i, p_j, density: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            density += self.ps.mass[p_i] * self.cubic_spline_kernel(
                (self.ps.position[p_i] - self.ps.position[p_j]).norm())
        elif self.ps.material[p_j] == self.ps.material_rigid:
            density += self.ps.density0 * self.ps.volume[p_j] * self.cubic_spline_kernel(
                (self.ps.position[p_i] - self.ps.position[p_j]).norm())

    @ti.kernel
    def update_density(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.material[i] == self.ps.material_fluid:
                density = self.ps.mass[i] * self.cubic_spline_kernel(0.0)
                self.ps.for_all_neighbors(i, self.update_density_task, density)
                self.ps.density[i] = density

    @ti.kernel
    def update_pressure(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.material[i] == self.ps.material_fluid:
                # TODO: check self.ps.density[i] = ti.max(self.ps.density[i], self.density0)
                self.ps.pressure[i] = self.B * ((self.ps.density[i] / self.ps.density0) ** self.gamma - 1)

    @ti.func
    def compute_pressure_force_task(self, p_i, p_j, acc: ti.template()):
        gradW = self.cubic_spline_kernel_derivative((self.ps.position[p_i] - self.ps.position[p_j]))
        p_rho_i = self.ps.pressure[p_i] / (self.ps.density[p_i] ** 2)
        if self.ps.material[p_j] == self.ps.material_fluid:
            m_j = self.ps.mass[p_j]
            p_rho_j = self.ps.pressure[p_j] / (self.ps.density[p_j] ** 2)
            acc -= m_j * (p_rho_i + p_rho_j) * gradW
        else:
            psi = self.ps.density0 * self.ps.volume[p_j]
            p_rho_j = self.ps.pressure[p_i] / (self.ps.density0 ** 2)
            acc_tmp = -psi * (p_rho_i+p_rho_j) * gradW  # TODO: Is this correct?
            acc += acc_tmp
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] -= acc_tmp * self.ps.mass[p_i] / self.ps.mass[p_j]  # TODO: Is this correct?

    @ti.kernel
    def compute_pressure_force(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.is_static_rigid_body(i):
                self.ps.acceleration[i].fill(0.0)
            elif self.ps.material[i] == self.ps.material_fluid:
                acc = ti.Vector.zero(ti.f32,self.ps.dim)
                self.ps.for_all_neighbors(i, self.compute_pressure_force_task, acc)
                self.ps.acceleration[i] += acc

    @ti.func
    def compute_non_pressure_force_task(self, p_i, p_j, acc: ti.template()):
        # Surface Tension
        if self.ps.material[p_j]==self.ps.material_fluid:
            r_vec=self.ps.position[p_i]-self.ps.position[p_j]
            if r_vec.norm()>self.ps.particle_diameter:
                acc-=self.surface_tension/self.ps.mass[p_i]*self.ps.mass[p_j]*r_vec*self.cubic_spline_kernel(r_vec.norm())
            else:
                acc-=self.surface_tension/self.ps.mass[p_i]*self.ps.mass[p_j]*r_vec*self.cubic_spline_kernel(self.ps.particle_diameter)

        # Viscosity Force
        d = 2 * (self.ps.dim + 2)
        r=self.ps.position[p_i]-self.ps.position[p_j]
        v_xy=(self.ps.velocity[p_i]-self.ps.velocity[p_j]).dot(r)
        if self.ps.material[p_j]==self.ps.material_fluid:
            f_v=d*self.viscosity*(self.ps.mass[p_j]/(self.ps.density[p_j]))*v_xy/(r.norm()**2+0.01*self.ps.support_length**2)*self.cubic_spline_kernel_derivative(r)
            #acc+=f_v


    @ti.kernel
    def compute_non_pressure_force(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.is_static_rigid_body(i):
                self.ps.acceleration[i].fill(0.0)

            else:
                acc = ti.Vector(self.g)
                self.ps.acceleration[i] = acc
                if self.ps.material[i] == self.ps.material_fluid:
                    self.ps.for_all_neighbors(i, self.compute_non_pressure_force_task, acc)
                    self.ps.acceleration[i] = acc

    @ti.kernel
    def advect(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.is_dynamic[i]:
                self.ps.velocity[i]+=self.ps.acceleration[i]*self.dt[None]
                self.ps.position[i]+=self.ps.velocity[i]*self.dt[None]

    def substep(self):
        self.update_density()
        self.update_pressure()
        self.compute_non_pressure_force()
        self.compute_pressure_force()
        self.advect()