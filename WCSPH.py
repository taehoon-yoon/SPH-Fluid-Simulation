import taichi as ti
import sph_base


class WCSPHSolver(sph_base.SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        self.gamma = self.ps.config['gamma']
        self.B = self.ps.config['B']
        self.surface_tension = ti.field(ti.f32, shape=())
        self.surface_tension[None] = self.ps.config['surfaceTension']

    @ti.func
    def update_density_task(self, p_i, p_j, density: ti.template()):
        """
        Versatile Rigid-Fluid Coupling for Incompressible SPH  equation (6)
        https://cg.informatik.uni-freiburg.de/publications/2012_SIGGRAPH_rigidFluidCoupling.pdf
        """
        if self.ps.material[p_j] == self.ps.material_fluid:
            density += self.ps.mass[p_i] * self.cubic_spline_kernel(
                (self.ps.position[p_i] - self.ps.position[p_j]).norm())  # First term in RHS of equation (6)
        elif self.ps.material[p_j] == self.ps.material_rigid:
            density += self.ps.density0 * self.ps.volume[p_j] * self.cubic_spline_kernel(
                (self.ps.position[p_i] - self.ps.position[p_j]).norm())  # Second term in RHS of equation (6)
            # With self.ps.density0 * self.ps.volume[p_j] equal to eq (5)

    @ti.kernel
    def update_density(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.material[i] == self.ps.material_fluid:
                density = self.ps.mass[i] * self.cubic_spline_kernel(0.0)
                self.ps.for_all_neighbors(i, self.update_density_task, density)
                self.ps.density[i] = density

    @ti.kernel
    def update_pressure(self):
        """
        Weakly compressible SPH for free surface flows   equation (7)  (Tait's equation)
        https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
        """
        for i in range(self.ps.total_particle_num):
            if self.ps.material[i] == self.ps.material_fluid:
                self.ps.density[i] = ti.max(self.ps.density[i], self.ps.density0)
                # TODO: check self.ps.density[i] = ti.max(self.ps.density[i], self.density0)
                self.ps.pressure[i] = self.B * ((self.ps.density[i] / self.ps.density0) ** self.gamma - 1)  # eq (7)

    @ti.func
    def compute_pressure_force_task(self, p_i, p_j, acc: ti.template()):
        """
        Weakly compressible SPH for free surface flows   equation (6)  (Momentum equation)
        https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
        """
        """
        Versatile Rigid-Fluid Coupling for Incompressible SPH  equation (9)
        https://cg.informatik.uni-freiburg.de/publications/2012_SIGGRAPH_rigidFluidCoupling.pdf
        """
        gradW = self.cubic_spline_kernel_derivative((self.ps.position[p_i] - self.ps.position[p_j]))
        p_rho_i = self.ps.pressure[p_i] / (self.ps.density[p_i] ** 2)
        if self.ps.material[p_j] == self.ps.material_fluid:
            m_j = self.ps.mass[p_j]
            p_rho_j = self.ps.pressure[p_j] / (self.ps.density[p_j] ** 2)
            acc -= m_j * (p_rho_i + p_rho_j) * gradW  # WCSPH equation (6)
            # Gravity term in eq (6) is omitted since gravity is handled in compute_non_pressure_force
        else:
            psi = self.ps.density0 * self.ps.volume[p_j]
            acc_tmp = -psi * p_rho_i * gradW  # Versatile Rigid... equation (9)
            acc += acc_tmp
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] -= acc_tmp * self.ps.mass[p_i] / self.ps.mass[p_j]  # TODO: Is this correct?

    @ti.kernel
    def compute_pressure_force(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.is_static_rigid_body(i):
                self.ps.acceleration[i].fill(0.0)
            elif self.ps.material[i] == self.ps.material_fluid:
                acc = ti.Vector.zero(ti.f32, self.ps.dim)
                self.ps.for_all_neighbors(i, self.compute_pressure_force_task, acc)
                self.ps.acceleration[i] += acc

    @ti.func
    def compute_non_pressure_force_task(self, p_i, p_j, acc: ti.template()):
        """
        Weakly compressible SPH for free surface flows   equation (16)  (Surface Tension)
        https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
        """
        """
        Versatile Rigid-Fluid Coupling for Incompressible SPH  equation (11)~(14)
        https://cg.informatik.uni-freiburg.de/publications/2012_SIGGRAPH_rigidFluidCoupling.pdf
        """

        # Surface Tension
        # WCSPH equation (16).
        # One thing to notice is that there should be extra term (x_a-x_b) in (16) check memo in the paper.
        if self.ps.material[p_j] == self.ps.material_fluid:
            r_vec = self.ps.position[p_i] - self.ps.position[p_j]
            # if r_vec.norm() > self.ps.particle_diameter:
            acc -= self.surface_tension[None] / self.ps.mass[p_i] * self.ps.mass[p_j] * r_vec * \
                   self.cubic_spline_kernel(r_vec.norm())
            """
            else:
                acc -= self.surface_tension / self.ps.mass[p_i] * self.ps.mass[p_j] * r_vec * self.cubic_spline_kernel(
                    self.ps.particle_diameter)
            """

        # Viscosity Force
        # Versatile Rigid-Fluid Coupling for Incompressible SPH  equation (11)~(14)
        if self.ps.material[p_j] == self.ps.material_fluid:
            nu = 2 * self.viscosity[None] * self.ps.support_length * self.c_s / (
                    self.ps.density[p_i] + self.ps.density[p_j])
            v_ij = self.ps.velocity[p_i] - self.ps.velocity[p_j]
            x_ij = self.ps.position[p_i] - self.ps.position[p_j]
            pi = -nu * ti.min(v_ij.dot(x_ij), 0.0) / (x_ij.dot(x_ij) + 0.01 * self.ps.support_length ** 2)  # eq (11)

            acc -= self.ps.mass[p_j] * pi * self.cubic_spline_kernel_derivative(x_ij)  # eq (12)

        else:
            sigma = self.ps.rigid_bodies_sigma[self.ps.object_id[p_j]]
            nu = sigma * self.ps.support_length * self.c_s / (2 * self.ps.density[p_i])  # eq (14)
            v_ij = self.ps.velocity[p_i] - self.ps.velocity[p_j]
            x_ij = self.ps.position[p_i] - self.ps.position[p_j]
            pi = -nu * ti.min(v_ij.dot(x_ij), 0.0) / (x_ij.dot(x_ij) + 0.01 * self.ps.support_length ** 2)  # eq (11)

            acc -= self.ps.density0 * self.ps.volume[p_j] * pi * self.cubic_spline_kernel_derivative(x_ij)  # eq (13)

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
                self.ps.velocity[i] += self.ps.acceleration[i] * self.dt[None]
                self.ps.position[i] += self.ps.velocity[i] * self.dt[None]

    def substep(self):
        self.update_density()
        self.update_pressure()
        self.compute_non_pressure_force()
        self.compute_pressure_force()
        self.advect()
