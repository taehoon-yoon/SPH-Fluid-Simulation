import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = np.array(self.ps.config['gravitation'])
        self.dt = ti.field(ti.f32, shape=())
        self.dt[None] = self.ps.config['dt']
        self.collision_factor = self.ps.config['collisionFactor']
        self.viscosity = ti.field(ti.f32, shape=())
        self.viscosity[None] = self.ps.config['viscosity']
        self.c_s = self.ps.config['c_s']  # speed of the numerical propagation
        # [Versatile Rigid-Fluid Coupling for Incompressible SPH], between (11) and (12)

    @ti.func
    def cubic_spline_kernel(self, r_norm):
        """
        Smoothed Particle Hydrodynamics     eq (3.31)
        https://arxiv.org/abs/1007.1245v2
        """
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
        """
        Smoothed Particle Hydrodynamics     eq (3.33)
        https://arxiv.org/abs/1007.1245v2
        """
        h = self.ps.support_length
        coeff = 16 / np.pi if self.ps.dim == 3 else 80 / 7 / np.pi
        coeff /= (h ** (self.ps.dim + 1))
        derivative = ti.Vector([0.0 for _ in range(self.ps.dim)])
        r_norm = r.norm()
        q = r_norm / h
        r_hat = ti.select(r_norm > 1e-7, r / r_norm, r / (r_norm + 1e-7))
        if q <= 1.0:  # TODO: if q <= 1.0 and r_norm > 1e-7: -> taichi lang error. But can't figure out why it is error.
            if q <= 0.5:
                derivative = coeff * (9 * q ** 2 - 6 * q) * r_hat
            else:
                derivative = coeff * (-3 * (1 - q) ** 2) * r_hat
        return derivative

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta_bi):
        if self.ps.material[p_j] == self.ps.material_rigid:
            delta_bi += self.cubic_spline_kernel((self.ps.position[p_i] - self.ps.position[p_j]).norm())

    @ti.kernel
    def compute_volume_of_boundary_particle(self):
        """
        Versatile Rigid-Fluid Coupling for Incompressible SPH     eq (4)
        https://cg.informatik.uni-freiburg.de/publications/2012_SIGGRAPH_rigidFluidCoupling.pdf
        """
        """
        You can find the reason for using this kind of volume for boundary particle from the following paper.
        Density Contrast SPH Interfaces
        https://people.inf.ethz.ch/~sobarbar/papers/Sol08b/Sol08b.pdf
        """
        for i in range(self.ps.total_particle_num):
            if self.ps.is_static_rigid_body(i):
                delta_bi = self.cubic_spline_kernel(0.0)
                self.ps.for_all_neighbors(i, self.compute_boundary_volume_task, delta_bi)
                self.ps.volume[i] = 1.0 / delta_bi  # TODO: check 1.0 / delta_bi * 3.0
                """
                I think the extra 3.0 factor is due to handle single layer of boundary particles?
                Since we are using full rather than hollow, to represent voxelized mesh, we don't have to use
                extra 3.0 factor? 'voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()' 
                
                'Therefore, using a single layer
                of boundary particles with (6) and taking the missing particles into
                account in (5) is a decent approximation in practice.' 
                -> Versatile Rigid-Fluid Coupling for Incompressible SPH [page 3]
                """

    @ti.func
    def simulate_collision(self, idx, vec):
        self.ps.velocity[idx] -= (1.0 + self.collision_factor) * self.ps.velocity[idx].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_3D(self):
        for i in range(self.ps.total_particle_num):
            if self.ps.is_dynamic[i]:
                pos = self.ps.position[i]
                collision_vec = ti.Vector.zero(ti.f32, self.ps.dim)
                for dim in ti.static(range(self.ps.dim)):
                    if pos[dim] > self.ps.domain_end[dim] - self.ps.padding:
                        collision_vec[dim] += 1.0
                        self.ps.position[i][dim] = self.ps.domain_end[dim] - self.ps.padding
                    elif pos[dim] < self.ps.padding:
                        collision_vec[dim] -= 1.0
                        self.ps.position[i][dim] = self.ps.padding
                collision_vec_normal = collision_vec.norm()
                if collision_vec_normal > 1e-6:
                    self.simulate_collision(i, collision_vec / collision_vec_normal)

    def initialize(self):
        self.ps.update_particle_system()
        self.compute_volume_of_boundary_particle()

    def substep(self):
        pass

    def step(self):
        self.ps.update_particle_system()
        self.substep()
        self.enforce_boundary_3D()
