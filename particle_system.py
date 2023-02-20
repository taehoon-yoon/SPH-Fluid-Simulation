import taichi as ti
import numpy as np
import trimesh as tm


@ti.data_oriented
class ParticleSystem:
    def __init__(self, simulation_config):
        self.simulation_config = simulation_config
        self.config = self.simulation_config['Configuration']
        self.rigidBodiesConfig = self.simulation_config['RigidBodies']  # list
        self.fluidBlocksConfig = self.simulation_config['FluidBlocks']  # list

        self.domain_start = np.array(self.config['domainStart'])
        self.domain_end = np.array(self.config['domainEnd'])
        self.particle_radius = self.config['particleRadius']
        self.density0 = self.config['density0']

        self.dim = len(self.domain_start)
        self.domain_size = self.domain_end - self.domain_start
        self.particle_diameter = 2 * self.particle_radius
        # TODO: Check coefficient (0.8 * self.particle_diameter ** self.dim)
        self.particle_volume = (4 / 3) * np.pi * (self.particle_radius ** self.dim)
        self.support_length = 4 * self.particle_radius
        self.grid_size = self.support_length
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(np.int32)
        self.material_rigid = 0
        self.material_fluid = 1
        self.object_collection = dict()
        self.rigid_object_id = set()
        self.memory_allocated_particle_num = 0

        # ========== Compute number of particles ==========#
        #### Process Fluid Blocks ####
        print("\n=================================================================")
        print("=                        Fluid Blocks                           =")
        print("=================================================================")
        self.total_fluid_particle_num = 0
        for fluid in self.fluidBlocksConfig:
            fluid_particle_num = self.compute_fluid_particle_num(fluid['start'], fluid['end'])
            fluid['particleNum'] = fluid_particle_num
            self.object_collection[fluid['objectId']] = fluid
            self.total_fluid_particle_num += fluid_particle_num
            print("* Object ID: {}         Fluid particle number: {}".format(fluid['objectId'], fluid_particle_num))
            print("-----------------------------------------------------------------")
        print("Total fluid particle number: {}".format(self.total_fluid_particle_num))
        print("-----------------------------------------------------------------")

        #### Process Rigid Bodies ####
        print("\n=================================================================")
        print("=                        Rigid Bodies                           =")
        print("=================================================================")
        self.total_rigid_particle_num = 0
        for rigid_body in self.rigidBodiesConfig:
            voxelized_points = self.load_rigid_body(rigid_body)
            rigid_particle_num = voxelized_points.shape[0]
            rigid_body['particleNum'] = rigid_particle_num
            rigid_body['voxelizedPoints'] = voxelized_points
            self.object_collection[rigid_body['objectId']] = rigid_body
            self.rigid_object_id.add(rigid_body['objectId'])
            self.total_rigid_particle_num += rigid_particle_num
            print("* Object ID: {}         Rigid Body particle number: {}".format(rigid_body['objectId'],
                                                                                  rigid_particle_num))
            print("-----------------------------------------------------------------")
        print("Total rigid particle number: {}".format(self.total_rigid_particle_num))
        print("-----------------------------------------------------------------")

        self.total_particle_num = self.total_rigid_particle_num + self.total_fluid_particle_num

        # ========== Allocate memory ==========#
        # Grid Related
        total_grid_num = 1
        for i in range(self.dim):
            total_grid_num *= self.grid_num[i]
        self.counting_sort_countArray = ti.field(dtype=ti.i32, shape=total_grid_num)
        self.counting_sort_accumulatedArray = ti.field(dtype=ti.i32, shape=total_grid_num)
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(total_grid_num)

        self.grid_id = ti.field(dtype=ti.i32, shape=self.total_particle_num)
        self.grid_id_buffer = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        # Particle Related
        self.object_id = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        self.position = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.position0 = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)

        self.volume = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.mass = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.density = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.pressure = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.material = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        self.color = ti.Vector.field(3, dtype=ti.i32, shape=self.total_particle_num)
        self.is_dynamic = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        # Buffer for sort
        self.object_id_buffer = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        self.position_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.position0_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.velocity_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)

        self.volume_buffer = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.mass_buffer = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.density_buffer = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.pressure_buffer = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.material_buffer = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        self.color_buffer = ti.Vector.field(3, dtype=ti.i32, shape=self.total_particle_num)
        self.is_dynamic_buffer = ti.field(dtype=ti.i32, shape=self.total_particle_num)

    def compute_fluid_particle_num(self, start, end):
        particle_num = 1
        for i in range(self.dim):
            particle_num *= len(np.arange(start[i], end[i], self.particle_diameter))
        return particle_num

    def load_rigid_body(self, rigid_body):
        mesh = tm.load(rigid_body['geometryFile'])
        mesh.apply_scale(rigid_body['scale'])
        offset = np.array(rigid_body['translation'])
        rotation_angle = rigid_body['rotationAngle'] * np.pi / 180
        rotation_axis = rigid_body['rotationAxis']
        rot_matrix = tm.transformations.rotation_matrix(rotation_angle, rotation_axis, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
        rigid_body['mesh'] = mesh.copy()
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        return voxelized_mesh.points

    @ti.kernel
    def add_particles(self,
                      object_id: int,
                      particle_num: int,
                      position: ti.types.ndarray(),
                      velocity: ti.types.ndarray(),
                      density: ti.types.ndarray(),
                      pressure: ti.types.ndarray(),
                      material: ti.types.ndarray(),
                      color: ti.types.ndarray(),
                      is_dynamic: ti.types.ndarray()):
        for idx in range(self.memory_allocated_particle_num, self.memory_allocated_particle_num + particle_num):
            relative_idx = idx - self.memory_allocated_particle_num
            pos = ti.Vector.zero(ti.f32, self.dim)
            vel = ti.Vector(ti.f32, self.dim)
            acc = ti.Vector(ti.f32, self.dim)
            col = ti.Vector([0, 0, 0])
            for dim_idx in ti.static(range(self.dim)):
                pos[dim_idx] = position[relative_idx, dim_idx]
                vel[dim_idx] = velocity[relative_idx, dim_idx]
            for dim_idx in ti.static(range(3)):
                col[dim_idx] = color[relative_idx, dim_idx]
            self.object_id[idx] = object_id
            self.position[idx] = pos
            self.position0[idx] = pos
            self.velocity[idx] = vel
            self.acceleration[idx] = acc

            self.volume[idx] = self.particle_volume
            self.density[idx] = density[relative_idx]
            self.mass[idx] = self.volume[idx] * self.density[idx]
            self.pressure[idx] = pressure[relative_idx]
            self.material[idx] = material[relative_idx]

            self.color[idx] = col
            self.is_dynamic[idx] = is_dynamic[relative_idx]

        self.memory_allocated_particle_num += particle_num

    def add_cube(self,object_id, box_start, box_end, velocity, density, color, is_dynamic, material):

        pass