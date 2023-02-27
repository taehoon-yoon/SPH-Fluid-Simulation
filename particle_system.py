import taichi as ti
import numpy as np
import trimesh as tm
import WCSPH


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
        self.padding = self.support_length  # padding is used for boundary condition when particle collide with wall
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(np.int32)
        self.material_rigid = 0
        self.material_fluid = 1
        self.memory_allocated_particle_num = ti.field(dtype=ti.i32, shape=())
        self.memory_allocated_particle_num[None] = 0
        self.cur_obj_id = 0

    def memory_allocation_and_initialization_only_position(self):
        self.memory_allocated_particle_num[None] = 0
        # ========== Compute number of particles ==========#
        # === Process Fluid Blocks ===
        self.total_fluid_particle_num = 0
        for fluid in self.fluidBlocksConfig:
            fluid_particle_num = self.compute_fluid_particle_num(fluid['start'], fluid['end'])
            fluid['particleNum'] = fluid_particle_num
            self.total_fluid_particle_num += fluid_particle_num
            self.cur_obj_id = ti.max(self.cur_obj_id, fluid['objectId'])

        # === Process Rigid Bodies ===

        self.total_rigid_particle_num = 0
        self.mesh_vertices = []
        self.mesh_indices = []

        for rigid_body in self.rigidBodiesConfig:
            voxelized_points = self.load_rigid_body(rigid_body)
            rigid_particle_num = voxelized_points.shape[0]
            rigid_body['particleNum'] = rigid_particle_num
            rigid_body['voxelizedPoints'] = voxelized_points

            self.total_rigid_particle_num += rigid_particle_num
            self.cur_obj_id = ti.max(self.cur_obj_id, rigid_body['objectId'])

        self.total_particle_num = self.total_rigid_particle_num + self.total_fluid_particle_num

        self.position = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_particle_num)
        self.material = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        # ========== Initialize particles ==========#

        # Fluid block
        for fluid in self.fluidBlocksConfig:
            offset = np.array(fluid['translation'])
            start = np.array(fluid['start'])
            end = np.array(fluid['end'])
            color = fluid['color']
            if type(color[0]) == int:
                color = [c / 255.0 for c in color]
            self.add_cube(box_start=start + offset, box_end=end + offset, color=color, material=self.material_fluid)

        # Rigid bodies
        for rigid_body in self.rigidBodiesConfig:
            rigid_body_particle_num = rigid_body['particleNum']
            color = rigid_body['color']
            if type(color[0]) == int:
                color = [c / 255.0 for c in color]
            self.add_particles_only_position(
                particle_num=rigid_body_particle_num,
                position=rigid_body['voxelizedPoints'],
                material=np.full((rigid_body_particle_num,), self.material_rigid, dtype=np.int32),
                color=np.tile(np.array(color, dtype=np.float32), (rigid_body_particle_num, 1)))

    def memory_allocation_and_initialization(self):
        self.memory_allocated_particle_num[None] = 0
        self.object_collection = dict()
        self.rigid_object_id = set()

        # === Process Fluid Blocks ===
        for fluid in self.fluidBlocksConfig:
            self.object_collection[fluid['objectId']] = fluid

        # === Process Rigid Bodies ===
        self.rigid_bodies_sigma = ti.field(dtype=ti.f32,
                                           shape=len(self.rigidBodiesConfig) + len(self.fluidBlocksConfig))
        # Fluid part does not require sigma info.
        # But to access it with object Id, we include fluid as well. Sigma of fluid is not used in program.
        # Sigma is the viscosity coefficient between fluid and rigid
        for rigid_body in self.rigidBodiesConfig:
            self.object_collection[rigid_body['objectId']] = rigid_body
            self.rigid_object_id.add(rigid_body['objectId'])
            self.rigid_bodies_sigma[rigid_body['objectId']] = rigid_body['sigma']

        # ========== Allocate memory ==========#
        # Grid Related
        total_grid_num = 1
        for i in range(self.dim):
            total_grid_num *= self.grid_num[i]
        self.counting_sort_countArray = ti.field(dtype=ti.i32, shape=total_grid_num)
        self.counting_sort_accumulatedArray = ti.field(dtype=ti.i32, shape=total_grid_num)
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.counting_sort_accumulatedArray.shape[0])
        # Don't know why but ti.algorithms.PrefixSumExecutor(total_grid_num) is error.

        self.grid_id = ti.field(dtype=ti.i32, shape=self.total_particle_num)
        self.grid_id_buffer = ti.field(dtype=ti.i32, shape=self.total_particle_num)
        self.grid_id_for_sort = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        # Particle Related
        self.object_id = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        self.velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)

        self.volume = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.mass = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.density = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.pressure = ti.field(dtype=ti.f32, shape=self.total_particle_num)

        self.is_dynamic = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        # Buffer for sort
        self.object_id_buffer = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        self.position_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.velocity_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_particle_num)

        self.volume_buffer = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.mass_buffer = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.density_buffer = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.pressure_buffer = ti.field(dtype=ti.f32, shape=self.total_particle_num)
        self.material_buffer = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=self.total_particle_num)
        self.is_dynamic_buffer = ti.field(dtype=ti.i32, shape=self.total_particle_num)

        # Memory allocation for object mesh rendering
        self.fluid_only_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_fluid_particle_num)
        self.fluid_only_position = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.total_fluid_particle_num)
        self.tmp_cnt = ti.field(ti.i32, shape=())

        # ========== Initialize particles ==========#

        # Fluid block
        for fluid in self.fluidBlocksConfig:
            fluid_particle_num = fluid['particleNum']
            velocity = np.tile(np.array(fluid['velocity'], dtype=np.float32), (fluid_particle_num, 1))
            density = fluid['density']
            self.add_particles(object_id=fluid['objectId'],
                               particle_num=fluid_particle_num,
                               velocity=velocity,
                               density=np.full((fluid_particle_num,), density, dtype=np.float32),
                               pressure=np.full((fluid_particle_num,), 0.0, dtype=np.float32),
                               is_dynamic=np.full((fluid_particle_num,), 1, dtype=np.int32))

        # Rigid bodies
        for rigid_body in self.rigidBodiesConfig:

            rigid_body_particle_num = rigid_body['particleNum']
            rigid_body_is_dynamic = 1 if rigid_body['isDynamic'] else 0
            if rigid_body_is_dynamic:
                velocity = np.tile(np.array(rigid_body['velocity'], dtype=np.float32), (rigid_body_particle_num, 1))
            else:
                velocity = np.full((rigid_body_particle_num, self.dim), 0.0, dtype=np.float32)
            density = rigid_body['density']
            self.add_particles(object_id=rigid_body['objectId'],
                               particle_num=rigid_body_particle_num,
                               velocity=velocity,
                               density=np.full((rigid_body_particle_num,), density, dtype=np.float32),
                               pressure=np.full((rigid_body_particle_num,), 0.0, dtype=np.float32),
                               is_dynamic=np.full((rigid_body_particle_num,), rigid_body_is_dynamic, dtype=np.int32))

    def free_memory_allocation(self):
        del self.object_collection
        del self.rigid_object_id
        del self.mesh_vertices
        del self.mesh_indices
        del self.rigid_bodies_sigma

        del self.counting_sort_countArray
        del self.counting_sort_accumulatedArray
        del self.prefix_sum_executor

        del self.grid_id
        del self.grid_id_buffer
        del self.grid_id_for_sort

        del self.object_id
        del self.position
        del self.velocity
        del self.acceleration
        del self.volume
        del self.mass
        del self.density
        del self.pressure
        del self.material
        del self.color
        del self.is_dynamic

        del self.object_id_buffer
        del self.position_buffer
        del self.velocity_buffer
        del self.acceleration_buffer
        del self.volume_buffer
        del self.mass_buffer
        del self.density_buffer
        del self.pressure_buffer
        del self.material_buffer
        del self.color_buffer
        del self.is_dynamic_buffer

        del self.fluid_only_color
        del self.fluid_only_position
        del self.tmp_cnt

    def compute_fluid_particle_num(self, start, end):
        particle_num = 1
        for i in range(self.dim):
            particle_num *= len(np.arange(start[i], end[i], self.particle_diameter))
        return particle_num

    @ti.kernel
    def update_fluid_position_info(self):
        self.tmp_cnt[None] = 0
        for i in self.position:
            if self.material[i] == self.material_fluid:
                self.fluid_only_position[ti.atomic_add(self.tmp_cnt[None], 1)] = self.position[i]

    @ti.kernel
    def update_fluid_color_info(self):
        self.tmp_cnt[None] = 0
        for i in self.color:
            if self.material[i] == self.material_fluid:
                self.fluid_only_color[ti.atomic_add(self.tmp_cnt[None], 1)] = self.color[i]

    @ti.kernel
    def update_mesh_info(self, vertices: ti.types.ndarray(), indices: ti.types.ndarray(),
                         ti_vertices: ti.template(), ti_indices: ti.template()):
        for i in range(vertices.shape[0]):
            vec = ti.Vector.zero(ti.f32, self.dim)
            for j in ti.static(range(self.dim)):
                vec[j] = vertices[i, j]
            ti_vertices[i] = vec
        for i in range(indices.shape[0]):
            ti_indices[i] = indices[i]

    def get_mesh_info(self, mesh):
        mesh_vertices = np.array(mesh.vertices, dtype=np.float32)
        mesh_indices = np.array(mesh.faces, dtype=np.int32).flatten()
        ti_mesh_vertices = ti.Vector.field(self.dim, dtype=ti.f32, shape=mesh_vertices.shape[0])
        ti_mesh_indices = ti.field(ti.i32, shape=mesh_indices.shape[0])
        self.update_mesh_info(mesh_vertices, mesh_indices, ti_mesh_vertices, ti_mesh_indices)
        self.mesh_vertices.append(ti_mesh_vertices)
        self.mesh_indices.append(ti_mesh_indices)

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
        self.get_mesh_info(mesh)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        return voxelized_mesh.points.astype(np.float32)

    @ti.kernel
    def add_particles_only_position(self,
                                    particle_num: int,
                                    position: ti.types.ndarray(),
                                    material: ti.types.ndarray(),
                                    color: ti.types.ndarray()):
        for idx in range(self.memory_allocated_particle_num[None],
                         self.memory_allocated_particle_num[None] + particle_num):
            relative_idx = idx - self.memory_allocated_particle_num[None]
            pos = ti.Vector.zero(ti.f32, self.dim)
            col = ti.Vector([0.0, 0.0, 0.0])
            for dim_idx in ti.static(range(self.dim)):
                pos[dim_idx] = position[relative_idx, dim_idx]
            for dim_idx in ti.static(range(3)):
                col[dim_idx] = color[relative_idx, dim_idx]
            self.position[idx] = pos
            self.material[idx] = material[relative_idx]
            self.color[idx] = col
        self.memory_allocated_particle_num[None] += particle_num

    @ti.kernel
    def add_particles(self,
                      object_id: int,
                      particle_num: int,
                      velocity: ti.types.ndarray(),
                      density: ti.types.ndarray(),
                      pressure: ti.types.ndarray(),
                      is_dynamic: ti.types.ndarray()):
        for idx in range(self.memory_allocated_particle_num[None],
                         self.memory_allocated_particle_num[None] + particle_num):
            relative_idx = idx - self.memory_allocated_particle_num[None]
            vel = ti.Vector.zero(ti.f32, self.dim)
            acc = ti.Vector.zero(ti.f32, self.dim)
            for dim_idx in ti.static(range(self.dim)):
                vel[dim_idx] = velocity[relative_idx, dim_idx]
            self.object_id[idx] = object_id
            self.velocity[idx] = vel
            self.acceleration[idx] = acc

            self.volume[idx] = self.particle_volume
            self.density[idx] = density[relative_idx]
            self.mass[idx] = self.volume[idx] * self.density[idx]
            self.pressure[idx] = pressure[relative_idx]
            self.is_dynamic[idx] = is_dynamic[relative_idx]
        self.memory_allocated_particle_num[None] += particle_num

    def add_cube(self, box_start, box_end, color, material):
        dim_array = []
        total_cube_particle_num = 1
        for i in range(self.dim):
            dim_array.append(np.arange(box_start[i], box_end[i], self.particle_diameter))
            total_cube_particle_num *= len(dim_array[i])
        position_arr = np.array(np.meshgrid(*dim_array, indexing='ij'), dtype=np.float32)
        # (3, len(dim_array[0]), len(dim_array[1]), len(dim_array[2]))

        position_arr = position_arr.reshape(self.dim, total_cube_particle_num).T
        """
        - position_arr
        [
        [ x_start, y_start, z_start],
        [x_start, y_start, z_start + 0.02],
        [x_start, y_start, z_start + 0.04]
        ...
        [x_start, y_start, z_end],
        [x_start, y_start + 0.02, z_start],
        [x_start, y_start + 0.02, z_start + 0.02],
        ...
        [x_start, y_end, z_end],
        [x_start + 0.02, y_start, z_start],
        [x_start + 0.02, y_start, z_start + 0.02],
        ...
        [x_end, y_end, z_end]
        ]
        """
        material_arr = np.full((total_cube_particle_num,), material, dtype=np.int32)
        color_arr = np.tile(np.array(color, dtype=np.float32), (total_cube_particle_num, 1))
        self.add_particles_only_position(total_cube_particle_num, position_arr, material_arr, color_arr)

    @ti.func
    def pos2index(self, position):
        return (position / self.grid_size).cast(ti.i32)

    @ti.func
    def flatten_grid_index(self, grid_idx):
        flatten_grid_idx = 0
        # flatten_grid_idx = 0 We need this, if I omit it, taichi outputs error : Name "flatten_grid_idx" is not defined
        if self.dim == 3:
            flatten_grid_idx = grid_idx[0] * self.grid_num[1] * self.grid_num[2] + grid_idx[1] * self.grid_num[2] + \
                               grid_idx[2]
        else:
            flatten_grid_idx = grid_idx[0] * self.grid_num[1] + grid_idx[1]
        return flatten_grid_idx

    @ti.func
    def get_grid_idx_from_pos(self, position):
        grid_idx = self.pos2index(position)  # floor operation
        return self.flatten_grid_index(grid_idx)

    @ti.kernel
    def update_grid_id(self):
        self.counting_sort_accumulatedArray.fill(0)
        for i in self.position:
            self.grid_id[i] = self.get_grid_idx_from_pos(self.position[i])
            self.counting_sort_accumulatedArray[self.grid_id[i]] += 1
        for i in self.counting_sort_accumulatedArray:
            self.counting_sort_countArray[i] = self.counting_sort_accumulatedArray[i]

    @ti.kernel
    def counting_sort(self):
        for i in range(self.total_particle_num):
            grid_idx = self.grid_id[i]
            base_offset = 0 if grid_idx == 0 else self.counting_sort_accumulatedArray[grid_idx - 1]
            self.grid_id_for_sort[i] = ti.atomic_sub(self.counting_sort_countArray[grid_idx], 1) + base_offset - 1
        for i in self.grid_id_for_sort:
            new_idx = self.grid_id_for_sort[i]
            self.grid_id_buffer[new_idx] = self.grid_id[i]

            self.object_id_buffer[new_idx] = self.object_id[i]
            self.position_buffer[new_idx] = self.position[i]
            self.velocity_buffer[new_idx] = self.velocity[i]
            self.acceleration_buffer[new_idx] = self.acceleration[i]

            self.volume_buffer[new_idx] = self.volume[i]
            self.density_buffer[new_idx] = self.density[i]
            self.mass_buffer[new_idx] = self.mass[i]
            self.pressure_buffer[new_idx] = self.pressure[i]
            self.material_buffer[new_idx] = self.material[i]

            self.color_buffer[new_idx] = self.color[i]
            self.is_dynamic_buffer[new_idx] = self.is_dynamic[i]

        for i in self.grid_id:
            self.grid_id[i] = self.grid_id_buffer[i]

            self.object_id[i] = self.object_id_buffer[i]
            self.position[i] = self.position_buffer[i]
            self.velocity[i] = self.velocity_buffer[i]
            self.acceleration[i] = self.acceleration_buffer[i]

            self.volume[i] = self.volume_buffer[i]
            self.density[i] = self.density_buffer[i]
            self.mass[i] = self.mass_buffer[i]
            self.pressure[i] = self.pressure_buffer[i]
            self.material[i] = self.material_buffer[i]

            self.color[i] = self.color_buffer[i]
            self.is_dynamic[i] = self.is_dynamic_buffer[i]

    @ti.func
    def for_all_neighbors(self, idx_i, task: ti.template(), ret: ti.template()):
        center_cell_grid_idx = self.pos2index(self.position[idx_i])
        for offset in ti.grouped(ti.ndrange(*(((-1, 2),) * self.dim))):
            neighbor_grid_flatten_idx = self.flatten_grid_index(offset + center_cell_grid_idx)
            start_idx = 0 if neighbor_grid_flatten_idx == 0 else self.counting_sort_accumulatedArray[
                neighbor_grid_flatten_idx - 1]
            # TODO: can we somewhat modify to enable using ti.static?
            for idx_j in range(start_idx, self.counting_sort_accumulatedArray[neighbor_grid_flatten_idx]):
                if idx_i != idx_j and (self.position[idx_i] - self.position[idx_j]).norm() < self.support_length:
                    task(idx_i, idx_j, ret)

    def update_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.counting_sort_accumulatedArray)
        self.counting_sort()

    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_rigid and (not self.is_dynamic[p])

    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_rigid and self.is_dynamic[p]

    def build_solver(self):
        return WCSPH.WCSPHSolver(self)

    def reset_particle_system(self):
        self.memory_allocated_particle_num[None] = 0
        for fluid in self.fluidBlocksConfig:
            offset = np.array(fluid['translation'])
            start = np.array(fluid['start'])
            end = np.array(fluid['end'])
            color = fluid['color']
            if type(color[0]) == int:
                color = [c / 255.0 for c in color]
            self.add_cube(box_start=start + offset, box_end=end + offset, color=color, material=self.material_fluid)

        for rigid_body in self.rigidBodiesConfig:
            rigid_body_particle_num = rigid_body['particleNum']
            color = rigid_body['color']
            if type(color[0]) == int:
                color = [c / 255.0 for c in color]
            density = rigid_body['density']
            self.add_particles_only_position(
                particle_num=rigid_body_particle_num,
                position=rigid_body['voxelizedPoints'],
                material=np.full((rigid_body_particle_num,), self.material_rigid, dtype=np.int32),
                color=np.tile(np.array(color, dtype=np.float32), (rigid_body_particle_num, 1)))

        self.memory_allocated_particle_num[None] = 0
        for fluid in self.fluidBlocksConfig:
            fluid_particle_num = fluid['particleNum']
            velocity = np.tile(np.array(fluid['velocity'], dtype=np.float32), (fluid_particle_num, 1))
            density = fluid['density']
            self.add_particles(object_id=fluid['objectId'],
                               particle_num=fluid_particle_num,
                               velocity=velocity,
                               density=np.full((fluid_particle_num,), density, dtype=np.float32),
                               pressure=np.full((fluid_particle_num,), 0.0, dtype=np.float32),
                               is_dynamic=np.full((fluid_particle_num,), 1, dtype=np.int32))

        for rigid_body in self.rigidBodiesConfig:
            rigid_body_is_dynamic = 1 if rigid_body['isDynamic'] else 0
            if rigid_body_is_dynamic:
                velocity = np.tile(np.array(rigid_body['velocity'], dtype=np.float32), (rigid_body_particle_num, 1))
            else:
                velocity = np.full((rigid_body_particle_num, self.dim), 0.0, dtype=np.float32)
            self.add_particles(object_id=rigid_body['objectId'],
                               particle_num=rigid_body_particle_num,
                               velocity=velocity,
                               density=np.full((rigid_body_particle_num,), density, dtype=np.float32),
                               pressure=np.full((rigid_body_particle_num,), 0.0, dtype=np.float32),
                               is_dynamic=np.full((rigid_body_particle_num,), rigid_body_is_dynamic, dtype=np.int32))

    def dump(self):
        return self.fluid_only_position.to_numpy()
