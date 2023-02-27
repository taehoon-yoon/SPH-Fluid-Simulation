import taichi as ti
import json
import particle_system
import numpy as np
import os

ti.init(arch=ti.gpu)

with open('./data/scenes/dragon_bath.json', 'r') as f:
    simulation_config = json.load(f)

config = simulation_config['Configuration']

box_x, box_y, box_z = config['domainEnd']

box_vertex_point = ti.Vector.field(3, dtype=ti.f32, shape=8)
box_vertex_point[0] = [0., 0., 0.]
box_vertex_point[1] = [0., box_y, 0.]
box_vertex_point[2] = [box_x, 0., 0.]
box_vertex_point[3] = [box_x, box_y, 0.]

box_vertex_point[4] = [0., 0., box_z]
box_vertex_point[5] = [0., box_y, box_z]
box_vertex_point[6] = [box_x, 0., box_z]
box_vertex_point[7] = [box_x, box_y, box_z]

box_edge_index = ti.field(dtype=ti.i32, shape=24)
for i, idx in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
    box_edge_index[i] = idx

window = ti.ui.Window("SPH", (1500, 1000))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(6.5, 3.5, 5)
camera.lookat(-1, -1.5, -3)
scene.set_camera(camera)
canvas.set_background_color((1, 1, 1))

ps = particle_system.ParticleSystem(simulation_config)
ps.memory_allocation_and_initialization_only_position()
substep = config['numberOfStepsPerRenderUpdate']

draw_object_in_mesh = False
gui = ti.ui.Gui(window.get_gui())
start_step = False
current_fluid_domain_start = [np.array(fluid['start']) for fluid in ps.fluidBlocksConfig]
current_fluid_domain_end = [np.array(fluid['end']) for fluid in ps.fluidBlocksConfig]
fluid_box_num = len(current_fluid_domain_start)

safe_boundary_start = ps.domain_start + np.array([ps.padding + ps.particle_radius])
safe_boundary_end = ps.domain_end - np.array([ps.padding + ps.particle_radius])
reallocate_memory_flag = False
object_config = ps.rigidBodiesConfig.copy()
include_rigid_object = True
pre_include_rigid_object = True

scene_name = 'Dragon Bath'
output_frames = False
output_interval = config['outputInterval']
output_ply = False
cnt = 0
cnt_ply = 0
series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
enter_second_phase_first_time = True
reset_scene_flag = False

while window.running:
    if start_step:
        for i in range(substep):
            solver.step()

    camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)

    gui.begin('Widget', 0, 0, 0.15, 1.0)
    gui.text("SPH Particle System")
    if not start_step:
        if gui.button('Start'):
            start_step = True
            ps.memory_allocation_and_initialization()
            solver = ps.build_solver()
            solver.initialize()
            draw_object_in_mesh = True
        if gui.button('Add Fluid Block'):
            cur_object_id = ps.cur_obj_id
            recent_fluid_config = ps.fluidBlocksConfig[-1]
            new_fluid_config = recent_fluid_config.copy()
            new_fluid_config['objectId'] = cur_object_id + 1
            ps.fluidBlocksConfig.append(new_fluid_config)
            current_fluid_domain_start = [np.array(fluid['start']) for fluid in ps.fluidBlocksConfig]
            current_fluid_domain_end = [np.array(fluid['end']) for fluid in ps.fluidBlocksConfig]
            fluid_box_num = len(current_fluid_domain_start)
            reallocate_memory_flag = True
        if gui.button('Delete Recent Fluid Block'):
            del ps.fluidBlocksConfig[-1]
            current_fluid_domain_start = [np.array(fluid['start']) for fluid in ps.fluidBlocksConfig]
            current_fluid_domain_end = [np.array(fluid['end']) for fluid in ps.fluidBlocksConfig]
            fluid_box_num = len(current_fluid_domain_start)
            reallocate_memory_flag = True
        include_rigid_object = gui.checkbox('Include Rigid Object', include_rigid_object)
        if include_rigid_object != pre_include_rigid_object:
            pre_include_rigid_object = include_rigid_object
            reallocate_memory_flag = True
        for idx in range(fluid_box_num):
            gui.text('----------------------------')
            gui.text('Fluid Box Number {}'.format(idx + 1))
            gui.text('Fluid Block start point')
            start_x = gui.slider_float('x0_{}'.format(idx + 1), current_fluid_domain_start[idx][0],
                                       safe_boundary_start[0], current_fluid_domain_end[idx][0] - ps.particle_diameter)
            start_y = gui.slider_float('y0_{}'.format(idx + 1), current_fluid_domain_start[idx][1],
                                       safe_boundary_start[1], current_fluid_domain_end[idx][1] - ps.particle_diameter)
            start_z = gui.slider_float('z0_{}'.format(idx + 1), current_fluid_domain_start[idx][2],
                                       safe_boundary_start[2], current_fluid_domain_end[idx][2] - ps.particle_diameter)
            gui.text('')
            gui.text('Fluid Block end point')
            end_x = gui.slider_float('x1_{}'.format(idx + 1), current_fluid_domain_end[idx][0],
                                     current_fluid_domain_start[idx][0] + ps.particle_diameter, safe_boundary_end[0])
            end_y = gui.slider_float('y1_{}'.format(idx + 1), current_fluid_domain_end[idx][1],
                                     current_fluid_domain_start[idx][1] + ps.particle_diameter, safe_boundary_end[1])
            end_z = gui.slider_float('z1_{}'.format(idx + 1), current_fluid_domain_end[idx][2],
                                     current_fluid_domain_start[idx][2] + ps.particle_diameter, safe_boundary_end[2])
            start = np.array([start_x, start_y, start_z]).round(2)
            end = np.array([end_x, end_y, end_z]).round(2)
            if (current_fluid_domain_start[idx] != start).any() or (current_fluid_domain_end[idx] != end).any():
                reallocate_memory_flag = True
                current_fluid_domain_start[idx] = start
                current_fluid_domain_end[idx] = end

        if reallocate_memory_flag:
            cur_object_id = 1
            del ps
            ps = particle_system.ParticleSystem(simulation_config)
            if include_rigid_object:
                ps.rigidBodiesConfig = object_config
            else:
                ps.rigidBodiesConfig = list()
            for idx in range(fluid_box_num):
                ps.fluidBlocksConfig[idx]['start'] = current_fluid_domain_start[idx]
                ps.fluidBlocksConfig[idx]['end'] = current_fluid_domain_end[idx]
            ps.memory_allocation_and_initialization_only_position()
            reallocate_memory_flag = False

        gui.text('----------------------------')
        gui.text('# of Fluid Particles')
        gui.text('{}'.format(ps.total_fluid_particle_num))
        gui.text('# of Rigid Particles')
        gui.text('{}'.format(ps.total_rigid_particle_num))
        gui.text('Total # of Particles')
        gui.text('{}'.format(ps.total_particle_num))

        gui.text('----------------------------')
        output_frames = gui.checkbox('Output in Image', output_frames)
        output_ply = gui.checkbox('Output [.ply] files', output_ply)
        gui.end()
    else:
        if gui.button('Reset Scene'):
            ps.reset_particle_system()
            reset_scene_flag = True
        if gui.button('Reset View'):
            camera.position(6.5, 3.5, 5)
            camera.lookat(-1, -1.5, -3)
        draw_object_in_mesh = gui.checkbox('Draw object in mesh', draw_object_in_mesh)
        gui.text('----------------------------')
        gui.text('Euler step time interval')
        solver.dt[None] = gui.slider_float('[10^-3]', solver.dt[None] * 1000, 0.2, 0.8) * 0.001
        gui.text('Viscosity')
        solver.viscosity[None] = gui.slider_float('', solver.viscosity[None], 0.001, 0.5)
        gui.text('Surface Tension')
        solver.surface_tension[None] = gui.slider_float('[N/m]', solver.surface_tension[None], 0.001, 5)
        if solver.viscosity[None] > 0.23 or solver.surface_tension[None] > 2.0:
            # Viscosity with over 0.23 cause numerical instability when time step is larger than 0.0005 typically.
            # Surface tension with over 2.0 cause numerical instability when time step is larger than 0.0005 typically.
            solver.dt[None] = ti.min(solver.dt[None], 0.0005)
        if solver.viscosity[None] > 0.23 and solver.surface_tension[None] > 2.0:
            # Both in high viscosity and high surface tension, for numerical stability it is recommend to set 0.0004
            solver.dt[None] = ti.min(solver.dt[None], 0.0004)
        gui.text('----------------------------')
        gui.text('# of Fluid Particles')
        gui.text('{}'.format(ps.total_fluid_particle_num))
        gui.text('# of Rigid Particles')
        gui.text('{}'.format(ps.total_rigid_particle_num))
        gui.text('Total # of Particles')
        gui.text('{}'.format(ps.total_particle_num))
        gui.end()

    scene.set_camera(camera)
    scene.point_light((2, 2, 2), color=(1, 1, 1))
    scene.lines(box_vertex_point, width=3.0, indices=box_edge_index, color=(0, 0, 0))

    if draw_object_in_mesh:
        ps.update_fluid_position_info()
        ps.update_fluid_color_info()
        scene.particles(ps.fluid_only_position, radius=ps.particle_radius, per_vertex_color=ps.fluid_only_color)
        for i in range(len(ps.mesh_vertices)):
            scene.mesh(ps.mesh_vertices[i], ps.mesh_indices[i])
    else:
        scene.particles(ps.position, radius=ps.particle_radius, per_vertex_color=ps.color)
    canvas.scene(scene)
    if start_step:
        if reset_scene_flag:
            cnt = 0
            cnt_ply = 0
            reset_scene_flag = False
        if enter_second_phase_first_time:
            if output_frames:
                os.makedirs(f"{scene_name}_output_img", exist_ok=True)  # output image
            if output_ply:
                os.makedirs(f"{scene_name}_output", exist_ok=True)
            enter_second_phase_first_time = False

        if cnt % output_interval == 0:
            if output_ply:
                ps.update_fluid_position_info()
                np_position = ps.dump()
                writer = ti.tools.PLYWriter(num_vertices=ps.total_fluid_particle_num)
                writer.add_vertex_pos(np_position[:, 0], np_position[:, 1], np_position[:, 2])
                writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
                for r_body_id in ps.rigid_object_id:
                    with open(f"{scene_name}_output/obj_{r_body_id}_{cnt_ply:06}.obj", "w") as f:
                        e = ps.object_collection[r_body_id]["mesh"].export(file_type='obj')
                        f.write(e)
                cnt_ply += 1
            if output_frames:
                window.save_image(f"{scene_name}_output_img/{cnt:06}.png")
        cnt += 1
    window.show()
