import taichi as ti
import json
import particle_system

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
solver = ps.build_solver()
solver.initialize()
substep = config['numberOfStepsPerRenderUpdate']

draw_object_in_mesh = True
gui = ti.ui.Gui(window.get_gui())

while window.running:
    for i in range(substep):
        solver.step()

    camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.RMB)
    """
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'm':
            draw_object_in_mesh = not draw_object_in_mesh
        elif window.event.key == 'r':
            camera.position(6.5, 3.5, 5)
            camera.lookat(-1, -1.5, -3)
        elif window.event.key == 'p':
            ps.reset_particle_system()
    """
    gui.begin('Widget', 0, 0, 0.15, 1.0)
    gui.text("SPH Particle System")
    if gui.button('Reset Scene'):
        ps.reset_particle_system()
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
    window.show()
