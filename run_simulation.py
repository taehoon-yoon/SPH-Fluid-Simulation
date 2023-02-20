import taichi as ti
import json
import particle_system
import numpy as np

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

draw_object_in_mesh = True
while window.running:
    camera.track_user_inputs(window, movement_speed=0.02, hold_key=ti.ui.LMB)
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'm':
            draw_object_in_mesh = not draw_object_in_mesh
        elif window.event.key == 'r':
            camera.position(6.5, 3.5, 5)
            camera.lookat(-1, -1.5, -3)
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
