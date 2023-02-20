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
ps=particle_system.ParticleSystem(simulation_config)

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    scene.point_light((2, 2, 2), color=(1, 1, 1))
    scene.lines(box_vertex_point, width=3.0, indices=box_edge_index, color=(0, 0, 0))
    canvas.scene(scene)
    window.show()
