import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        camera_pos = (3.5, -1.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov = 40,
    ),
    rigid_options = gs.options.RigidOptions(
        dt = 0.01,
    ),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    # gs.morphs.URDF(
    #     file='/dataset/urdfs/iiwa14_rqhe.urdf',
    #     fixed=True,
    # ),
    gs.morphs.MJCF(file="/dataset/xmls/iiwa/iiwa.xml"),
)

B = 20
scene.build(
    n_envs=B, 
    env_spacing=(1.0, 1.0)
)

for i in range(1000):
    scene.step()
