import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene()

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

scene.build()
for i in range(1000):
    scene.step()
