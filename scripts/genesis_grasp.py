import numpy as np
import genesis as gs

# initializing
gs.init(backend=gs.gpu)

# generating the scene
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
        substeps = 10, # for stable contacts
    ),
    show_viewer = True,
)

# adding an entity in the scene
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.038, 0.038, 0.038),
        pos  = (0.645, 0.0, 0.02),
    )
)
iiwa = scene.add_entity(
    gs.morphs.MJCF(file='/dataset/xmls/iiwa/iiwa.xml'),
)

jnt_names = [
    'iiwa_joint_1',
    'iiwa_joint_2',
    'iiwa_joint_3',
    'iiwa_joint_4',
    'iiwa_joint_5',
    'iiwa_joint_6',
    'iiwa_joint_7',
    'hande_left_finger_joint',
    'hande_right_finger_joint',
]
dofs_idx = [iiwa.get_joint(name).dof_idx_local for name in jnt_names]

# builduing the scene
scene.build()

# degrees of freedom
motors_dof = np.arange(7)
fingers_dof = np.array([7, 8])

# setting the position gains
iiwa.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 500, 500]),
    dofs_idx_local = dofs_idx,
)

# setting the velocity gains
iiwa.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 50, 50]),
    dofs_idx_local = dofs_idx,
)

# setting the force range for the safety
iiwa.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)

# get the eef's link
end_effector = iiwa.get_link('hand')

# calculatuing the inverse kinematics of single target link
qpos = iiwa.inverse_kinematics(
    link = end_effector,  # link showing the end-effector
    pos  = np.array([0.65, 0.0, 0.25]),  # position
    quat = np.array([0, 1, 0, 0]),  # quaternion
)

# planning the path to qpos_goal
qpos[-2:] = 0.00
path = iiwa.plan_path(
    qpos_goal     = qpos,  # goal joint position
    num_waypoints = 200,
)

# execute the motion plan
for waypoint in path:
    # setting target position
    iiwa.control_dofs_position(waypoint)
    scene.step()

# run steps
for i in range(100):
    scene.step()

# reaching
qpos = iiwa.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.16]),
    quat = np.array([0, 1, 0, 0]),
)
iiwa.control_dofs_position(qpos[:7], motors_dof)
iiwa.control_dofs_position(np.array([0.00, 0.00]), fingers_dof)

# run steps
for i in range(100):
    scene.step()

# grasping
iiwa.control_dofs_position(qpos[:7], motors_dof)
iiwa.control_dofs_force(np.array([80, 80]), fingers_dof)

# run steps
for i in range(100):
    scene.step()

# lifting
qpos = iiwa.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.3]),
    quat=np.array([0, 1, 0, 0]),
)
iiwa.control_dofs_position(qpos[:7], motors_dof)

# run steps
for i in range(200):
    scene.step()
