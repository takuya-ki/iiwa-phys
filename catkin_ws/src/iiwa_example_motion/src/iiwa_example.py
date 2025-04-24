#!/usr/bin/env python3

import tf
import rospy
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion


def euler_to_quaternion(euler):
    """Convert Euler Angles to Quaternion
    euler: geometry_msgs/Vector3
    quaternion: geometry_msgs/Quaternion
    """
    q = tf.transformations.quaternion_from_euler(
        euler.x, euler.y, euler.z)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def run():
    rospy.init_node("iiwa_example_motion")
    robot = moveit_commander.RobotCommander()

    rospy.sleep(rospy.Duration(5.0))
    print("=" * 10, " Robot Groups:")
    print(robot.get_group_names())
    print("=" * 10, " Printing robot state")
    print(robot.get_current_state())
    print("=" * 10)

    arm = moveit_commander.MoveGroupCommander("arm")
    print("=" * 10, " Reference frame: %s" % arm.get_planning_frame())
    print("=" * 10, " Reference frame: %s" % arm.get_end_effector_link())

    arm_initial_pose = arm.get_current_pose().pose
    print("=" * 10, " Printing arm initial pose: ")
    print(arm_initial_pose)

    print("=" * 10," Moving ...")
    target_pose = geometry_msgs.msg.Pose()
    target_pose.orientation = Quaternion(
        x=0.00010335,
        y=-0.7071,
        z=0.70712,
        w=-7.2472e-05)
    target_pose.position.x = 0.16759
    target_pose.position.y = -0.052212
    target_pose.position.z = 1.0183
    arm.set_pose_target(target_pose)
    arm.go(wait=True)
    rospy.sleep(rospy.Duration(3.0))

    print("=" * 10, " Initializing pose ...")
    arm.set_pose_target(arm_initial_pose)
    arm.go(wait=True)
    rospy.sleep(rospy.Duration(2.0))


if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass