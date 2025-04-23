import os
import xacro
import launch
import launch_ros


def generate_launch_description():
    pkg_path = launch_ros.substitutions.FindPackageShare(
        package='iiwa_description').find('iiwa_description')
    xacro_path = os.path.join(pkg_path, 'urdf/iiwa14_rq140.xacro')
    robot_xacro = xacro.process_file(xacro_path)
    params = {'robot_description': robot_xacro.toxml()}
    
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )
    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[params],
        condition=launch.conditions.UnlessCondition(
            launch.substitutions.LaunchConfiguration('gui'))
    )
    joint_state_publisher_gui_node = launch_ros.actions.Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=launch.conditions.IfCondition(
            launch.substitutions.LaunchConfiguration('gui'))
    )
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', [os.path.join(pkg_path, 'rviz', 'display_iiwa.rviz')]]
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='gui',
            default_value='True',
            description='This is a flag for joint_state_publisher_gui'),
        launch.actions.DeclareLaunchArgument(
            name='model',
            default_value=xacro_path,
            description='Path to the urdf model file'),
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ]) 