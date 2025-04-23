cmake_minimum_required(VERSION 3.5)
project(iiwa_description)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)
find_package(rclcpp REQUIRED)
find_package(urdf REQUIRED)
find_package(robot_state_publisher REQUIRED)

# Install directories
install(DIRECTORY urdf/
    DESTINATION share/${PROJECT_NAME}/urdf
)

install(DIRECTORY meshes/
    DESTINATION share/${PROJECT_NAME}/meshes
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY rviz/
    DESTINATION share/${PROJECT_NAME}/rviz
)

install(DIRECTORY ros2_control/
    DESTINATION share/${PROJECT_NAME}/ros2_control
)

install(DIRECTORY gazebo/
    DESTINATION share/${PROJECT_NAME}/gazebo
)

ament_package()