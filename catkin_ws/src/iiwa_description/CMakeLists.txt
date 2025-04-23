cmake_minimum_required(VERSION 3.0.2)
project(iiwa_description)

find_package(catkin REQUIRED COMPONENTS
    xacro
)

include_directories(
# include
    ${catkin_INCLUDE_DIRS}
)
