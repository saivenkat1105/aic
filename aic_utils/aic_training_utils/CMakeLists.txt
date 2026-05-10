cmake_minimum_required(VERSION 3.8)
project(aic_training_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/xacro_expander.py
  scripts/proximity_data_generator.py
  scripts/training_frame_sink.py
  scripts/bin_to_webp_converter.py
  scripts/port_detector_c3.py
  scripts/port_detector_c3_occlusion.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
