cmake_minimum_required(VERSION 3.5)
project(lab2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external)

find_package(OpenGL REQUIRED)

# --- Target: lab2_building ---
add_executable(lab2_building
    lab2/lab2_building.cpp
    lab2/render/shader.cpp
)
target_include_directories(lab2_building PRIVATE
    "${PROJECT_SOURCE_DIR}/lab2"
    "${PROJECT_SOURCE_DIR}/external/glfw-3.1.2/include"
    "${PROJECT_SOURCE_DIR}/external/glm-0.9.7.1"
    "${PROJECT_SOURCE_DIR}/external"
)
target_link_libraries(lab2_building PRIVATE
    glad
    glfw
    OpenGL::GL
)

# --- Target: lab2_skybox ---
add_executable(lab2_skybox
    lab2/lab2_skybox.cpp
    lab2/render/shader.cpp
)
target_include_directories(lab2_skybox PRIVATE
    "${PROJECT_SOURCE_DIR}/lab2"
    "${PROJECT_SOURCE_DIR}/external/glfw-3.1.2/include"
    "${PROJECT_SOURCE_DIR}/external/glm-0.9.7.1"
    "${PROJECT_SOURCE_DIR}/external"
)
target_link_libraries(lab2_skybox PRIVATE
    glad
    glfw
    OpenGL::GL
)
add_executable(final
  lab2/final.cpp
  lab2/render/shader.cpp
)

target_include_directories(final PRIVATE
  "${PROJECT_SOURCE_DIR}/lab2"
  "${PROJECT_SOURCE_DIR}/external"
  "${PROJECT_SOURCE_DIR}/external/glfw-3.1.2/include"
  "${PROJECT_SOURCE_DIR}/external/glm-0.9.7.1"
  "${CMAKE_SOURCE_DIR}/external/tinygltf-2.9.3"
)

target_link_libraries(final PRIVATE
  glad
  glfw
  OpenGL::GL
)

