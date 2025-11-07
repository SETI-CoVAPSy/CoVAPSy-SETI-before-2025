cmake_minimum_required(VERSION 2.8)

project(seuillage LANGUAGES CXX CUDA)
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
#set(CMAKE_C_COMPILER "gcc-8")
#set(CMAKE_CXX_COMPILER "g++-8")

#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

list(APPEND HEADER_REP ${ROOT_DIR}/ ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "/usr/local/cuda/samples/common/inc/" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../samples/common/inc")

file(GLOB CPP_FILES ${ROOT_DIR}/*.cpp)
file(GLOB CU_FILES ${ROOT_DIR}/*.cu)
file(GLOB_RECURSE HEADER_FILES ${ROOT_DIR}/*.cuh ${ROOT_DIR}/*.h)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")


set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS "-lineinfo;-I\"${ROOT_DIR}/inc\" -I\"${ROOT_DIR}/\" -I\"/usr/local/cuda/samples/common/inc/\"")


add_executable(seuillage_CUDA ${CPP_FILES} ${CU_FILES} ${HEADER_FILES})
set_target_properties(seuillage_CUDA PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(seuillage_CUDA PUBLIC ${ROOT_DIR}/inc/ /usr/local/cuda/samples/common/inc/)









