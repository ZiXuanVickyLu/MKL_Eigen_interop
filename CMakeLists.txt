cmake_minimum_required(VERSION 3.26)
project(mkl_eigen_interop)
set(CMAKE_CXX_STANDARD 17)
set(MKL_ARCH intel64)
set(MKL_THREADING intel_thread)
set(MKL_INTERFACE lp64) # Eigen MKL interface only support lp64, see https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html

find_package(MKL CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

#when your enviorment has no eigen, you can use the following subdirectory to add eigen
#add_subdirectory(eigen)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    # for passing correct __cplusplus macro to the compiler
endif()

set(CL_INCLUDE "$ENV{CMPLR_ROOT}/include" "$ENV{CMPLR_ROOT}/include/sycl/")
include_directories(${CL_INCLUDE})
message(STATUS "STATUS: Use intel oneAPI to enable multi-thread")
set(BENCHMARK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/benchmark")

set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")

message(STATUS
	"STATUS: Set Asset to " ${DATA_DIR})
add_compile_definitions(DATA_PATH="${DATA_DIR}")


add_executable(benchmark_1 "${BENCHMARK_DIR}/main.cpp")
add_executable(benchmark_2 "${BENCHMARK_DIR}/from_matrix.cpp")
add_executable(benchmark_3 "${BENCHMARK_DIR}/from_matrix_1000x1000.cpp")
target_include_directories(benchmark_1 PUBLIC ${CL_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(benchmark_1 PUBLIC MKL::MKL TBB::tbb Eigen3::Eigen)

target_include_directories(benchmark_2 PUBLIC ${CL_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(benchmark_2 PUBLIC MKL::MKL TBB::tbb Eigen3::Eigen)

target_include_directories(benchmark_3 PUBLIC ${CL_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(benchmark_3 PUBLIC MKL::MKL TBB::tbb Eigen3::Eigen)