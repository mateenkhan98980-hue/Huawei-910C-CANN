# ============================================================================
# CMake Build Configuration - Ascend CANN 8.0 Wrapper Stack
# ============================================================================
cmake_minimum_required(VERSION 3.10)
project(AscendCANNWrapper VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

option(BUILD_CUDA_COMPAT "Build CUDA-compatible PyTorch wrapper library" ON)
option(BUILD_TESTS "Build validation test executables" ON)
option(ASCEND_QUIET_LOGS "Compile with reduced runtime logging" OFF)

if(ASCEND_QUIET_LOGS)
    add_compile_definitions(ASCEND_QUIET_LOGS=1)
endif()

# ============================================================================
# CANN / DRIVER PATHS
# ============================================================================
if(NOT DEFINED ASCEND_HOME)
    if(DEFINED ENV{ASCEND_HOME})
        set(ASCEND_HOME $ENV{ASCEND_HOME})
    else()
        set(ASCEND_HOME "/usr/local/Ascend")
    endif()
endif()

if(NOT DEFINED ASCEND_DRIVER)
    if(DEFINED ENV{ASCEND_DRIVER})
        set(ASCEND_DRIVER $ENV{ASCEND_DRIVER})
    else()
        set(ASCEND_DRIVER "/usr/local/Ascend/driver")
    endif()
endif()

message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")
message(STATUS "ASCEND_DRIVER: ${ASCEND_DRIVER}")

find_path(CANN_INCLUDE_DIR acl/acl.h
    PATHS ${ASCEND_HOME}/ascend-toolkit/latest/include ${ASCEND_HOME}/include
    NO_DEFAULT_PATH)
if(NOT CANN_INCLUDE_DIR)
    message(FATAL_ERROR "CANN headers not found. Set ASCEND_HOME.")
endif()

find_path(CANN_LIB_DIR libacl.so
    PATHS ${ASCEND_HOME}/ascend-toolkit/latest/lib64 ${ASCEND_HOME}/lib64
    NO_DEFAULT_PATH)
if(NOT CANN_LIB_DIR)
    message(FATAL_ERROR "CANN libraries not found. Set ASCEND_HOME.")
endif()

find_path(HCCL_INCLUDE_DIR hccl/hccl.h
    PATHS ${ASCEND_HOME}/ascend-toolkit/latest/include ${ASCEND_HOME}/include
    NO_DEFAULT_PATH)

find_library(ASCEND_HAL_LIB ascend_hal PATHS ${ASCEND_DRIVER}/lib64 NO_DEFAULT_PATH)
if(NOT ASCEND_HAL_LIB)
    message(FATAL_ERROR "libascend_hal.so not found. Set ASCEND_DRIVER.")
endif()

macro(find_and_attach_lib LIB_NAME)
    find_library(${LIB_NAME}_LIB ${LIB_NAME}
        PATHS ${CANN_LIB_DIR} ${ASCEND_DRIVER}/lib64 NO_DEFAULT_PATH)
    if(${LIB_NAME}_LIB)
        list(APPEND CANN_LIBRARIES ${${LIB_NAME}_LIB})
    else()
        message(WARNING "Optional library ${LIB_NAME} not found")
    endif()
endmacro()

set(CANN_LIBRARIES)
find_and_attach_lib(acl)
find_and_attach_lib(aclnn)
find_and_attach_lib(acl_op_compiler)
find_and_attach_lib(acl_pbmodel)
find_and_attach_lib(opapi)
find_and_attach_lib(runtime)
find_and_attach_lib(hccl)
list(APPEND CANN_LIBRARIES ${ASCEND_HAL_LIB})

include_directories(${CANN_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(HCCL_INCLUDE_DIR)
    include_directories(${HCCL_INCLUDE_DIR})
endif()

function(attach_cann target_name)
    target_link_libraries(${target_name} PRIVATE ${CANN_LIBRARIES})
    target_compile_options(${target_name} PRIVATE -fPIC)
endfunction()

# ============================================================================
# LIBRARIES
# ============================================================================

add_library(ascendcl SHARED
    src/libascendcl.cpp src/acl_runtime.cpp src/acl_memory.cpp
    src/acl_device.cpp src/acl_stream.cpp)
attach_cann(ascendcl)
set_target_properties(ascendcl PROPERTIES OUTPUT_NAME ascendcl VERSION 1.0.0 SOVERSION 1)

add_library(aclnn_ops SHARED src/libaclnn_ops.cpp src/gemm_kernel.cpp)
target_link_libraries(aclnn_ops PUBLIC ascendcl)
attach_cann(aclnn_ops)
set_target_properties(aclnn_ops PROPERTIES OUTPUT_NAME aclnn_ops VERSION 1.0.0 SOVERSION 1)

add_library(acl_op_compiler SHARED src/libacl_op_compiler.cpp src/acl_graph.cpp src/acl_compiler.cpp)
target_link_libraries(acl_op_compiler PUBLIC aclnn_ops ascendcl)
attach_cann(acl_op_compiler)
set_target_properties(acl_op_compiler PROPERTIES OUTPUT_NAME acl_op_compiler VERSION 1.0.0 SOVERSION 1)

if(BUILD_CUDA_COMPAT)
    add_library(ascendcl_cuda_compat SHARED
        src/ascendcl_cuda_compat.cpp src/cuda_compat_builtin_ops.cpp)
    target_link_libraries(ascendcl_cuda_compat PUBLIC ascendcl)
    attach_cann(ascendcl_cuda_compat)
    set_target_properties(ascendcl_cuda_compat PROPERTIES OUTPUT_NAME ascendcl_cuda_compat VERSION 1.0.0 SOVERSION 1)

    # Exact CUDA drop-in library (includes all new features)
    add_library(cuda_compat SHARED
        src/cuda_dropin.cpp
        src/ascendcl_kernel.cpp
        src/ascendcl_memory.cpp
        src/cuda_graph.cpp
        src/cublas_stub.cpp
        src/profiler.cpp)
    target_link_libraries(cuda_compat PUBLIC ascendcl_cuda_compat ascendcl)
    attach_cann(cuda_compat)
    set_target_properties(cuda_compat PROPERTIES OUTPUT_NAME cuda_compat VERSION 1.0.0 SOVERSION 1)
endif()

# ============================================================================
# TESTS
# ============================================================================
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_gemm tests/test_gemm.cpp)
    target_link_libraries(test_gemm PRIVATE acl_op_compiler aclnn_ops ascendcl)
    attach_cann(test_gemm)
    add_test(NAME gemm_suite COMMAND test_gemm)

    if(BUILD_CUDA_COMPAT)
        add_executable(test_cuda_compat tests/test_cuda_compat.cpp)
        target_link_libraries(test_cuda_compat PRIVATE ascendcl_cuda_compat cuda_compat)
        attach_cann(test_cuda_compat)
        add_test(NAME cuda_compat_suite COMMAND test_cuda_compat)
    endif()
endif()

# ============================================================================
# INSTALL
# ============================================================================
install(TARGETS ascendcl aclnn_ops acl_op_compiler LIBRARY DESTINATION lib64)
if(BUILD_CUDA_COMPAT)
    install(TARGETS ascendcl_cuda_compat cuda_compat LIBRARY DESTINATION lib64)
    install(FILES
        include/ascendcl_cuda_compat.h
        include/cuda_compat_extra.h
        DESTINATION include)
endif()
install(FILES
    include/ascendcl_wrapper.h include/aclnn_wrapper.h include/acl_compiler.h
    include/gemm_kernel.h include/ascend_util.h include/ascendcl_kernel.h
    include/ascendcl_memory.h DESTINATION include)
install(FILES docs/ARCHITECTURE.md DESTINATION share/doc/ascend-cann-wrapper)
if(BUILD_TESTS)
    install(TARGETS test_gemm RUNTIME DESTINATION bin)
    if(BUILD_CUDA_COMPAT)
        install(TARGETS test_cuda_compat RUNTIME DESTINATION bin)
    endif()
endif()

message(STATUS "========== BUILD SUMMARY ==========")
message(STATUS "CANN Include: ${CANN_INCLUDE_DIR}")
message(STATUS "CANN Libs   : ${CANN_LIB_DIR}")
message(STATUS "Ascend HAL  : ${ASCEND_HAL_LIB}")
message(STATUS "CUDA compat : ${BUILD_CUDA_COMPAT}")
message(STATUS "Tests       : ${BUILD_TESTS}")