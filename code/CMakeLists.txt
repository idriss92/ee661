#name for homework
project(ee661)
cmake_minimum_required(VERSION 2.6)

set(LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(EXEC_HW6 hw6)
set(EXEC_HW8 hw8)
set(EXEC ${EXEC_HW6} ${EXEC_HW8})
file(GLOB SRC_HW6 "${CMAKE_CURRENT_SOURCE_DIR}/src/hw6/*.cpp")
file(GLOB SRC_HW8 "${CMAKE_CURRENT_SOURCE_DIR}/src/hw8/*.cpp")

include_directories("${INCLUDE}")
link_directories ("${LIB}")
add_executable(${EXEC_HW6} ${SRC_HW6})
add_executable(${EXEC_HW8} ${SRC_HW8})

if (MSVC)
SET_TARGET_PROPERTIES(${EXEC} PROPERTIES PREFIX "../")
target_link_libraries(${EXEC_HW6} highgui200 cv200 cxcore200 lapack_win32_MDd blas_win32_MDd levmar ${CMAKE_REQUIRED_LIBRARIES})
target_link_libraries(${EXEC_HW8} highgui200 cv200 cxcore200 ${CMAKE_REQUIRED_LIBRARIES})
else ()
message( FATAL_ERROR "Only support building with Visual Studio platform.")
endif ()


# additional compiler flags
# add_definitions(-Wall -Wextra -pedantic -Werror -std=c99 -D_BSD_SOURCE)
