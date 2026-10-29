cmake_minimum_required(VERSION 3.16)
project(hnsw_quant_eval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

file(GLOB HQE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(hqe STATIC ${HQE_SOURCES})
target_include_directories(hqe PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hqe PUBLIC Threads::Threads)
if(NOT MSVC)
  target_compile_options(hqe PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Every file in tools/ is a command line program.
file(GLOB HQE_TOOLS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tools/*.cpp)
foreach(tool_src ${HQE_TOOLS})
  get_filename_component(tool_name ${tool_src} NAME_WE)
  add_executable(${tool_name} ${tool_src})
  target_link_libraries(${tool_name} PRIVATE hqe)
endforeach()

# Every tests/test_*.cpp is a separate test program.
enable_testing()
file(GLOB HQE_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
foreach(test_src ${HQE_TESTS})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE hqe)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
