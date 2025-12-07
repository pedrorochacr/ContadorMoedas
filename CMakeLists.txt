cmake_minimum_required(VERSION 3.10)
project(CoinCounter VERSION 1.0.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Main executable
add_executable(coin_counter coin_counter.cpp)
target_link_libraries(coin_counter ${OpenCV_LIBS})

# Print OpenCV info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Optional: Build tests
option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_executable(test_detection test_detection.cpp)
    target_link_libraries(test_detection ${OpenCV_LIBS})
    add_test(NAME DetectionTest COMMAND test_detection)
endif()

# Install targets
install(TARGETS coin_counter DESTINATION bin)
install(FILES coin_counter.h DESTINATION include)
