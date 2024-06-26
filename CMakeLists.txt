cmake_minimum_required(VERSION 3.0)
project(pfc-2D)

set(CMAKE_CXX_STANDARD 20)

# specify the FFTW3 library path
set(FFTW_DIR ../extern/fftw3)
set(FPNG_DIR ../extern/fpng/src)

# find the FFTW3 library
find_library(FFTW_LIB NAMES libfftw3-3 PATHS ${FFTW_DIR} NO_DEFAULT_PATH)
# find_library(FFTW_LIB libfftw3-3 ${FFTW_DIR})

# specify the directory where fftw3.h can be found
include_directories(${FFTW_DIR} ${FPNG_DIR})

add_executable(pfc-2D
    main.cpp
    ${FPNG_DIR}/fpng.h
    ${FPNG_DIR}/fpng.cpp)

# link the fftw3 library
target_link_libraries(pfc-2D ${FFTW_LIB})

# On Windows, copy the fftw3 dll to the output directory [not working, copied manually]
# if(WIN32)
#     add_custom_command(TARGET pfc-2D POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         "${FFTW_DIR}/libfftw3-3.dll"
#         $<TARGET_FILE_DIR:pfc-2D>)
# endif()