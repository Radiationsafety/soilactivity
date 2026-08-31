#----------------------------------------------------------------------------
# CMakeLists.txt для примера b1_soil (на основе basic/B1, Geant4 10.7+ / 11.x)
#
#   cmake -B build -S .
#   cmake --build build -j
#----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16...3.27)
project(b1soil)

find_package(Geant4 REQUIRED ui_all vis_all)

include(${Geant4_USE_FILE})

#----------------------------------------------------------------------------
# Исходники
#----------------------------------------------------------------------------
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

add_executable(b1soil b1soil.cc ${sources} ${headers})
target_link_libraries(b1soil ${Geant4_LIBRARIES})
target_include_directories(b1soil PRIVATE ${PROJECT_SOURCE_DIR}/include)

#----------------------------------------------------------------------------
# Копируем макросы и python-скрипты в каталог сборки для удобного запуска
#----------------------------------------------------------------------------
set(B1SOIL_MACROS
    macros/vis.mac
    macros/model_cells.mac
    macros/sensitivity.mac
    macros/sensitivity_det.mac
    macros/sensitivity_src.mac
    macros/sensitivity_fast.mac
    macros/model_source.mac
    macros/model_source_det.mac
    macros/model_source_fast.mac)
set(B1SOIL_PYTHON
    python/b1soil_io.py
    python/make_model_macro.py
    python/build_sensitivity.py
    python/reconstruct.py
    python/visualize.py)

foreach(_script ${B1SOIL_MACROS} ${B1SOIL_PYTHON})
  configure_file(${PROJECT_SOURCE_DIR}/${_script}
                 ${PROJECT_BINARY_DIR}/${_script} COPYONLY)
endforeach()

install(TARGETS b1soil DESTINATION bin)
