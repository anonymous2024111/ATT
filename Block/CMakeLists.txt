

file(GLOB_RECURSE CURRENT_HEADERS  *.h)
# file(GLOB CURRENT_SOURCES  *.cpp src/*.cu)
file(GLOB CURRENT_SOURCES  example1.cpp)

add_library(MBlock STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(MBlock "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)
# 添加 -fPIC 编译选项
set_property(TARGET MBlock PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS MBlock ARCHIVE DESTINATION "Block/lib")
install(FILES ${CURRENT_HEADERS} DESTINATION "Block/include")
set_property(TARGET MBlock PROPERTY CXX_STANDARD 17)