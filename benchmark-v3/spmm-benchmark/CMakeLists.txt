

file(GLOB_RECURSE CURRENT_HEADERS  spmm_utils/*.h)
# file(GLOB CURRENT_SOURCES  src/*.cpp src/*.cu)
file(GLOB CURRENT_SOURCES  src/mGCNKernel.cu)

add_library(MGCN2 STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(MGCN2 "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

install(TARGETS MGCN2 ARCHIVE DESTINATION "benchmark/GCNv2-benchmark/lib")
install(FILES ${CURRENT_HEADERS} DESTINATION "benchmark/GCNv2-benchmark/include")
set_property(TARGET MGCN2 PROPERTY CXX_STANDARD 17)