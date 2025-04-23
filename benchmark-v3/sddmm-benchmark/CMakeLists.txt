

file(GLOB_RECURSE CURRENT_HEADERS  sddmm_utils/*.h)
# file(GLOB CURRENT_SOURCES  src/*.cpp src/*.cu)
file(GLOB CURRENT_SOURCES  src/mGATKernel.cu)

add_library(MGAT2 STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(MGAT2 "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

install(TARGETS MGAT2 ARCHIVE DESTINATION "benchmark/GATv2-benchmark/lib")
install(FILES ${CURRENT_HEADERS} DESTINATION "benchmark/GATv2-benchmark/include")
set_property(TARGET MGAT2 PROPERTY CXX_STANDARD 17)