# HLSL -> SPIR-V at build time, embedded into the library.
#
# caelix_add_shaders(<target> <out_var> SOURCES <file.hlsl>...)
#
# Every source that defines `main(` is a compute kernel: DXC compiles it to
# <build>/shaders/<name>.spv, where <name> is the file name without extension and
# without its folder (shaders/scene/caelix_finalize.hlsl -> caelix_finalize). The
# names are the keys of LoadShader and of CAELIX_SHADER_DIR overrides, so they stay
# flat and must be unique across folders. The other sources are include files,
# included by their path relative to shaders/ ("common/caelix_platform.hlsl").
# DXC reports which of them a kernel includes in a depfile, so editing a header
# rebuilds exactly the kernels that include it. Then cmake/EmbedShaders.cmake
# turns every .spv into a uint32_t array in a generated .cpp that is added to
# <target> (lookup: lib/src/rhi/shaders.h). <out_var> receives the list of .spv files.
#
# The .spv files do not depend on the C++ configuration: Debug and Release use
# the same output paths, so a shader compiles once.

find_program(CAELIX_DXC NAMES dxc
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
  NO_DEFAULT_PATH)
find_program(CAELIX_DXC NAMES dxc REQUIRED)

set(CAELIX_EMBED_SHADERS_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/EmbedShaders.cmake)

function(caelix_add_shaders target out_var)
  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
  set(out_dir ${CMAKE_BINARY_DIR}/shaders)
  file(MAKE_DIRECTORY ${out_dir})
  set(dxc_flags
    -T cs_6_6 -E main -spirv -fspv-target-env=vulkan1.3 -HV 2021 -O3
    # Shader warnings fail the build, like /W4 warnings in review.
    -WX
    # Include root of our shaders: "common/...", "scene/...", "render/...".
    -I ${PROJECT_SOURCE_DIR}/shaders
    # NRD.hlsli + NRDConfig.hlsli (cmake/Dependencies.cmake). DXC predefines
    # __hlsl_dx_compiler, which NRD.hlsli uses to select its DXC branch.
    -I ${CAELIX_NRD_SHADER_INCLUDE_DIR})

  set(spv_files)
  set(embed_list)
  set(names)
  foreach(source IN LISTS ARG_SOURCES)
    get_filename_component(source ${source} ABSOLUTE)
    file(READ ${source} text)
    if(NOT text MATCHES "void[ \t\r\n]+main[ \t\r\n]*\\(")
      continue()  # an include file
    endif()
    get_filename_component(name ${source} NAME_WE)
    if(name IN_LIST names)
      message(FATAL_ERROR "caelix_add_shaders: two kernels are named ${name}; kernel names "
                          "are flat (no folder), so every kernel file name must be unique")
    endif()
    list(APPEND names ${name})
    set(spv ${out_dir}/${name}.spv)
    set(depfile ${out_dir}/${name}.d)
    # DXC 1.8 does not compile when -MF is given: the first command writes the
    # depfile only, the second one compiles.
    add_custom_command(
      OUTPUT ${spv}
      COMMAND ${CAELIX_DXC} ${dxc_flags} -Fo ${spv} -MF ${depfile} ${source}
      COMMAND ${CAELIX_DXC} ${dxc_flags} -Fo ${spv} ${source}
      DEPENDS ${source}
      DEPFILE ${depfile}
      COMMENT "DXC ${name}.hlsl -> ${name}.spv"
      VERBATIM)
    list(APPEND spv_files ${spv})
    list(APPEND embed_list "${name}=${spv}")
  endforeach()

  # '|' separates the entries: a ';' would split the -D argument.
  list(JOIN embed_list "|" embed_arg)
  set(embedded_cpp ${CMAKE_CURRENT_BINARY_DIR}/caelix_shaders_embedded.cpp)
  add_custom_command(
    OUTPUT ${embedded_cpp}
    COMMAND ${CMAKE_COMMAND} "-DCAELIX_SHADERS=${embed_arg}" "-DCAELIX_OUTPUT=${embedded_cpp}"
            -P ${CAELIX_EMBED_SHADERS_SCRIPT}
    DEPENDS ${spv_files} ${CAELIX_EMBED_SHADERS_SCRIPT}
    COMMENT "Embedding SPIR-V into caelix_shaders_embedded.cpp"
    VERBATIM)
  target_sources(${target} PRIVATE ${embedded_cpp} ${ARG_SOURCES})
  set_source_files_properties(${ARG_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)
  set(${out_var} ${spv_files} PARENT_SCOPE)
endfunction()
