# HLSL -> SPIR-V at build time, embedded into the library.
#
# caelix_add_shaders(<target> <out_var> SOURCES <file.hlsl>...)
# caelix_compile_shaders(<out_var> SOURCES <file.hlsl>...)
#
# Every source that defines `main(` is a compute kernel (or carries the marker CAELIX_KERNEL_VARIANT: a
# kernel variant that defines macros and includes another kernel's source): DXC compiles it to
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
#
# caelix_compile_shaders is the compile part alone (same flags, depfiles and output folder; no
# embedding): <out_var> receives the .spv files, which a test loads from
# ${CMAKE_BINARY_DIR}/shaders at run time (tests/light_tree_test.cpp). Kernel names are unique
# across every call of both functions in the build (they share the output folder).

find_program(CAELIX_DXC NAMES dxc
  HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
  NO_DEFAULT_PATH)
find_program(CAELIX_DXC NAMES dxc REQUIRED)

set(CAELIX_EMBED_SHADERS_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/EmbedShaders.cmake)

# A kernel name must be unique in the whole build: the .spv files share one folder.
function(_caelix_claim_kernel_name name)
  get_property(claimed GLOBAL PROPERTY CAELIX_SHADER_KERNEL_NAMES)
  if(name IN_LIST claimed)
    message(FATAL_ERROR "caelix shaders: two kernels are named ${name}; kernel names are flat "
                        "(no folder) and shared by every caelix_compile_shaders / caelix_add_shaders "
                        "call, so every kernel file name must be unique")
  endif()
  set_property(GLOBAL APPEND PROPERTY CAELIX_SHADER_KERNEL_NAMES ${name})
endfunction()

function(caelix_compile_shaders out_var)
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
  # CAELIX_SHADER_DEBUG_INFO: embed the HLSL source and line markers (DXC -Zi: OpSource + OpLine), so
  # Nsight maps SASS, samples and live registers to source lines. -O3 stays: the register count is
  # the same as without it (measured on caelix_path_trace, 2026-10-22). The richer formats
  # (-fspv-debug=vulkan, vulkan-with-source) crash DXC 1.8 at -O3, and so does -Zi on the depfile
  # command (-MF): the flag goes on the compile command only.
  set(dxc_compile_flags)
  if(CAELIX_SHADER_DEBUG_INFO)
    list(APPEND dxc_compile_flags -Zi)
  endif()

  set(spv_files)
  foreach(source IN LISTS ARG_SOURCES)
    get_filename_component(source ${source} ABSOLUTE)
    file(READ ${source} text)
    if(NOT text MATCHES "void[ \t\r\n]+main[ \t\r\n]*\\(" AND NOT text MATCHES "CAELIX_KERNEL_VARIANT")
      continue()  # an include file
    endif()
    get_filename_component(name ${source} NAME_WE)
    _caelix_claim_kernel_name(${name})
    set(spv ${out_dir}/${name}.spv)
    set(depfile ${out_dir}/${name}.d)
    # DXC 1.8 does not compile when -MF is given: the first command writes the
    # depfile only, the second one compiles.
    add_custom_command(
      OUTPUT ${spv}
      COMMAND ${CAELIX_DXC} ${dxc_flags} -Fo ${spv} -MF ${depfile} ${source}
      COMMAND ${CAELIX_DXC} ${dxc_flags} ${dxc_compile_flags} -Fo ${spv} ${source}
      DEPENDS ${source}
      DEPFILE ${depfile}
      COMMENT "DXC ${name}.hlsl -> ${name}.spv"
      VERBATIM)
    list(APPEND spv_files ${spv})
  endforeach()
  set(${out_var} ${spv_files} PARENT_SCOPE)
endfunction()

function(caelix_add_shaders target out_var)
  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
  caelix_compile_shaders(spv_files SOURCES ${ARG_SOURCES})
  # Every .spv is embedded under its file name (the kernel name).
  set(embed_list)
  foreach(spv IN LISTS spv_files)
    get_filename_component(name ${spv} NAME_WE)
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
