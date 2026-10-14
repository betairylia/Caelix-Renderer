# cmake -DDEST=<dir> -DFILES=<a;b;...> -P CopyPlugin.cmake
# Copies every file of FILES that exists (a configuration without debug info has no .pdb).
file(MAKE_DIRECTORY "${DEST}")
foreach(file IN LISTS FILES)
  if(EXISTS "${file}")
    get_filename_component(file_name "${file}" NAME)
    file(COPY_FILE "${file}" "${DEST}/${file_name}" ONLY_IF_DIFFERENT RESULT copy_result)
    if(copy_result)
      message(FATAL_ERROR "copying ${file} to ${DEST} failed: ${copy_result} (is the Unity Editor running and holding the DLL?)")
    endif()
    message(STATUS "up to date: ${DEST}/${file_name}")
  endif()
endforeach()
