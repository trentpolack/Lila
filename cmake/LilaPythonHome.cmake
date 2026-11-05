# The executable may be in bin/, a framework, or a venv. PyConfig.home needs the installation prefix.
execute_process(COMMAND "${Python_EXECUTABLE}" -I -c
    "import pathlib, sys, sysconfig; print(pathlib.Path(sys.base_prefix).as_posix()); print(pathlib.Path(sysconfig.get_path('stdlib')).relative_to(sys.base_prefix).as_posix())"
    OUTPUT_VARIABLE _lila_python_paths OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _lila_python_result ERROR_VARIABLE _lila_python_error)
if(NOT _lila_python_result EQUAL 0)
    message(FATAL_ERROR "Cannot locate Python's runtime: ${_lila_python_error}")
endif()
string(REPLACE "\r\n" "\n" _lila_python_paths "${_lila_python_paths}")
string(REPLACE "\n" ";" _lila_python_paths "${_lila_python_paths}")
list(GET _lila_python_paths 0 LILA_PYTHON_HOME)
list(GET _lila_python_paths 1 LILA_PYTHON_STDLIB_SUBDIR)
