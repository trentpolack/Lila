lila_module(lila_editor UI
    CodeEditor.cpp CodeEditor.h
    CompletionPopup.cpp CompletionPopup.h
    DocFormat.cpp DocFormat.h
    DocPopup.cpp DocPopup.h
    EditorPalette.cpp EditorPalette.h
    EditorTypes.h
    FindBar.cpp FindBar.h
    Interpreter.cpp Interpreter.h
    LspClient.cpp LspClient.h
    OutputView.cpp OutputView.h
    PythonHighlighter.cpp PythonHighlighter.h
    Format.cpp Format.h
    Ruff.cpp Ruff.h
    ScriptEditor.cpp ScriptEditor.h
    ScriptStorage.cpp ScriptStorage.h
    TextUtil.cpp TextUtil.h
    Tools.cpp Tools.h
    VimBinding.cpp VimBinding.h
    VimEngine.cpp VimEngine.h)
set_target_properties(lila_editor PROPERTIES AUTOMOC ON)
target_link_libraries(lila_editor PUBLIC lila_python lila_layouts Qt6::Widgets nlohmann_json::nlohmann_json PRIVATE Python::Python)

target_compile_definitions(lila_editor PRIVATE LILA_PYTHON_HOME="${LILA_PYTHON_HOME}")
