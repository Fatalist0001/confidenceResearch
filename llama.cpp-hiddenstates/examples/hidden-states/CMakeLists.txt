set(TARGET llama-hidden-states)
add_executable(${TARGET} hidden-states.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# diagnostic: dump the rendered prompt + token ids (vocab only, no weights)
# optional: the Kaggle kernel uploads only the three files it needs, so skip it there
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/prompt_dump.cpp")
    set(TARGET_DUMP hs-prompt-dump)
    add_executable(${TARGET_DUMP} prompt_dump.cpp)
    target_link_libraries(${TARGET_DUMP} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET_DUMP} PRIVATE cxx_std_17)
endif()
