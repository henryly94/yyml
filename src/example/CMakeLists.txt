add_executable(
  dense_main
  dense_main.cc
)
target_include_directories(
  dense_main PUBLIC ${CMAKE_PROJECT_BASE_DIR}
)
target_link_directories(
  dense_main PUBLIC ${CMAKE_PROJECT_BASE_DIR}
)
target_link_libraries(
  dense_main PUBLIC 
  util::function  
)

add_executable(
  conv_main
  conv_main.cc
)
target_include_directories(
  conv_main PUBLIC ${CMAKE_PROJECT_BASE_DIR}
)
