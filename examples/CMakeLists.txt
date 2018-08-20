cmake_minimum_required (VERSION 3.3)
project (llama)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

ADD_SUBDIRECTORY("simpletest")
ADD_SUBDIRECTORY("nbody")
ADD_SUBDIRECTORY("uidtest")
ADD_SUBDIRECTORY("treemaptest")
ADD_SUBDIRECTORY("vectoradd")
ADD_SUBDIRECTORY("virtualviewtest")
