# - Config file for the llama package
# It defines the following variables
#  LLAMA_INCLUDE_DIR      - include directory for LLAMA
#  LLAMA_DEFINITIONS      - necessary definitions
#  LLAMA_FOUND            - whether LLAMA was found and is useable

###############################################################################
# LLAMA
###############################################################################
cmake_minimum_required (VERSION 3.3.0)

set(llama_INCLUDE_DIR ${llama_INCLUDE_DIR} "${llama_DIR}/include")

################################################################################
# BOOST LIB
################################################################################
find_package(Boost 1.56.0 REQUIRED)
set(llama_INCLUDE_DIR ${llama_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
set(llama_DEFINITIONS ${llama_DEFINITIONS} -DBOOST_ALL_NO_LIB)

################################################################################
# Warning if C++11 is not activated
################################################################################
if (CMAKE_CXX_STANDARD EQUAL 98)
    message( FATAL_ERROR "At least C++ standard 11 must be enabled!" )
endif()

################################################################################
# Returning whether LLAMA could be found
################################################################################

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS("llama"
                                    REQUIRED_VARS
                                        llama_INCLUDE_DIR
                                        Boost_FOUND
                                )
