cmake_minimum_required(VERSION 3.13.4)

project(GammaIndex)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/GammaIndex")
set(EXTENSION_CATEGORY "Radiotherapy")
set(EXTENSION_CONTRIBUTORS "Alex Vergara (INSERM), Gan Quan (INSERM)")
set(EXTENSION_DESCRIPTION "This is an extension to calculate gamma matrix and gamma index from two dose images")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/GammaIndex.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/GammaIndex/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Gamma)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
