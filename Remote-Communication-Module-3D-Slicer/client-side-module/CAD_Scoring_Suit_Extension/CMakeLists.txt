cmake_minimum_required(VERSION 3.13.4)

project(CAD_Scoring_Suit_Extension)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/CAD_Scoring_Suit_Extension")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Abdullah Alrefaey (Biomedical Engineering Department, Cairo University.), Ahmed Abdelmageed (Biomedical Engineering Department, Cairo University.), Ahmed Mahdy (Biomedical Engineering Department, Cairo University.), Ahmed Salah (Biomedical Engineering Department, Cairo University.)")
set(EXTENSION_DESCRIPTION "This extension is developed by a group of students in SBME Class 2021 for the Graduation Project.

This extension for Automatic Calcium segmentation and measurement in CT images.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/CAD_Scoring_Suit_Extension.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/CAD_Scoring_Suit_Extension/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CaScoreModule)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
