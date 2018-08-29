# find png++, and also libpng

if(png++_FIND_QUIETLY)
  set(_FIND_PNG_ARG QUIET)
endif()

find_package(PNG ${_FIND_PNG_ARG} REQUIRED)

find_path(png++_INCLUDE_DIR
  NAMES  png++/color.hpp  png++/config.hpp  png++/consumer.hpp  png++/convert_color_space.hpp
  png++/end_info.hpp  png++/error.hpp  png++/ga_pixel.hpp  png++/generator.hpp
  png++/gray_pixel.hpp  png++/image.hpp  png++/image_info.hpp  png++/index_pixel.hpp
  png++/info.hpp  png++/info_base.hpp  png++/io_base.hpp  png++/packed_pixel.hpp
  png++/palette.hpp  png++/pixel_buffer.hpp  png++/pixel_traits.hpp  png++/png.hpp
  png++/reader.hpp  png++/require_color_space.hpp  png++/rgb_pixel.hpp  png++/rgba_pixel.hpp
  png++/streaming_base.hpp  png++/tRNS.hpp  png++/types.hpp  png++/writer.hpp)

set(png++_INCLUDE_DIRS ${png++_INCLUDE_DIR} ${PNG_INCLUDE_DIRS})
# png++ is a header-only program, so no libraries of its own.
set(png++_LIBRARIES ${PNG_LIBRARIES})

find_package_handle_standard_args(png++ DEFAULT_MSG png++_INCLUDE_DIR)

mark_as_advanced(png++_LIBRARY png++_INCLUDE_DIR)
