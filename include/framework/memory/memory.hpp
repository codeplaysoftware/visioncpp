// This file is part of VisionCpp, a lightweight C++ template library
// for computer vision and image processing.
//
// Copyright (C) 2016 Codeplay Software Limited. All Rights Reserved.
//
// Contact: visioncpp@codeplay.com
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// \file memory.hpp
/// this file contains a set of forward declarations and include headers
/// required for constructing and accessing memory on both host and device.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_HPP_

namespace visioncpp {
namespace internal {
/// \struct MemoryProperties
/// \brief this is used to detect the Properties of a memory such as element
/// category, channel type, and channel size for different types of input raw
/// data
template <typename T>
struct MemoryProperties;
///  Definition of \ref VisionMemory
template <bool MapAlloc, size_t ScalarType, size_t MemoryType, typename Sclr,
          size_t Width, size_t Height, typename ElementTp, size_t elements,
          size_t Sc, size_t Level>
struct VisionMemory;

/// \struct OutputMemory:
/// \brief OutputMemory is used to deduce the output type of each node in the
/// expression tree by using certain parameters from its child(ren).
/// template parameters:
/// \tparam ElementType: the Pixel type of each element of the output memory.
/// \tparam LeafType: the memory_type of the output memory { e.g.Buffer1D,
/// Buffer2D, Image, Host}
/// \tparam Cols: the column size of the memory
/// \tparam Rows: the row size of the memory
/// \tparam LVL : the level of the output memory in the expression tree.
template <typename ElementType, size_t LeafType, size_t Cols, size_t Rows,
          size_t LVL>
struct OutputMemory {
  using Type = VisionMemory<
      false, MemoryProperties<ElementType>::ElementCategory, LeafType,
      typename MemoryProperties<ElementType>::ChannelType, Cols, Rows,
      ElementType, MemoryProperties<ElementType>::ChannelSize, scope::Global,
      LVL>;
};

/// The definition can be found in \ref ConstMemory
template <typename T>
struct ConstMemory;

/// \struct ImageProperties
/// \brief this file is used to create the image properties required to create
/// opencl image for different types of pixel
template <typename T, typename Scalar>
struct ImageProperties;

/// \brief two category of element exist : basic which is the primary types and
/// struct which is user define types like F32C3, U8C3, ...
namespace element_category {
constexpr size_t Basic = 0;
constexpr size_t Struct = 1;
};

/// \struct SyclRange
/// \brief This is used to determine the range for creating a syclbuffer based on
/// the memory dimension
/// template parameters:
/// \tparam Dim : the memory dimension
template <size_t Dim>
struct SyclRange;

/// \brief specialisation of the SyclRange when the dimension is 2
template <>
struct SyclRange<2> {
  /// function get_range
  /// \brief return the sycl range<2>
  /// \param r: row size
  /// \param c: column size
  /// \return cl::sycl::range<2>
  static inline cl::sycl::range<2> get_range(size_t r, size_t c) {
    return cl::sycl::range<2>(r, c);
  }
};

/// \brief specialisation of the SyclRange when the dimension is 1
template <>
struct SyclRange<1> {
  /// function get_range
  /// \brief return the sycl range<1>
  /// \param r: row size
  /// \param c: column size
  /// \return cl::sycl::range<1>
  static inline cl::sycl::range<1> get_range(size_t r, size_t c) {
    return cl::sycl::range<1>(r * c);
  }
};

/// function get_range
/// \brief template deduction for SyclRange
/// template parameters:
/// \tparam Dim : the memory dimension
/// function parameters:
/// \param r: row size
/// \param c: column size
/// \return cl::sycl::range<Dim>
template <size_t Dim>
inline cl::sycl::range<Dim> get_range(size_t r, size_t c) {
  /// column major
  return SyclRange<Dim>::get_range(c, r);
  /// row major   /// \brief this will apply when sycl changes
  //  return SyclRange<Dim>::get_range(r, c);
}

/// \struct SyclScope
/// \brief determines the memory target on the device based on the
/// memory type and suggested target.
/// template parameters
/// \tparam LeafType determines the memory type
/// \tparam Sc: determines the suggested target
template <size_t LeafType, size_t Sc>
struct SyclScope;

template <size_t LeafType>
struct SyclScope<LeafType, scope::Global> {
  static constexpr cl::sycl::access::target scope =
      cl::sycl::access::target::global_buffer;
};

template <size_t LeafType>
struct SyclScope<LeafType, scope::Host_Buffer> {
  static constexpr cl::sycl::access::target scope =
      cl::sycl::access::target::host_buffer;
};

template <size_t LeafType>
struct SyclScope<LeafType, scope::Local> {
  static constexpr cl::sycl::access::target scope =
      cl::sycl::access::target::local;
};

template <size_t LeafType>
struct SyclScope<LeafType, scope::Constant> {
  static constexpr cl::sycl::access::target scope =
      cl::sycl::access::target::constant_buffer;
};

/// \brief specialisation of the SyclScope when the memory type is Image
template <>
struct SyclScope<memory_type::Image, scope::Global> {
  static constexpr cl::sycl::access::target scope =
      cl::sycl::access::target::image;
};

/// \struct ConvertToVisionScope
/// this struct is used to convert the sycl target to visioncpp target
/// \tparam Sc represent the sycl target
template <cl::sycl::access::target Sc>
struct ConvertToVisionScope;

/// \brief specialisation of \ref ConvertToVisionScope where the target is
/// global_buffer
template <>
struct ConvertToVisionScope<cl::sycl::access::target::global_buffer> {
  static constexpr size_t scope = scope::Global;
};

/// \brief specialisation of \ref ConvertToVisionScope where the target is
/// host_buffer
template <>
struct ConvertToVisionScope<cl::sycl::access::target::host_buffer> {
  static constexpr size_t scope = scope::Host_Buffer;
};

/// \brief specialisation of \ref ConvertToVisionScope where the target is
/// local
template <>
struct ConvertToVisionScope<cl::sycl::access::target::local> {
  static constexpr size_t scope = scope::Local;
};

/// \brief specialisation of \ref ConvertToVisionScope where the target is
/// constant_buffer
template <>
struct ConvertToVisionScope<cl::sycl::access::target::constant_buffer> {
  static constexpr size_t scope = scope::Constant;
};

/// \brief specialisation of \ref ConvertToVisionScope where the target is image
template <>
struct ConvertToVisionScope<cl::sycl::access::target::image> {
  static constexpr size_t scope = scope::Global;
};

/// \struct SyclAccessor
/// \brief This struct is used to create a sycl accessor  type based on access
/// mode; dimension and memory type.
/// template parameters:
/// \tparam LeafType: determines memory type
/// \tparam Dim :determine buffer dimension
/// \tparam AccMd: determines access mode
template <size_t LeafType, size_t Dim, cl::sycl::access::mode AccMd,
          typename ElementType, typename Scalar, size_t scope>
struct SyclAccessor {
  using Accessor = cl::sycl::accessor<ElementType, Dim, AccMd,
                                      SyclScope<LeafType, scope>::scope>;
  using access_type = ElementType;
};

/// \brief specialisation of the \ref SyclAccessor when the memory_type is Image
template <size_t Dim, cl::sycl::access::mode AccMd, typename ElementType,
          typename Scalar, size_t scope>
struct SyclAccessor<memory_type::Image, Dim, AccMd, ElementType, Scalar,
                    scope> {
  using Properties = ImageProperties<ElementType, Scalar>;
  using Accessor =
      cl::sycl::accessor<typename Properties::access_type, Dim, AccMd,
                         SyclScope<memory_type::Image, scope>::scope>;
  using access_type = typename Properties::access_type;
};

/// \brief specialisation of the \ref SyclAccessor when the memory_type is
/// constant variable
template <size_t Dim, cl::sycl::access::mode AccMd, typename ElementType,
          typename Scalar, size_t scope>
struct SyclAccessor<memory_type::Const, Dim, AccMd, ElementType, Scalar,
                    scope> {
  using access_type = ElementType;
  using Accessor = ConstMemory<ElementType>;
};

/// \struct SyclMem
/// \brief SyclMem is used to create VisionMemory data storage. It has been
/// specialised based on the memory type and input data
/// template parameters:
/// \tparam MapAlloc: determines whether or not a host pointer has been
/// allocated for this memory
/// \tparam LeafType: determines the memory type
/// \tparam Dim : determines the memory dimension
/// \tparam ElementType: determines the type of each element in the storage
template <bool MapAlloc, size_t LeafType, size_t Dim, typename ElementType>
struct SyclMem;
/// \brief specialisation of SyclMem when there is host memory allocated.
template <size_t LeafType, size_t Dim, typename ElementType>
struct SyclMem<true, LeafType, Dim, ElementType> {
  using Type =
      cl::sycl::buffer<ElementType, Dim, cl::sycl::map_allocator<ElementType>>;
};

/// \brief specialisation of SyclMem when there is no host memory allocated.
/// This is used to construct a device only buffer type.
template <size_t LeafType, size_t Dim, typename ElementType>
struct SyclMem<false, LeafType, Dim, ElementType> {
  using Type = cl::sycl::buffer<ElementType, Dim>;
};

/// \brief specialisation of SyclMem when the memory_type is Image
template <size_t Dim, typename ElementType>
struct SyclMem<true, memory_type::Image, Dim, ElementType> {
  using Type = cl::sycl::image<Dim, cl::sycl::map_allocator<uint8_t>>;
};

/// \brief specialisation of SyclMem when the memory_type is Constant
/// variable
template <size_t Dim, typename ElementType>
struct SyclMem<true, memory_type::Const, Dim, ElementType> {
  using Type = ElementType;
};

/// \brief specialisation of the SyclMem when the memory_type is Image and no
/// device pointer is allocated to the memory.
template <size_t Dim, typename ElementType>
struct SyclMem<false, memory_type::Image, Dim, ElementType> {
  using Type = cl::sycl::image<Dim>;
};

/// \struct MemDimension
/// \brief this is used to determine the dimension of the memory based on the
/// memory type
/// template parameters:
/// \tparam LeafType : the type of the memory
template <size_t LeafType>
struct MemDimension {
  static constexpr size_t Dim = 2;
};

/// \brief specialisation of the MemDimension when the memory type is Buffer1D
template <>
struct MemDimension<memory_type::Buffer1D> {
  static constexpr size_t Dim = 1;
};

/// \brief specialisation of the MemDimension when the memory type is Host
template <>
struct MemDimension<memory_type::Host> {
  static constexpr size_t Dim = 1;
};

/// \brief specialisation of the MemDimension when the memory type is constant
/// variable
template <>
struct MemDimension<memory_type::Const> {
  static constexpr size_t Dim = 1;
};

/// \struct CreateSyclBuffer
/// \brief This class is used to instantiate the sycl memory based on the memory
/// types.
/// template parameters:
/// \tparam LeafType: determines the memory type
/// \tparam ElemType : determines the type of the element in each memory
/// \tparam Scalar : determines the type of each channel of each element
/// \tparam VisionMem: represent the type of the memory created by using SyclMem
/// \tparam RNG : the sycl range type for creating memory
template <size_t LeafType, typename ElemType, typename Scalar,
          typename VisionMem, typename RNG>
struct CreateSyclBuffer {
  /// function create_buffer
  /// \brief This function is used to create a sycl buffer when the host memory
  /// allocated for synchronization.
  /// parameters:
  /// \param ptr : shared_ptr containing the VisionMem
  /// \param dt : the input pointer for creating buffer
  /// \param rng : the sycl range for creating buffer
  /// \return void
  static inline void create_buffer(std::shared_ptr<VisionMem> &ptr, Scalar *dt,
                                   RNG rng) {
    ptr = std::make_shared<VisionMem>(
        VisionMem((static_cast<ElemType *>(static_cast<void *>(dt))), rng));
  }

  /// function create_buffer
  /// \brief This function is used to create a device only buffer when there is
  /// no host pointer allocated for synchronization.
  /// parameters:
  /// \param ptr : shared_ptr containing the VisionMem
  /// \param rng : the sycl range for creating buffer
  /// \return void
  static inline void create_buffer(std::shared_ptr<VisionMem> &ptr, RNG rng) {
    ptr = std::make_shared<VisionMem>(VisionMem(rng));
    ptr.get()->set_final_data(nullptr);
  }
};

/// \brief specialisation of create CreateSyclBuffer when the memory type is
/// image
template <typename ElemType, typename Scalar, typename VisionMem, typename RNG>
struct CreateSyclBuffer<memory_type::Image, ElemType, Scalar, VisionMem, RNG> {
  using Properties = ImageProperties<ElemType, Scalar>;
  /// function create_buffer
  /// \brief This function is used to create a sycl buffer when the host memory is
  /// allocated for synchronization.
  /// parameters:
  /// \param ptr : shared_ptr containing the VisionMem
  /// \param dt : the input pointer for creating image
  /// \param rng : the sycl range for creating image
  /// \return void
  static inline void create_buffer(std::shared_ptr<VisionMem> &ptr, Scalar *dt,
                                   RNG rng) {
    ptr = std::make_shared<VisionMem>(VisionMem(dt, Properties::channel_order,
                                                Properties::channel_type, rng));
  }

  /// function create_buffer
  /// \brief This function is used to create a device only buffer when there is
  /// no host pointer allocated for synchronization.
  /// parameters:
  /// \param ptr : shared_ptr containing the VisionMem
  /// \param rng : the sycl range for creating image
  /// \return void
  static inline void create_buffer(std::shared_ptr<VisionMem> &ptr, RNG rng) {
    ptr = std::make_shared<VisionMem>(
        VisionMem(Properties::channel_order, Properties::channel_type, rng));
    ptr.get()->set_final_data(nullptr);
  }
};

/// \brief specialisation of create CreateSyclBuffer when the memory type is
/// constant variable
template <typename ElemType, typename Scalar, typename VisionMem, typename RNG>
struct CreateSyclBuffer<memory_type::Const, ElemType, Scalar, VisionMem, RNG> {
  /// function create_buffer
  /// \brief This function is used to create a sycl buffer when the host memory
  /// allocated for synchronization.
  /// parameters:
  /// \param ptr : shared_ptr containing the VisionMem
  /// \param dt : the input pointer for creating const memory
  /// \param rng : the sycl range for creating const memory
  /// \return void
  static inline void create_buffer(std::shared_ptr<VisionMem> &ptr,
                                   VisionMem dt, RNG rng) {
    ptr = std::make_shared<VisionMem>(VisionMem(dt));
  }
};

/// function create_sycl_buffer
/// \brief template deduction for CreateSyclBuffer struct when there is host
/// pointer
/// template parameters:
/// \tparam LeafType: determines the memory type
/// \tparam ElemType : determines the type of the element in each memory
/// \tparam Scalar : determines the type of each channel of each element
/// \tparam VisionMem: represent the type of the memory created by using SyclMem
/// \tparam RNG : the sycl range type for creating memory
/// function parameters
/// parameters:
/// \param ptr : shared_ptr containing the VisionMem
/// \param dt : the input pointer for creating buffer
/// \param rng : the sycl range for creating buffer
/// \return void
template <size_t LeafType, typename ElemType, typename Scalar,
          typename VisionMem, typename RNG>
inline void create_sycl_buffer(std::shared_ptr<VisionMem> &ptr, Scalar *dt,
                               RNG rng) {
  CreateSyclBuffer<LeafType, ElemType, Scalar, VisionMem, RNG>::create_buffer(
      ptr, dt, rng);
}

/// function create_sycl_buffer
/// \brief template deduction for CreateSyclBuffer struct. this one create
/// another buffer from accepting an input buffer. It is used for creating
/// sub-buffer
/// template parameters:
/// \tparam LeafType: determines the memory type
/// \tparam ElemType : determines the type of the element in each memory
/// \tparam Scalar : determines the type of each channel of each element
/// \tparam VisionMem: represent the type of the memory created by using SyclMem
/// \tparam RNG : the sycl range type for creating memory
/// function parameters
/// parameters:
/// \param ptr : shared_ptr containing the VisionMem
/// \param rng : the sycl range for creating buffer
/// \param dt : the input buffer for creating new sub-buffer from it
/// \return void
template <size_t LeafType, typename ElemType, typename Scalar,
          typename VisionMem, typename RNG>
inline void create_sycl_buffer(std::shared_ptr<VisionMem> &ptr, VisionMem dt,
                               RNG rng) {
  CreateSyclBuffer<memory_type::Const, ElemType, Scalar, VisionMem,
                   RNG>::create_buffer(ptr, dt, rng);
}

/// function create_sycl_buffer
/// \brief template deduction for CreateSyclBuffer struct when there is host
/// pointer
/// template parameters:
/// \tparam LeafType: determines the memory type
/// \tparam ElemType : determines the type of the element in each memory
/// \tparam Scalar : determines the type of each channel of each element
/// \tparam VisionMem: represent the type of the memory created by using SyclMem
/// \tparam RNG : the sycl range type for creating memory
/// function parameters
/// parameters:
/// \param ptr : shared_ptr containing the VisionMem
/// \param rng : the sycl range for creating buffer
/// \return void
template <size_t LeafType, typename ElemType, typename Scalar,
          typename VisionMem, typename RNG>
inline void create_sycl_buffer(std::shared_ptr<VisionMem> &ptr, RNG rng) {
  CreateSyclBuffer<LeafType, ElemType, Scalar, VisionMem, RNG>::create_buffer(
      ptr, rng);
}

/// \struct BufferUpdate
/// \brief This is used to update the Vision Memory with new value
/// update sycl buffer at the moment we use ptr.reset() because it was faster
/// than getting the host pointer and updating it in the host side.
/// template parameters:
/// \tparam LeafType : is the memory type
/// \tparam Rows: is the row size of the buffer
/// \tparam Cols: is the column size of the buffer
/// \tparam ElemType: is the type of element in the buffer
/// \tparam Scalar is the type of each channel of the element
/// \tparam VisionMem is the created SyclMem
template <size_t LeafType, size_t Rows, size_t Cols, typename ElemType,
          typename Scalar, typename VisionMem>
struct BufferUpdate {
  /// function buffer_update
  /// \brief this function is used to update the sycl buffer with a new value
  /// parameters:
  /// \param ptr : is the shared_ptr containing the SyclMem
  /// \param dt: is the pointer containing the new value for the buffer
  /// \return void
  static inline void buffer_update(std::shared_ptr<VisionMem> &ptr,
                                   Scalar *dt) {
    auto host_ptr =
        (*ptr)
            .template get_access<cl::sycl::access::mode::discard_write,
                                 cl::sycl::access::target::host_buffer>()
            .get_pointer();

    memcpy(host_ptr, dt, sizeof(Scalar) * ElemType::elements * Rows * Cols);
  }
};

/// \brief specialisation of the BufferUpdate when the memory_type is Image
template <size_t Rows, size_t Cols, typename ElemType, typename Scalar,
          typename VisionMem>
struct BufferUpdate<memory_type::Image, Rows, Cols, ElemType, Scalar,
                    VisionMem> {
  /// function buffer_update
  /// \brief this function is used to update the sycl buffer with a new value
  /// parameters:
  /// \param ptr : is the shared_ptr containing the SyclMem
  /// \param dt: is the pointer containing the new value for the buffer
  /// \return void
  using Properties = ImageProperties<ElemType, Scalar>;
  static inline void buffer_update(std::shared_ptr<VisionMem> &ptr,
                                   Scalar *dt) {
    static_assert(true, "image is not supported in this version");
  }
};

/// \brief specialisation of the BufferUpdate when the memory_type is Constant
/// variable
template <size_t Rows, size_t Cols, typename ElemType, typename Scalar,
          typename VisionMem>
struct BufferUpdate<memory_type::Const, Rows, Cols, ElemType, Scalar,
                    VisionMem> {
  /// function buffer_update
  /// \brief this function is used to update the sycl buffer with a new value
  /// parameters:
  /// \param ptr : is the shared_ptr containing the SyclMem
  /// \param dt: is the pointer containing the new value for the buffer
  /// \return void
  static inline void buffer_update(std::shared_ptr<VisionMem> &ptr,
                                   VisionMem dt) {
    *ptr = dt;
  }
};

/// function buffer_update
/// \brief template deduction function for BufferUpdate
/// template parameters:
/// \tparam LeafType : is the memory type
/// \tparam Rows: is the row size of the buffer
/// \tparam Cols: is the column size of the buffer
/// \tparam ElemType: is the type of element in the buffer
/// \tparam Scalar is the type of each channel of the element
/// \tparam VisionMem is the created SyclMem
/// function parameters:
/// \param ptr : is the shared_ptr containing the SyclMem
/// \param dt: is the pointer containing the new value for the buffer
/// \return void
template <size_t LeafType, size_t Rows, size_t Cols, typename ElemType,
          typename Scalar, typename VisionMem>
inline void buffer_update(std::shared_ptr<VisionMem> &ptr, Scalar *dt) {
  BufferUpdate<LeafType, Rows, Cols, ElemType, Scalar,
               VisionMem>::buffer_update(ptr, dt);
}
}  // internal
}  // visioncpp

// Vision Memories Headers
#include "mem_const.hpp"
#include "mem_prop.hpp"
#include "mem_virtual.hpp"
#include "mem_vision.hpp"
// memory_access in sycl
#include "memory_access/memory_access.hpp"
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_HPP_
