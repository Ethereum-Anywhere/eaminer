#pragma once

#if defined(__has_include)
#    if __has_include(<sycl/sycl.hpp>)
#        include <sycl/sycl.hpp>
#    else
#        include <CL/sycl.hpp>
#    endif
#else
#    include <sycl/sycl.hpp>
#endif

#if !defined(SYCL_LANGUAGE_VERSION)
#    error "You need to use a SYCL compiler"
#endif

#if defined(SYCL_IMPLEMENTATION_HIPSYCL)
#    define SYCL_ATOMIC_REF sycl::atomic_ref
#    define OPT_CONSTEXPR
#elif defined(SYCL_IMPLEMENTATION_ONEAPI) || defined(SYCL_IMPLEMENTATION_INTEL)
#    define OPT_CONSTEXPR constexpr
#    if defined(__INTEL_LLVM_COMPILER)
#        define SYCL_ATOMIC_REF sycl::ext::oneapi::atomic_ref
#    else
#        define SYCL_ATOMIC_REF sycl::atomic_ref
#    endif
#else
#    error "Untested SYCL implementation"
#endif

#ifdef __HIPSYCL_ENABLE_CUDA_TARGET__
#    define BOOST_NO_CXX11_HDR_SYSTEM_ERROR
#endif