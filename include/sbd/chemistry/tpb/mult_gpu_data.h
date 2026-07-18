/**
@file sbd/chemistry/tpb/mult_gpu_data.h
@brief Persistent GPU residency for the iteration-invariant TaskHelpers
       connectivity arrays used by the matrix-free offload mult().

The single/double excitation connectivity (the flattened `*_flat` index arrays
and their `*Offset` prefix-sum arrays) is built once per determinant set in
`helper` and is CONSTANT across every Davidson matrix-vector product. The
offload kernels in `mult.h` list these arrays in `map(to: ...)` clauses, which
-- if the data is not already resident -- re-copies them on every kernel launch
(every task, every matvec). For a subspace with millions of couplings this host
-> device traffic dominates and starves the GPU.

OpenMP semantics: `map(to:)` on data already present in the device data
environment does NOT copy; it only increments the reference count. So mapping
these arrays once with `target enter data` here makes the per-kernel `map(to:)`
clauses in mult.h no-copy present-lookups, with no change to the kernels.

Call `MapHelpersToDevice(helper)` once after `helper` is built (and after the
integrals are mapped) and `UnmapHelpersFromDevice(helper)` at teardown -- see
`sbdiag.h`. `helper` must outlive the mapped region and its vector storage must
not reallocate in between (it does not: it is const across the Davidson loop).
*/
#ifndef SBD_CHEMISTRY_TPB_MULT_GPU_DATA_H
#define SBD_CHEMISTRY_TPB_MULT_GPU_DATA_H

#include <cstddef>
#include <vector>

#include "helper.h"

#ifdef USE_HIJ_OMP_OFFLOAD

namespace sbd {

namespace detail {

// Map one index vector to the device (persistent), skipping empties. The base
// pointer matches helper[task].<array>.data() used in the mult.h kernels, so
// the kernel's map(to:) finds it present and does not re-copy.
inline void MapIndexVec(const std::vector<size_t> &v) {
  const size_t *p = v.data();
  const size_t n = v.size();
  if (n == 0) {
    return;
  }
#pragma omp target enter data map(to : p[0 : n])
}

inline void UnmapIndexVec(const std::vector<size_t> &v) {
  const size_t *p = v.data();
  const size_t n = v.size();
  if (n == 0) {
    return;
  }
#pragma omp target exit data map(delete : p[0 : n])
}

} // namespace detail

// Make the invariant per-task connectivity arrays GPU-resident. Idempotent per
// (helper, device) as long as it is paired 1:1 with UnmapHelpersFromDevice.
inline void MapHelpersToDevice(const std::vector<TaskHelpers> &helper) {
  for (const auto &h : helper) {
    // Flattened excitation index arrays (the large ones).
    detail::MapIndexVec(h.SinglesFromAlpha_flat);
    detail::MapIndexVec(h.DoublesFromAlpha_flat);
    detail::MapIndexVec(h.SinglesFromBeta_flat);
    detail::MapIndexVec(h.DoublesFromBeta_flat);
    // Prefix-sum offset arrays.
    detail::MapIndexVec(h.SinglesFromAlphaOffset);
    detail::MapIndexVec(h.DoublesFromAlphaOffset);
    detail::MapIndexVec(h.SinglesFromBetaOffset);
    detail::MapIndexVec(h.DoublesFromBetaOffset);
  }
}

// Release the arrays mapped by MapHelpersToDevice (same set, reverse is not
// required for target data).
inline void UnmapHelpersFromDevice(const std::vector<TaskHelpers> &helper) {
  for (const auto &h : helper) {
    detail::UnmapIndexVec(h.SinglesFromAlpha_flat);
    detail::UnmapIndexVec(h.DoublesFromAlpha_flat);
    detail::UnmapIndexVec(h.SinglesFromBeta_flat);
    detail::UnmapIndexVec(h.DoublesFromBeta_flat);
    detail::UnmapIndexVec(h.SinglesFromAlphaOffset);
    detail::UnmapIndexVec(h.DoublesFromAlphaOffset);
    detail::UnmapIndexVec(h.SinglesFromBetaOffset);
    detail::UnmapIndexVec(h.DoublesFromBetaOffset);
  }
}

} // namespace sbd

#endif // USE_HIJ_OMP_OFFLOAD

#endif // SBD_CHEMISTRY_TPB_MULT_GPU_DATA_H
