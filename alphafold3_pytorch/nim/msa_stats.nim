# Nim computation of the per-residue MSA profile and deletion means, mirroring
# `make_one_hot_np(...).mean(0)` and `np.clip(deletion_matrix, 0.0, 1.0).mean(0)`
# in `alphafold3_pytorch.data.data_pipeline.make_msa_features`.
#
# The int32 buffers produced zero-copy by `msa_chain_to_int_features` are
# consumed directly, avoiding the Python-side `tolist()` and `np.array`
# round-trips.
#
# Bit-for-bit parity notes:
# * `profile_all_seq` is the mean over sequences of an *int64* one-hot array,
#   so numpy computes an exact integer sum divided once in float64. The same
#   single float64 division of the exact int64 count is used here.
# * `deletion_mean_all_seq` is the mean of *float32* `clip(values, 0, 1)`
#   values, i.e. exact 0.0/1.0 entries; any summation order is exact, and one
#   float32 division matches numpy's mean.

import nimpy
import nimpy/py_lib as lib
import nimpy/py_types
import nimpy/raw_buffers

proc newTypedByteArray[T](size: int): (PPyObject, ptr UncheckedArray[T]) =
  ## Create a zero-copy typed buffer over a fresh `bytes` object.
  let obj = lib.pyLib.PyBytes_FromStringAndSize(nil, Py_ssize_t(size))
  if obj.isNil:
    raise newException(ValueError, "nimpy: PyBytes_FromStringAndSize failed")
  var s: ptr char
  var ln: Py_ssize_t
  if lib.pyLib.PyBytes_AsStringAndSize(obj, addr s, addr ln) != 0:
    raise newException(ValueError, "nimpy: failed to obtain bytes buffer")
  (obj, cast[ptr UncheckedArray[T]](s))

proc bytesToInt32Ptr(o: PyObject): ptr UncheckedArray[int32] =
  ## nimporter_plus passes `bytes` args as writable `bytearray`s, so the
  ## buffer protocol is used rather than the bytes-specific C API.
  var buf: RawPyBuffer
  getBuffer(o, buf, PyBUF_SIMPLE)
  result = cast[ptr UncheckedArray[int32]](buf.buf)
  release(buf)

proc msa_profile*(
    int_msa_bytes: PyObject,  # row-major int32 (num_sequences, num_res)
    deletion_bytes: PyObject, # row-major int32 (num_sequences, num_res)
    num_sequences: int32,
    num_res: int32,
    num_msa_one_hot: int32,
): (PPyObject, PPyObject) {.exportpy.} =
  ## Returns `(profile_bytes, deletion_mean_bytes)`.
  ## * `profile_all_seq`: float64 (num_res, num_msa_one_hot)
  ## * `deletion_mean_all_seq`: float32 (num_res)
  let int_ptr = bytesToInt32Ptr(int_msa_bytes)
  let del_ptr = bytesToInt32Ptr(deletion_bytes)

  var counts = newSeq[int64](num_res * num_msa_one_hot)
  var deletion_counts = newSeq[int32](num_res)

  for s in 0 ..< num_sequences:
    let row_offset = s * num_res
    for r in 0 ..< num_res:
      let value = int_ptr[row_offset + r]
      if value < 0 or value >= num_msa_one_hot:
        raise newException(
          ValueError,
          "MSA residue type " & $value & " is out of range for " &
          $num_msa_one_hot & " one-hot classes"
        )
      counts[r * num_msa_one_hot + value] += 1
      if del_ptr[row_offset + r] > 0:
        inc deletion_counts[r]

  var profile_obj, deletion_mean_obj: PPyObject
  var profile_ptr: ptr UncheckedArray[float64]
  var deletion_mean_ptr: ptr UncheckedArray[float32]
  (profile_obj, profile_ptr) = newTypedByteArray[float64](counts.len * sizeof(float64))
  (deletion_mean_obj, deletion_mean_ptr) = newTypedByteArray[float32](num_res * sizeof(float32))

  for i, count in counts:
    profile_ptr[i] = float64(count) / float64(num_sequences)
  for r in 0 ..< num_res:
    deletion_mean_ptr[r] = float32(deletion_counts[r]) / float32(num_sequences)

  (profile_obj, deletion_mean_obj)
