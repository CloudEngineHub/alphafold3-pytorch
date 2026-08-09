# Nim implementation of the per-residue MSA featurization loop in
# `alphafold3_pytorch.data.data_pipeline.make_msa_features`, with the same
# semantics for polymer residues, ligands, and modified (atomized) residues.
#
# Bulk integer data crosses the Python boundary zero-copy as int32 `bytes`
# buffers, since nimpy's element-wise seq/list conversion dominates the
# runtime at production MSA sizes.

import nimpy
import nimpy/py_lib as lib
import nimpy/py_types
import nimpy/raw_buffers

proc newByteArray(size: int): (PPyObject, ptr UncheckedArray[int32]) =
  ## Create a zero-copy int32 buffer over a fresh `bytes` object.
  let obj = lib.pyLib.PyBytes_FromStringAndSize(nil, Py_ssize_t(size))
  if obj.isNil:
    raise newException(ValueError, "nimpy: PyBytes_FromStringAndSize failed")
  var s: ptr char
  var ln: Py_ssize_t
  if lib.pyLib.PyBytes_AsStringAndSize(obj, addr s, addr ln) != 0:
    raise newException(ValueError, "nimpy: failed to obtain bytes buffer")
  (obj, cast[ptr UncheckedArray[int32]](s))

proc tablesToPointers(tables: seq[seq[int32]]): seq[ptr UncheckedArray[int32]] =
  ## Flat views into the per-chemtype lookup tables; nested seq access per
  ## residue is ~10x slower due to ORC refcounting.
  result = newSeq[ptr UncheckedArray[int32]](tables.len)
  for i in 0 ..< tables.len:
    if tables[i].len > 0:
      result[i] = cast[ptr UncheckedArray[int32]](unsafeAddr tables[i][0])

proc bytesToInt32Ptr(o: PyObject): ptr UncheckedArray[int32] =
  ## nimporter_plus passes `bytes` args as writable `bytearray`s, so the
  ## buffer protocol is used rather than the bytes-specific C API.
  var buf: RawPyBuffer
  getBuffer(o, buf, PyBUF_SIMPLE)
  result = cast[ptr UncheckedArray[int32]](buf.buf)
  release(buf)

proc msa_chain_to_int_features*(
    sequences: seq[string],
    chemtypes: seq[int32],
    residue_indices: seq[int32],
    deletion_matrix_bytes: PyObject, # row-major int32 (num_sequences, row lengths)
    deletion_row_lengths: seq[int32], # polymer residue count per sequence
    msa_char_to_id_tables: seq[seq[int32]], # [chemtype][256], -1 = unknown char
    restype_nums: seq[int32],               # [chemtype]
    ligand_chemtype_index: int32,
): (PPyObject, PPyObject, seq[int32]) {.exportpy.} =
  ## Returns `(int_msa_bytes, deletions_bytes, polymer_counts)`.
  let num_res = chemtypes.len
  let num_sequences = sequences.len
  let total = num_sequences * num_res

  let deletion_matrix_flat = bytesToInt32Ptr(deletion_matrix_bytes)

  var int_obj, del_obj: PPyObject
  var int_ptr, del_ptr: ptr UncheckedArray[int32]
  (int_obj, int_ptr) = newByteArray(total * int32.sizeof)
  (del_obj, del_ptr) = newByteArray(total * int32.sizeof)

  let table_ptrs = tablesToPointers(msa_char_to_id_tables)

  var polymer_counts = newSeq[int32](num_sequences)
  var deletion_offset = 0

  for s, seq in sequences:
    var polymer_residue_index = -1

    for idx in 0 ..< num_res:
      let chemtype = chemtypes[idx]
      let is_polymer = chemtype < ligand_chemtype_index
      let increment_index = idx > 0 and residue_indices[idx - 1] != residue_indices[idx]

      if is_polymer and (idx == 0 or increment_index):
        inc polymer_residue_index

      let restype_num = restype_nums[chemtype]
      let table = table_ptrs[chemtype]

      var msa_res_type: int32
      var msa_deletion_value: int32 = 0

      if not is_polymer:
        # ligands use the unknown residue type and have no deletions
        msa_res_type = restype_num
      else:
        let res = seq[polymer_residue_index]
        let c = int(res)
        msa_res_type = if c < 256 and table[c] >= 0: table[c] else: restype_num
        msa_deletion_value = deletion_matrix_flat[deletion_offset + polymer_residue_index]

      int_ptr[s * num_res + idx] = msa_res_type
      del_ptr[s * num_res + idx] = msa_deletion_value

    if polymer_residue_index + 1 != seq.len:
      raise newException(
        ValueError,
        "Polymer residue index length mismatch: " &
        $(polymer_residue_index + 1) & " != " & $seq.len
      )

    polymer_counts[s] = int32(polymer_residue_index + 1)
    deletion_offset += deletion_row_lengths[s]

  (int_obj, del_obj, polymer_counts)
