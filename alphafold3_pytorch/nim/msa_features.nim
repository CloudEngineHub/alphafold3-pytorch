# Nim implementation of the per-residue MSA featurization loop in
# `alphafold3_pytorch.data.data_pipeline.make_msa_features`, with the same
# semantics for polymer residues, ligands, and modified (atomized) residues.
#
# Bulk integer data crosses the Python boundary zero-copy as int32 `bytes`
# buffers, since nimpy's element-wise seq/list conversion dominates the
# runtime at production MSA sizes.
#
# Sequence indexing mirrors Python `str` (code-point indexing): ASCII
# sequences take a byte-level fast path, while sequences containing unicode
# are indexed rune by rune. Out-of-range polymer residue lookups raise a
# `ValueError` (the Python path raises `IndexError`/`AssertionError` for the
# same malformed inputs) instead of reading out of bounds.

import nimpy
import nimpy/py_lib as lib
import nimpy/py_types
import nimpy/raw_buffers
import std/unicode

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

proc isAsciiChar(c: char): bool =
  c.int <= 0x7F

proc isAllAscii(s: string): bool =
  for c in s:
    if not isAsciiChar(c):
      return false
  true

proc utf8Width(b: uint8): int =
  if b < 0x80: 1
  elif (b shr 5) == 0b110: 2
  elif (b shr 4) == 0b1110: 3
  else: 4

proc codePointAt(s: string, index: int): (bool, int) =
  ## Python `str`-style code-point lookup; `(false, 0)` when out of range.
  var byte_offset = 0
  var code_point_count = 0
  while code_point_count < index:
    if byte_offset >= s.len:
      return (false, 0)
    byte_offset += utf8Width(s[byte_offset].uint8)
    inc code_point_count
  if byte_offset >= s.len:
    return (false, 0)
  let b = int(s[byte_offset].uint8)
  if b < 0x80:
    (true, b)
  elif (b shr 5) == 0b110:
    (true, (b and 0x1F) shl 6 or int(s[byte_offset + 1].uint8 and 0x3F))
  elif (b shr 4) == 0b1110:
    (true, (b and 0x0F) shl 12 or (int(s[byte_offset + 1].uint8 and 0x3F) shl 6) or
     int(s[byte_offset + 2].uint8 and 0x3F))
  else:
    (true, (b and 0x07) shl 18 or (int(s[byte_offset + 1].uint8 and 0x3F) shl 12) or
     (int(s[byte_offset + 2].uint8 and 0x3F) shl 6) or int(s[byte_offset + 3].uint8 and 0x3F))

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
    let seq_all_ascii = isAllAscii(seq)
    # python `len(sequence)` counts code points, nim `seq.len` counts bytes
    let seq_len = if seq_all_ascii: seq.len else: seq.toRunes.len

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
        if polymer_residue_index >= seq_len:
          # python raises `IndexError: string index out of range` here
          raise newException(
            ValueError,
            "Polymer residue index length mismatch: " &
            $(polymer_residue_index + 1) & " > " & $seq_len
          )

        if seq_all_ascii:
          msa_res_type = if table[int(seq[polymer_residue_index])] >= 0:
              table[int(seq[polymer_residue_index])]
            else:
              restype_num
        else:
          let (_, code) = codePointAt(seq, polymer_residue_index)
          # python: `MSA_CHAR_TO_ID.get(res, restype_num)`; table entries are
          # ASCII chars, so anything outside the table falls back
          msa_res_type = if code < 256 and table[code] >= 0: table[code] else: restype_num

        if polymer_residue_index >= deletion_row_lengths[s]:
          # python raises `IndexError` from the deletion matrix row lookup
          raise newException(
            ValueError,
            "Deletion matrix row length mismatch for sequence " & $s & ": " &
            $(polymer_residue_index + 1) & " > " & $deletion_row_lengths[s]
          )
        msa_deletion_value = deletion_matrix_flat[deletion_offset + polymer_residue_index]

      int_ptr[s * num_res + idx] = msa_res_type
      del_ptr[s * num_res + idx] = msa_deletion_value

    if polymer_residue_index + 1 != seq_len:
      raise newException(
        ValueError,
        "Polymer residue index length mismatch: " &
        $(polymer_residue_index + 1) & " != " & $seq.len
      )

    polymer_counts[s] = int32(polymer_residue_index + 1)
    deletion_offset += deletion_row_lengths[s]

  (int_obj, del_obj, polymer_counts)
