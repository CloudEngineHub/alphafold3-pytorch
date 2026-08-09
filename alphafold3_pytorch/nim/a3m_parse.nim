# Nim implementation of a3m format parsing, mirroring
# `alphafold3_pytorch.data.msa_parsing.parse_a3m`:
# FASTA-style headers, multi-line sequences, and lowercase insertions counted
# into the deletion vector.
#
# Returns `(sequences, deletion_matrix_bytes, deletion_row_lengths,
# descriptions)`; the deletion matrix crosses the boundary zero-copy as an
# int32 `bytes` buffer.

import nimpy
import nimpy/py_lib as lib
import nimpy/py_types
import std/strutils

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

proc parse_a3m*(
    a3m_string: string
): (seq[string], PPyObject, seq[int32], seq[string]) {.exportpy.} =
  var sequences: seq[string] = @[]
  var descriptions: seq[string] = @[]
  var deletion_matrix: seq[seq[int32]] = @[]
  var pending_deletion_counts: seq[int32] = @[]

  for line in a3m_string.splitLines:
    let stripped = line.strip
    if stripped.len == 0:
      continue

    if stripped[0] == '>':
      descriptions.add(stripped[1 .. ^1])
      sequences.add("")
      deletion_matrix.add(@[])
      pending_deletion_counts.add(0)
      continue

    let seq_index = sequences.len - 1
    if seq_index < 0:
      raise newException(ValueError, "a3m sequence line encountered before any '>' header")

    var deletion_count = pending_deletion_counts[seq_index]

    for c in stripped:
      if c.isLowerAscii:
        inc deletion_count
      else:
        deletion_matrix[seq_index].add(deletion_count)
        deletion_count = 0
        sequences[seq_index].add(c)

    pending_deletion_counts[seq_index] = deletion_count

  # flatten the deletion matrix into a single int32 buffer
  var total = 0
  for row in deletion_matrix:
    total += row.len

  var del_obj: PPyObject
  var del_ptr: ptr UncheckedArray[int32]
  (del_obj, del_ptr) = newByteArray(total * int32.sizeof)

  var row_lengths = newSeq[int32](deletion_matrix.len)
  var offset = 0
  for s, row in deletion_matrix:
    row_lengths[s] = int32(row.len)
    for i, value in row:
      del_ptr[offset + i] = value
    offset += row.len

  (sequences, del_obj, row_lengths, descriptions)
