# Nim implementation of a3m format parsing, mirroring
# `alphafold3_pytorch.data.msa_parsing.parse_a3m`:
# FASTA-style headers, multi-line sequences, and lowercase insertions counted
# into the deletion vector.
#
# The semantics match the pure-Python path exactly, including the Python
# `str` quirks this port relies on:
#
# * `str.splitlines()` line boundaries: `\n`, `\r\n`, `\r`, `\v`, `\f`,
#   `\x1c`, `\x1d`, `\x1e`, `\x85`, `\u2028` and `\u2029`. A trailing line
#   break does not produce a trailing empty line.
# * `str.strip()` unicode whitespace, applied to every line before parsing.
# * `str.islower()` is unicode-aware: non-ASCII lowercase characters count
#   toward the deletion count but are *kept* in the sequence, because the
#   Python path removes only ASCII lowercase characters via `str.translate`.
#
# Pure-ASCII input (the overwhelmingly common case) takes a byte-level fast
# path; unicode input is decoded rune by rune.
#
# Returns `(sequences, deletion_matrix_bytes, deletion_row_lengths,
# descriptions)`; the deletion matrix crosses the boundary zero-copy as an
# int32 `bytes` buffer.

import nimpy
import nimpy/py_lib as lib
import nimpy/py_types
import std/strutils
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

# python `str.splitlines()` line breaks, by unicode code point

proc isPyLineBreak(code: int): bool =
  code == 0x0A or code == 0x0D or code == 0x0B or code == 0x0C or
  code == 0x1C or code == 0x1D or code == 0x1E or code == 0x85 or
  code == 0x2028 or code == 0x2029

# python `str.isspace()` characters

proc isPySpace(code: int): bool =
  code == 0x20 or code == 0x09 or code == 0x0A or code == 0x0B or
  code == 0x0C or code == 0x0D or code == 0x1C or code == 0x1D or
  code == 0x1E or code == 0x85 or code == 0xA0 or code == 0x1680 or
  (code >= 0x2000 and code <= 0x200A) or code == 0x2028 or code == 0x2029 or
  code == 0x202F or code == 0x205F or code == 0x3000

proc utf8Width(b: uint8): int =
  if b < 0x80: 1
  elif (b shr 5) == 0b110: 2
  elif (b shr 4) == 0b1110: 3
  else: 4

proc decodeRuneAt(s: string, i: int): (int, int) =
  ## Decode the unicode code point starting at byte `i`; returns `(code, width)`.
  let b = int(s[i].uint8)
  if b < 0x80:
    (b, 1)
  elif (b shr 5) == 0b110:
    ((b and 0x1F) shl 6 or int(s[i + 1].uint8 and 0x3F), 2)
  elif (b shr 4) == 0b1110:
    ((b and 0x0F) shl 12 or (int(s[i + 1].uint8 and 0x3F) shl 6) or
     int(s[i + 2].uint8 and 0x3F), 3)
  else:
    ((b and 0x07) shl 18 or (int(s[i + 1].uint8 and 0x3F) shl 12) or
     (int(s[i + 2].uint8 and 0x3F) shl 6) or int(s[i + 3].uint8 and 0x3F), 4)

proc runeStartBefore(s: string, i: int): int =
  ## Start byte of the rune ending at byte offset `i` (i.e. the rune that
  ## contains byte `i - 1`).
  var j = i - 1
  while j > 0 and s[j].uint8 >= 0x80 and s[j].uint8 <= 0xBF:
    dec j
  j

proc isAllAscii(s: string): bool =
  for c in s:
    if c.int > 0x7F:
      return false
  true

proc pySplitLines(s: string): seq[string] =
  ## Python `str.splitlines(keepends = False)` equivalent over raw UTF-8.
  result = @[]
  if s.len == 0:
    return

  var line_start = 0
  var i = 0
  while i < s.len:
    let (code, width) = decodeRuneAt(s, i)
    if isPyLineBreak(code):
      var line_end = i + width
      if code == 0x0D and line_end < s.len and s[line_end] == '\n':
        line_end += 1  # `\r\n` is a single boundary
      result.add(s[line_start ..< i])
      line_start = line_end
      i = line_end
    else:
      i += width

  if line_start < s.len:
    result.add(s[line_start .. ^1])

proc stripPy(line: string): (int, int) =
  ## Python `str.strip()` over raw UTF-8; returns the byte range.
  var start = 0
  var finish = line.len

  while start < finish:
    let (code, width) = decodeRuneAt(line, start)
    if not isPySpace(code):
      break
    start += width

  while finish > start:
    let rune_start = runeStartBefore(line, finish)
    let (code, _) = decodeRuneAt(line, rune_start)
    if not isPySpace(code):
      break
    finish = rune_start

  (start, finish)

proc appendRune(seq: var string, line: string, rune_start: int, width: int) =
  for k in rune_start ..< rune_start + width:
    seq.add(line[k])

proc parseA3mLines(a3m_string: string): (seq[string], seq[string], seq[seq[int32]]) =
  ## Shared line-parsing logic; the per-character lowercasing semantics differ
  ## between the ASCII fast path and the unicode path.
  result = (newSeq[string](), newSeq[string](), newSeq[seq[int32]]())
  var pending_deletion_counts = newSeq[int32]()
  let ascii_only = isAllAscii(a3m_string)

  for line in pySplitLines(a3m_string):
    let (start, finish) = stripPy(line)
    if finish <= start:
      continue

    let (first_code, _) = decodeRuneAt(line, start)
    if first_code == ord('>'):
      result[1].add(line[start + 1 ..< finish])
      result[0].add("")
      result[2].add(@[])
      pending_deletion_counts.add(0)
      continue

    let seq_index = result[0].len - 1
    if seq_index < 0:
      raise newException(ValueError, "a3m sequence line encountered before any '>' header")

    # python joins multi-line sequences before counting deletions, so the
    # count carries across the lines of a sequence
    var deletion_count = pending_deletion_counts[seq_index]

    if ascii_only:
      for j in start ..< finish:
        let c = line[j]
        if c.isLowerAscii:
          inc deletion_count
        else:
          result[2][seq_index].add(deletion_count)
          deletion_count = 0
          result[0][seq_index].add(c)
    else:
      var j = start
      while j < finish:
        let (code, width) = decodeRuneAt(line, j)
        if isLower(Rune(code)):
          inc deletion_count
          if code > 0x7F:
            # non-ASCII lowercase: python counts it as a deletion but keeps it
            # in the sequence (`str.translate` only removes ASCII lowercase)
            appendRune(result[0][seq_index], line, j, width)
        else:
          result[2][seq_index].add(deletion_count)
          deletion_count = 0
          appendRune(result[0][seq_index], line, j, width)
        j += width

    pending_deletion_counts[seq_index] = deletion_count

proc parse_a3m*(
    a3m_string: string
): (seq[string], PPyObject, seq[int32], seq[string]) {.exportpy.} =
  let (sequences, descriptions, deletion_matrix) = parseA3mLines(a3m_string)

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
