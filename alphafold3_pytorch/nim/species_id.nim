# Nim implementation of species id extraction from MSA descriptions, mirroring
# `alphafold3_pytorch.data.msa_parsing.get_identifiers`:
#
# * `tab_separated_alignment_headers = True`: species id is the last
#   tab-separated token of the description (stripped), or `""` when the
#   description contains no tab.
# * `tab_separated_alignment_headers = False`: the first whitespace-delimited
#   token (up to the first `/`) is matched against the UniProt identifier
#   pattern, e.g. `tr|A0A146SKV9|A0A146SKV9_FUNHE` -> `FUNHE`. The Python path
#   uses `re.search` with
#   `^(?:tr|sp)\|[A-Za-z0-9]{6,10}(?:_\d)?\|(?:[A-Za-z0-9]+)_([A-Za-z0-9]){1,5}(?:_\d+)?$`
#   (VERBOSE, anchored) on the whitespace-stripped identifier; the manual
#   matcher below is equivalent, including greedy-quantifier backtracking
#   cases such as 6-10 digit accession lengths.

import nimpy

proc isAlphaNum(c: char): bool =
  (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9')

proc isDigit(c: char): bool =
  c >= '0' and c <= '9'

proc isAsciiSpace(c: char): bool =
  c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == '\v' or c == '\f'

proc parseSpeciesIdentifier(identifier: string): string =
  ## Python `_parse_sequence_identifier`: regex on `identifier.strip()`; returns
  ## `""` when there is no match.
  var start = 0
  var finish = identifier.len
  while start < finish and isAsciiSpace(identifier[start]):
    inc start
  while finish > start and isAsciiSpace(identifier[finish - 1]):
    dec finish

  let n = finish
  var i = start

  # (?:tr|sp)\|
  if n - i >= 3 and
      (identifier[i] == 't' and identifier[i + 1] == 'r' or
       identifier[i] == 's' and identifier[i + 1] == 'p') and
      identifier[i + 2] == '|':
    i += 3
  else:
    return ""

  # [A-Za-z0-9]{6,10}
  var acc = 0
  while i < n and isAlphaNum(identifier[i]):
    inc i
    inc acc
  if acc < 6 or acc > 10:
    return ""

  # (?:_\d)?
  if i + 1 < n and identifier[i] == '_' and isDigit(identifier[i + 1]):
    i += 2

  # \|
  if i >= n or identifier[i] != '|':
    return ""
  inc i

  # (?:[A-Za-z0-9]+)_
  var entry = 0
  while i < n and isAlphaNum(identifier[i]):
    inc i
    inc entry
  if entry == 0 or i >= n or identifier[i] != '_':
    return ""
  inc i

  # ([A-Za-z0-9]){1,5}
  var species = ""
  var sp = 0
  while i < n and isAlphaNum(identifier[i]) and sp < 5:
    species.add(identifier[i])
    inc i
    inc sp
  if sp == 0:
    return ""

  # (?:_\d+)?
  if i < n and identifier[i] == '_' and i + 1 < n and isDigit(identifier[i + 1]):
    i += 1
    while i < n and isDigit(identifier[i]):
      inc i

  if i != n:
    return ""

  species

proc extractSequenceIdentifier(description: string): string =
  ## Python `_extract_sequence_identifier`:
  ## `description.split()[0].partition("/")[0]` (empty when none).
  var start = 0
  let n = description.len
  while start < n and isAsciiSpace(description[start]):
    inc start
  var finish = start
  while finish < n and not isAsciiSpace(description[finish]):
    inc finish

  let token = description[start ..< finish]
  let slash = token.find('/')
  if slash >= 0:
    token[0 ..< slash]
  else:
    token

proc lastTabToken(description: string): string =
  ## Python `_parse_species_identifier` (tab-separated path):
  ## `description.split("\t")[-1].strip()`, or `""` when there is no tab.
  var last_start = -1
  var i = 0
  let n = description.len
  while i < n:
    if description[i] == '\t':
      last_start = i + 1
    inc i
  if last_start < 0:
    return ""
  var finish = n
  while finish > last_start and isAsciiSpace(description[finish - 1]):
    dec finish
  description[last_start ..< finish]

proc species_ids*(
    descriptions: seq[string],
    tab_separated_alignment_headers: bool,
): seq[string] {.exportpy.} =
  ## Python `get_identifiers(description, ...).species_id` per description.
  result = newSeq[string](descriptions.len)
  for i, description in descriptions:
    result[i] =
      if tab_separated_alignment_headers:
        lastTabToken(description)
      else:
        parseSpeciesIdentifier(extractSequenceIdentifier(description))
