# Reusable GPU hash table (as built, 2026-10-24)

Status: as built on `main`. A reusable module: `lib/src/rhi/hash_table.{h,cpp}`,
`lib/src/rhi/hash_table_layout.h`, `shaders/common/caelix_hash_table.hlsl` (the device API) and
`shaders/hash/caelix_hash_{kernels,init,begin,maintain,finish}.hlsl` (the maintenance kernels). It was
first written for the light NEE's light cells, which were removed on 2026-10-24 (`docs/LIGHT_NEE.md`,
section "History: light cells"); no library code uses it today. A per-voxel colour average was a
planned second user. Test: `caelix_hash_table_test` (`tests/hash_table_test.cpp`,
`tests/shaders/caelix_hash_probe.hlsl`) against a CPU model.

## 1. Design

The module has no knowledge of its users. The user owns a payload pool; the table maps 64-bit keys to
payload indices.

- **Key slot** (16 B): `uint key_lo, key_hi` (64-bit fingerprint; `key_lo` 0 = empty, real keys have
  `key_lo >= 1` and `key_hi != 0`), `uint payload` (`kHashNone` = none), `uint pad`.
- **Payload header contract** (the user's payload starts with these words): `state` (free /
  uninitialised / ready), `touched` (frame of the newest lookup), and the user's key fields (so the key
  can be recomputed from the payload).
- **Capacity**: key slots = a power of two >= 4 x payload capacity (about 20 % fill when the payload is
  80 % full: 1.2 probes per hit, < 0.1 % failed inserts with 8 probes; table in section 3).
- **Lookup(key)**: probe `(hash + i) & mask`, i < 8. Both words equal = hit: the caller loads the payload
  index with ACQUIRE order and writes the payload's `touched`. `key_lo` equal and `key_hi` 0 = an insert
  in progress = pending (no entry this frame, no insert). Empty = miss (stop). Other keys are skipped.
- **LookupOrInsert(key)**: after a miss, CAS `key_lo` from 0 on the first empty slot of the window. The
  CAS returns the value it found:
  - 0 (won): pop a payload index; the user writes the payload header (key fields, `touched = frame`,
    `state = uninitialised`; never the rest of the payload); store `payload`; then a RELEASE store of
    `key_hi`. HLSL `Interlocked*` imply no barrier: the platform's acquire/release atomics
    (`CAELIX_ATOMIC_*`, inline SPIR-V) are used on both sides.
  - the same `key_lo`: another thread inserts or inserted this key: treat as pending; never insert further
    on. This keeps a key from being present twice.
  - another value: continue the probe.
  An empty free stack stores `payload = kHashNone`: the entry counts as absent until the next rebuild.
- **Free stack**: `int top` + `uint stack[P]`, initially `0 .. P-1`. Pop (inserts, during the user's lookups):
  `i = atomicAdd(top, -1) - 1`, `i < 0` = empty. Push (evictions, in the maintenance pass):
  `stack[atomicAdd(top, 1)] = freed`. A one-thread fix-up clamps `top` to 0 before any push. Pops and
  pushes never run in the same dispatch.
- **Maintenance pass** (once per frame per table, before the user's own pass):
  1. over the payload pool: a ready or uninitialised payload with `frame - touched > max_age` is freed
     (`state = free`, push); the pass also writes the compact list of ACTIVE payloads (touched by the
     previous frame) and the indirect dispatch arguments of the user's pass;
  2. the key table is rebuilt from the live payloads: clear, then insert every live payload's key
     (recomputed from its header). No tombstones, no rehash threshold, no back reference; pending and
     payload-less entries of the previous frame disappear by construction. A key whose 8-probe window is
     full at the rebuild frees its payload. Estimated ~0.1 ms at 262144 payloads (a 16 MiB clear + the
     inserts; the light cells measured 0.05 .. 0.06 ms for the whole maintenance at 262144 payloads).
  Counters: live, active, inserted, pending, failed (no slot), failed (no payload), evicted, rebuild ms.
- **C++ side**: `HashTable` owns the key buffer, the free stack, the counters and their readback; the user
  owns the payload buffer. Every layout is defined once with names and mirrored in HLSL with
  static_asserts (owner rule: no magic offsets).
- **Fingerprint**: splitmix64 rounds over the user's key words (as the light cells' old `LightCellKey`),
  `key_lo >= 1`, `key_hi != 0` forced. A 64-bit collision merges two keys (~2^-64 per pair);
  documented, not handled.

## 2. As built

- `HashTable::RecordMaintenance` takes `items_per_group` (>= 1): the active payloads per workgroup of
  the user's pass, so the indirect dispatch fits a pass that packs several payloads into one workgroup.
- The table counts `rebuild_dropped` (keys whose probe window was full at the rebuild) apart from the
  lookups' `failed_slot`.
- The counters have 8 words for the user (`kHashCounterUser`, cleared with the table's).
- `caelix_frame_layout_test` compares every constant of `caelix_hash_table.hlsl` with
  `hash_table_layout.h` by name, and the push constant block of the hash kernels with `HashPush`.

## 3. Linear probing reference (simulation, 2^18 slots, cap 8)

| fill | probes per hit | probes per miss | new keys failing (8 probes) |
| --- | --- | --- | --- |
| 0.10 | 1.06 | 1.12 | 0.00 % |
| 0.25 | 1.17 | 1.38 | 0.06 % |
| 0.40 | 1.32 | 1.89 | 0.9 % |
| 0.50 | 1.45 | 2.44 | 3.5 % |
| 0.60 | 1.60 | 3.39 | 8.7 % |
