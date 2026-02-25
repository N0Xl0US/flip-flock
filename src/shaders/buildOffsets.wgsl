// Build cell offset table — for each cell, find the start index in the sorted array
struct Params {
  numBoids  : u32,
  numCells  : u32,
  _pad1     : u32,
  _pad2     : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> cellIndices : array<u32>;
@group(0) @binding(2) var<storage, read_write> cellOffsets : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= params.numBoids) { return; }

  let cell = cellIndices[i];

  // If this is the first boid in its cell (or the very first boid),
  // record the start offset
  if (i == 0u || cell != cellIndices[i - 1u]) {
    cellOffsets[cell] = i;
  }
}
