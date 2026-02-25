// Scatter boids into sorted order using atomic counters on cell offsets
struct Params {
  gridSizeX : u32,
  gridSizeY : u32,
  cellSize  : f32,
  numBoids  : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> positions : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> cellOffsets : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIndices : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= params.numBoids) { return; }

  let pos = positions[i];
  let cellX = min(u32(pos.x / params.cellSize), params.gridSizeX - 1u);
  let cellY = min(u32(pos.y / params.cellSize), params.gridSizeY - 1u);
  let cellIdx = cellY * params.gridSizeX + cellX;

  // Atomically get the next slot in this cell's range
  let slot = atomicAdd(&cellOffsets[cellIdx], 1u);
  sortedIndices[slot] = i;
}
