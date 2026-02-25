// Count how many boids are in each grid cell using atomic operations
struct Params {
  gridSizeX : u32,
  gridSizeY : u32,
  cellSize  : f32,
  numBoids  : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> positions : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> cellCounts : array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= params.numBoids) { return; }

  let pos = positions[i];
  let cellX = min(u32(pos.x / params.cellSize), params.gridSizeX - 1u);
  let cellY = min(u32(pos.y / params.cellSize), params.gridSizeY - 1u);
  let cellIdx = cellY * params.gridSizeX + cellX;

  atomicAdd(&cellCounts[cellIdx], 1u);
}
