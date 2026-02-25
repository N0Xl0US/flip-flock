// Assign each boid to a grid cell based on its position
struct Params {
  gridSizeX : u32,
  gridSizeY : u32,
  cellSize  : f32,
  numBoids  : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> positionsIn : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> cellIndices : array<u32>;
@group(0) @binding(3) var<storage, read_write> boidIndices : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= params.numBoids) { return; }

  let pos = positionsIn[i];
  let cellX = u32(pos.x / params.cellSize);
  let cellY = u32(pos.y / params.cellSize);
  let cellIdx = cellY * params.gridSizeX + cellX;

  cellIndices[i] = cellIdx;
  boidIndices[i] = i;
}
