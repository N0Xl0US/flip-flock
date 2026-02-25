// Bitonic sort — sorts cellIndices and boidIndices by cell value
// Each dispatch call does one step of the sort. The host dispatches
// log2(n) * (log2(n)+1) / 2 passes with different block/step uniforms.

struct SortParams {
  numBoids  : u32,
  blockStep : u32,  // current comparison distance
  subStep   : u32,  // current sub-step distance
  _pad      : u32,
}

@group(0) @binding(0) var<uniform> params : SortParams;
@group(0) @binding(1) var<storage, read_write> cellIndices : array<u32>;
@group(0) @binding(2) var<storage, read_write> boidIndices : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= params.numBoids) { return; }

  let blockStep = params.blockStep;
  let subStep = params.subStep;

  // Determine partner index
  let partner = i ^ subStep;

  if (partner > i && partner < params.numBoids) {
    // Determine sort direction based on block
    let ascending = ((i & blockStep) == 0u);

    let cellA = cellIndices[i];
    let cellB = cellIndices[partner];

    let shouldSwap = (ascending && cellA > cellB) || (!ascending && cellA < cellB);

    if (shouldSwap) {
      // Swap cell indices
      cellIndices[i] = cellB;
      cellIndices[partner] = cellA;

      // Swap boid indices
      let boidA = boidIndices[i];
      let boidB = boidIndices[partner];
      boidIndices[i] = boidB;
      boidIndices[partner] = boidA;
    }
  }
}
