// Simple sequential prefix sum — works for small arrays (grid cells < 65536)
// Single thread computes exclusive prefix sum. This is fast enough because
// the grid typically has only ~1500 cells.

struct Params {
  numCells : u32,
  _pad1 : u32,
  _pad2 : u32,
  _pad3 : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> cellCounts : array<u32>;
@group(0) @binding(2) var<storage, read_write> cellOffsets : array<u32>;

@compute @workgroup_size(1)
fn main() {
  var sum = 0u;
  for (var i = 0u; i < params.numCells; i++) {
    cellOffsets[i] = sum;
    sum += cellCounts[i];
  }
}
