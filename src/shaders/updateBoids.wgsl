// 2D Boids update — uses counting-sort spatial grid for O(n) neighbor lookup
struct SimParams {
  deltaT       : f32,
  visualRange  : f32,
  cohesion     : f32,
  separation   : f32,
  alignment    : f32,
  separationDist : f32,
  maxSpeed     : f32,
  minSpeed     : f32,
  boundX       : f32,
  boundY       : f32,
  turnMargin   : f32,
  turnFactor   : f32,
  numBoids     : u32,
  gridSizeX    : u32,
  gridSizeY    : u32,
  cellSize     : f32,
}

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> posIn : array<vec2f>;
@group(0) @binding(2) var<storage, read> velIn : array<vec2f>;
@group(0) @binding(3) var<storage, read_write> posOut : array<vec2f>;
@group(0) @binding(4) var<storage, read_write> velOut : array<vec2f>;
@group(0) @binding(5) var<storage, read> sortedIndices : array<u32>;
@group(0) @binding(6) var<storage, read> cellOffsets : array<u32>;
@group(0) @binding(7) var<storage, read> cellCounts : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let idx = gid.x;
  if (idx >= params.numBoids) { return; }

  var pos = posIn[idx];
  var vel = velIn[idx];

  let myCellX = i32(pos.x / params.cellSize);
  let myCellY = i32(pos.y / params.cellSize);
  let gridXY = vec2i(pos / params.cellSize); // Renamed myCellX/Y to gridXY

  var center = vec2(0.0);
  var close = vec2(0.0);
  var avgVel = vec2(0.0);
  var neighbours = 0u;

  let visualRangeSq = params.visualRange * params.visualRange;
  let separationDistSq = params.separationDist * params.separationDist;

  // Check 3x3 neighboring cells
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let nx = gridXY.x + dx;
      let ny = gridXY.y + dy;
      if (nx >= 0 && nx < i32(params.gridSizeX) && ny >= 0 && ny < i32(params.gridSizeY)) {
        let cellIdx = u32(ny) * params.gridSizeX + u32(nx);
        let count = cellCounts[cellIdx];
        let offset = cellOffsets[cellIdx];

        for (var j = 0u; j < count; j++) {
          let otherIdx = sortedIndices[offset + j];
          if (otherIdx == idx) { continue; }

          let otherPos = posIn[otherIdx];
          let otherVel = velIn[otherIdx];
          let diff = pos - otherPos;
          let distSq = dot(diff, diff);

          if (distSq < visualRangeSq && distSq > 0.0) {
            if (distSq < separationDistSq) {
              let invDistSq = 1.0 / distSq;
              close += diff * invDistSq;
            }
            center += otherPos;
            avgVel += otherVel;
            neighbours += 1u;
          }
        }
      }
    }
  }

  // Exact math from user reference snippet
  if (neighbours > 0u) {
    center /= f32(neighbours);
    avgVel /= f32(neighbours);
    vel += (center - pos) * (params.cohesion * params.deltaT);
    vel += (avgVel - vel) * (params.alignment * params.deltaT);
  }
  vel += close * (params.separation * params.deltaT);

  // Clamp speed
  let speed = length(vel);
  let clampedSpeed = clamp(speed, params.minSpeed, params.maxSpeed);
  if (speed > 0.0001) {
    vel *= clampedSpeed / speed;
  }

  pos += vel * params.deltaT;
  
  // Toroidal screen wrap (infinite flowing space)
  if (pos.x < 0.0) { pos.x += params.boundX; } // Changed boundsX to boundX
  else if (pos.x >= params.boundX) { pos.x -= params.boundX; } // Changed boundsX to boundX
  
  if (pos.y < 0.0) { pos.y += params.boundY; } // Changed boundsY to boundY
  else if (pos.y >= params.boundY) { pos.y -= params.boundY; } // Changed boundsY to boundY

  posOut[idx] = pos;
  velOut[idx] = vel;
}
