// Simple white 2D boid triangles oriented by velocity

struct RenderParams {
  canvasWidth  : f32,
  canvasHeight : f32,
  boidSize     : f32,
  _pad         : f32,
}

struct VertexOutput {
  @builtin(position) position : vec4f,
}

@group(0) @binding(0) var<uniform> params : RenderParams;

@vertex
fn vert_main(
  @location(0) boidPos : vec2f,
  @location(1) boidVel : vec2f,
  @builtin(vertex_index) vid : u32
) -> VertexOutput {
  var output : VertexOutput;

  // Ignore orientation, just render a tiny dot (equilateral triangle)
  let vi = vid % 3u;
  var localPos : vec2f;
  if (vi == 0u) {
    localPos = vec2(0.0, params.boidSize);
  } else if (vi == 1u) {
    localPos = vec2(-params.boidSize * 0.866, -params.boidSize * 0.5);
  } else {
    localPos = vec2(params.boidSize * 0.866, -params.boidSize * 0.5);
  }

  let worldPos = boidPos + localPos;
  let ndc = vec2(
    (worldPos.x / params.canvasWidth) * 2.0 - 1.0,
    1.0 - (worldPos.y / params.canvasHeight) * 2.0
  );

  output.position = vec4(ndc, 0.0, 1.0);
  return output;
}

@fragment
fn frag_main() -> @location(0) vec4f {
  // Soft white dot appearance
  return vec4(0.9, 0.9, 1.0, 0.8);
}
