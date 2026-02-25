// GPU clear shader — fills a buffer with 0xFFFFFFFF
@group(0) @binding(0) var<storage, read_write> data : array<u32>;

struct ClearParams {
  count : u32,
  _pad1 : u32,
  _pad2 : u32,
  _pad3 : u32,
}

@group(0) @binding(1) var<uniform> params : ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= params.count) { return; }
  data[i] = 0xFFFFFFFFu;
}
