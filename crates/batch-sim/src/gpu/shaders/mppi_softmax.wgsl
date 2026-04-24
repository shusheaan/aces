// MPPI softmax reduction kernel (WGSL).
//
// This file is concatenated at load time with `mppi_helpers.wgsl` +
// `mppi_rollout.wgsl` (see `shader.rs::full_mppi_source`). The rollout
// file owns the `MppiDims` struct plus all 11 bind-group declarations
// (bindings 0..=10), so this file declares no new bindings and no new
// structs — it only adds workgroup-scoped shared memory and the
// `softmax_reduce` @compute entry point.
//
// # Dispatch convention
//
// One workgroup = one drone. Invoke with
// `dispatch_workgroups(n_drones, 1, 1)` and `@workgroup_size(256, 1, 1)`.
// `workgroup_id.x` = drone index; `local_invocation_id.x` = thread id
// within the workgroup.
//
// # Algorithm (mirrors `crates/mppi/src/optimizer.rs:177-342`)
//
//   1. Phase 1 — min cost across samples (grid-stride load into
//      `shared_min`, then tree reduction).
//   2. Phase 2 — total weight `Σ exp(-(c - c_min) / T)` (grid-stride
//      load into `shared_sum`, then tree reduction).
//   3. Phase 3 — weighted mean control per (horizon timestep, motor).
//      Each thread owns one (h, m) slot indexed by `tid = h * 4 + m`.
//      Writes to `result[drone, h, m]`.
//
// This kernel does NOT modify `mean_ctrls` (binding 2) and does NOT do
// the warm-start shift (`[new_mean[1..H-1], hover]`) — those are a
// separate CPU or tiny-kernel step, out of scope for this slice.
//
// # Constraint
//
// `horizon * 4 <= 256` (workgroup size). Production horizon is 50 → 200
// slots used (56 idle); test horizon is 20 → 80 slots used (176 idle).
// Both fit. If horizon grows beyond 64, this kernel must be revisited
// (either larger workgroup, or a grid-stride Phase 3 loop).

// Workgroup-shared scratch buffers for the tree reductions. Each slot
// is one f32 (4 B) × 256 threads = 1024 B per buffer.
var<workgroup> shared_min: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn softmax_reduce(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let drone_idx = wid.x;
    let tid = lid.x;
    // `drone_idx` is the same for every thread in a workgroup, so this
    // early return is workgroup-uniform and does not create non-uniform
    // control flow around the later `workgroupBarrier()` calls.
    if (drone_idx >= dims.n_drones) { return; }

    let n_samples = dims.n_samples;
    let horizon = dims.horizon;
    let temperature = dims.temperature;
    let cost_base = drone_idx * n_samples;

    // ------------------------------------------------------------------
    // Phase 1 — min cost across samples.
    // ------------------------------------------------------------------
    // Grid-stride load: each thread scans `n_samples / 256` entries and
    // keeps the running min. Use a large finite value as the identity so
    // we never do arithmetic on +inf.
    var local_min: f32 = 3.4e38; // near f32::MAX
    for (var s: u32 = tid; s < n_samples; s = s + 256u) {
        local_min = min(local_min, costs[cost_base + s]);
    }
    shared_min[tid] = local_min;
    workgroupBarrier();

    // Tree reduction 256 → 128 → 64 → ... → 1. `stride` halves each
    // iteration; the loop exits when `stride` reaches 0. The inner
    // `if (tid < stride)` is non-uniform but the `workgroupBarrier()`
    // sits outside the `if`, which WGSL requires for correctness.
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let global_min = shared_min[0];

    // ------------------------------------------------------------------
    // Phase 2 — total weight Σ exp(-(c - c_min) / T).
    // ------------------------------------------------------------------
    var local_sum: f32 = 0.0;
    for (var s: u32 = tid; s < n_samples; s = s + 256u) {
        let c = costs[cost_base + s];
        local_sum = local_sum + exp(-(c - global_min) / temperature);
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let total_weight = shared_sum[0];
    // Guard against total_weight == 0 (every weight underflowed to 0).
    // `select(false_val, true_val, cond)` in WGSL.
    let inv_total = select(0.0, 1.0 / total_weight, total_weight > 0.0);

    // ------------------------------------------------------------------
    // Phase 3 — weighted mean of controls.
    // ------------------------------------------------------------------
    // Each thread owns exactly one (h, m) output slot, where
    // `tid = h * 4 + m` with `0 <= h < horizon` and `0 <= m < 4`.
    // Threads outside the valid range (`tid >= horizon * 4`) skip the
    // loop entirely — they've already contributed to the Phase 1/2
    // reductions above, so this is safe.
    let hm_count = horizon * 4u;
    if (tid < hm_count) {
        let h = tid / 4u;
        let m = tid % 4u;
        var acc: f32 = 0.0;
        // Layout: ctrls_out is [N_DRONES × N_SAMPLES × HORIZON × 4].
        let sample_stride = horizon * 4u;
        let drone_ctrls_base = drone_idx * n_samples * sample_stride;
        for (var s: u32 = 0u; s < n_samples; s = s + 1u) {
            let c = costs[cost_base + s];
            let w = exp(-(c - global_min) / temperature) * inv_total;
            let idx = drone_ctrls_base + s * sample_stride + h * 4u + m;
            acc = acc + w * ctrls_out[idx];
        }
        // result layout: [N_DRONES × H × 4].
        let out_idx = drone_idx * sample_stride + h * 4u + m;
        result[out_idx] = acc;
    }
}
