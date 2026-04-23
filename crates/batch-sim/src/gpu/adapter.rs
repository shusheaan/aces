//! WGPU hardware detection and capability reporting.
//!
//! Probes available GPU adapters and reports what's available for
//! compute shader acceleration (MPPI trajectory rollout).

/// Simple blocking executor for wgpu futures.
/// Avoids external dependency on pollster.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    // wgpu futures resolve immediately on native platforms (Metal/Vulkan/DX12)
    // so a trivial executor suffices.
    let waker = futures_waker();
    let mut cx = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(val) => return val,
            std::task::Poll::Pending => {
                // Yield to allow GPU driver to make progress
                std::thread::yield_now();
            }
        }
    }
}

fn futures_waker() -> std::task::Waker {
    use std::task::{RawWaker, RawWakerVTable};
    fn no_op(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
    unsafe { std::task::Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

/// Information about a detected GPU adapter.
#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub backend: String,
    pub device_type: String,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    pub supports_f16: bool,
}

/// Result of probing GPU hardware.
#[derive(Debug)]
pub struct GpuProbeResult {
    pub adapters: Vec<GpuAdapterInfo>,
    pub recommended: Option<usize>,
    pub compute_capable: bool,
}

/// Probe all available GPU adapters and report capabilities.
pub fn probe_gpu() -> GpuProbeResult {
    block_on(probe_gpu_async())
}

async fn probe_gpu_async() -> GpuProbeResult {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let mut adapters_info = Vec::new();
    let mut recommended = None;

    for power in [
        wgpu::PowerPreference::HighPerformance,
        wgpu::PowerPreference::LowPower,
    ] {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await;

        if let Some(adapter) = adapter {
            let info = adapter.get_info();
            let limits = adapter.limits();
            let features = adapter.features();

            let adapter_info = GpuAdapterInfo {
                name: info.name.clone(),
                vendor: info.vendor,
                device: info.device,
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
                max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
                max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                max_buffer_size: limits.max_buffer_size,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                supports_f16: features.contains(wgpu::Features::SHADER_F16),
            };

            if recommended.is_none() && power == wgpu::PowerPreference::HighPerformance {
                recommended = Some(adapters_info.len());
            }

            let is_duplicate = adapters_info.iter().any(|a: &GpuAdapterInfo| {
                a.name == adapter_info.name && a.backend == adapter_info.backend
            });

            if !is_duplicate {
                adapters_info.push(adapter_info);
            }
        }
    }

    let compute_capable = !adapters_info.is_empty();
    if recommended.is_none() && compute_capable {
        recommended = Some(0);
    }

    GpuProbeResult {
        adapters: adapters_info,
        recommended,
        compute_capable,
    }
}

/// Run a minimal compute shader to verify GPU execution works.
///
/// Returns true if the GPU can execute compute shaders correctly.
pub fn verify_compute() -> bool {
    block_on(verify_compute_async())
}

async fn verify_compute_async() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
    {
        Some(a) => a,
        None => return false,
    };

    let (device, queue) = match adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("aces-compute"),
                ..Default::default()
            },
            None,
        )
        .await
    {
        Ok(dq) => dq,
        Err(_) => return false,
    };

    // Minimal compute shader: doubles each element in a buffer
    let shader_source = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            if idx < arrayLength(&data) {
                data[idx] = data[idx] * 2.0;
            }
        }
    "#;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("verify_compute"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("verify_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("verify_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("verify_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Input data: [1.0, 2.0, 3.0, 4.0]
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_bytes: &[u8] = bytemuck::cast_slice(&input_data);

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: input_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&storage_buffer, 0, input_bytes);

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: input_bytes.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("verify_bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    // Dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &storage_buffer,
        0,
        &readback_buffer,
        0,
        input_bytes.len() as u64,
    );
    queue.submit(Some(encoder.finish()));

    // Read back
    let slice = readback_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);

    match receiver.recv() {
        Ok(Ok(())) => {
            let data = slice.get_mapped_range();
            let result: &[f32] = bytemuck::cast_slice(&data);
            let expected = [2.0f32, 4.0, 6.0, 8.0];
            result
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6)
        }
        _ => false,
    }
}

/// Print a human-readable GPU capability report.
pub fn print_gpu_report() {
    let probe = probe_gpu();

    println!("=== WGPU GPU Hardware Report ===\n");

    if probe.adapters.is_empty() {
        println!("No GPU adapters found!");
        return;
    }

    for (i, adapter) in probe.adapters.iter().enumerate() {
        let marker = if probe.recommended == Some(i) {
            " [RECOMMENDED]"
        } else {
            ""
        };
        println!("Adapter #{}{}", i, marker);
        println!("  Name:     {}", adapter.name);
        println!("  Backend:  {}", adapter.backend);
        println!("  Type:     {}", adapter.device_type);
        println!("  Vendor:   0x{:04X}", adapter.vendor);
        println!("  Device:   0x{:04X}", adapter.device);
        println!(
            "  Compute:  workgroup_size_x={}, max_workgroups={}",
            adapter.max_compute_workgroup_size_x, adapter.max_compute_workgroups_per_dimension
        );
        println!(
            "  Memory:   max_buffer={}MB, max_storage_binding={}MB",
            adapter.max_buffer_size / (1024 * 1024),
            adapter.max_storage_buffer_binding_size / (1024 * 1024)
        );
        println!("  f16:      {}", adapter.supports_f16);
        println!();
    }

    if let Some(idx) = probe.recommended {
        let adapter = &probe.adapters[idx];
        let bytes_per_sample: u64 = 50 * 4 * 4 + 50 * 13 * 4;
        let max_samples = (adapter.max_storage_buffer_binding_size as u64) / bytes_per_sample;

        println!("--- MPPI Capacity Estimate ---");
        println!(
            "  Max parallel workgroups: {}",
            adapter.max_compute_workgroups_per_dimension
        );
        println!(
            "  Max storage buffer: {} MB",
            adapter.max_storage_buffer_binding_size / (1024 * 1024)
        );
        println!("  Bytes per MPPI sample: {} bytes", bytes_per_sample);
        println!("  Max MPPI samples in one buffer: {}", max_samples);
        println!("  At 1024 samples/drone: max {} drones", max_samples / 1024);
        println!(
            "  At 1024 samples/drone, 2 drones/battle: max {} battles",
            max_samples / 1024 / 2
        );
    }

    println!("\n--- Compute Verification ---");
    if verify_compute() {
        println!("  Compute shader execution: PASS");
    } else {
        println!("  Compute shader execution: FAIL");
    }
}
