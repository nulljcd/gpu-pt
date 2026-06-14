use std::fs;

use bytemuck::{bytes_of, checked::cast_slice};
use glam::{EulerRot, Mat3, Vec3};
use image::{DynamicImage, ExtendedColorType, ImageError, ImageReader, save_buffer};
use wgpu::util::DeviceExt;

struct BoundingBox {
    min: Vec3,
    max: Vec3,
}

impl BoundingBox {
    fn empty() -> Self {
        Self {
            min: Vec3::INFINITY,
            max: Vec3::NEG_INFINITY,
        }
    }

    fn expand_to_bounds(&mut self, min: Vec3, max: Vec3) {
        self.min = self.min.min(min);
        self.max = self.max.max(max);
    }

    fn expand_to_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    fn area(&self) -> f32 {
        let size: Vec3 = self.max - self.min;
        2. * (size.x * size.y + size.x * size.z + size.y * size.z)
    }
}

struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
}

impl Triangle {
    fn min(&self) -> Vec3 {
        self.v0.min(self.v1).min(self.v2)
    }

    fn max(&self) -> Vec3 {
        self.v0.max(self.v1).max(self.v2)
    }

    fn centroid(&self) -> Vec3 {
        (self.v0 + self.v1 + self.v2) * (1. / 3.)
    }
}

enum BvhNode {
    Internal {
        left_child_index: usize,
        right_child_index: usize,
        bounding_box: BoundingBox,
    },
    Leaf {
        triangles_range_start: usize,
        triangles_range_end: usize,
        bounding_box: BoundingBox,
    },
}

struct GpuUniforms {
    bytes: Vec<u8>,
}

impl GpuUniforms {
    fn build(
        resolution: (usize, usize),
        num_samples: usize,
        num_bounces: usize,
        bvh_root_node_index: usize,
        camera_position: Vec3,
        camera_rotation: Mat3,
        camera_fov: f32,
        environment_image_resolution: (usize, usize),
    ) -> Self {
        let mut bytes: Vec<u8> = Vec::new();

        bytes.extend_from_slice(cast_slice(&[
            resolution.0 as u32,
            num_samples as u32,
            num_bounces as u32,
            bvh_root_node_index as u32,
        ]));
        bytes.extend_from_slice(cast_slice(&[
            camera_position.x,
            camera_position.y,
            camera_position.z,
            camera_fov,
            camera_rotation.x_axis.x,
            camera_rotation.x_axis.y,
            camera_rotation.x_axis.z,
            1. / (num_samples as f32),
            camera_rotation.y_axis.x,
            camera_rotation.y_axis.y,
            camera_rotation.y_axis.z,
            (resolution.0 as f32) / (resolution.1 as f32),
            camera_rotation.z_axis.x,
            camera_rotation.z_axis.y,
            camera_rotation.z_axis.z,
            (camera_fov * 0.5).tan(),
            1. / (resolution.0 as f32),
            1. / (resolution.1 as f32),
            environment_image_resolution.0 as f32,
            environment_image_resolution.1 as f32,
        ]));

        Self { bytes }
    }
}

struct GpuTrianglesSection0 {
    bytes: Vec<u8>,
}

impl GpuTrianglesSection0 {
    fn build(triangles: &[Triangle]) -> Self {
        let mut bytes: Vec<u8> = Vec::new();

        triangles.iter().for_each(|triangle| {
            let u: Vec3 = triangle.v1 - triangle.v0;
            let v: Vec3 = triangle.v2 - triangle.v0;
            let uu: f32 = u.length_squared();
            let uv: f32 = u.dot(v);
            let vv: f32 = v.length_squared();
            let normal: Vec3 = u.cross(v).normalize();
            let inv_d1: f32 = 1. / (uu * vv - uv * uv);

            bytes.extend_from_slice(cast_slice(&[
                triangle.v0.x,
                triangle.v0.y,
                triangle.v0.z,
                uu,
                u.x,
                u.y,
                u.z,
                uv,
                v.x,
                v.y,
                v.z,
                vv,
                normal.x,
                normal.y,
                normal.z,
                inv_d1,
            ]));
        });

        Self { bytes }
    }
}

struct GpuTrianglesSection1 {
    bytes: Vec<u8>,
}

impl GpuTrianglesSection1 {
    fn build(triangles: &[Triangle]) -> Self {
        let mut bytes: Vec<u8> = Vec::new();

        triangles.iter().for_each(|triangle| {
            let normal: Vec3 = (triangle.v1 - triangle.v0)
                .cross(triangle.v2 - triangle.v0)
                .normalize();

            bytes.extend_from_slice(cast_slice(&[normal.x, normal.y, normal.z, 0.]));
        });

        Self { bytes }
    }
}

struct GpuBvhNodesSection0 {
    bytes: Vec<u8>,
}

impl GpuBvhNodesSection0 {
    fn build(bvh_nodes: &[BvhNode]) -> Self {
        let mut bytes: Vec<u8> = Vec::new();

        bvh_nodes.iter().for_each(|bvh_node| {
            let (a, b, node_type) = match bvh_node {
                BvhNode::Internal {
                    left_child_index,
                    right_child_index,
                    bounding_box: _,
                } => (*left_child_index as u32, *right_child_index as u32, 0),
                BvhNode::Leaf {
                    triangles_range_start,
                    triangles_range_end,
                    bounding_box: _,
                } => (
                    *triangles_range_start as u32,
                    *triangles_range_end as u32,
                    1,
                ),
            };

            bytes.extend_from_slice(cast_slice(&[a, b, node_type, 0]));
        });

        Self { bytes }
    }
}

struct GpuBvhNodesSection1 {
    bytes: Vec<u8>,
}

impl GpuBvhNodesSection1 {
    fn build(bvh_nodes: &[BvhNode]) -> Self {
        let mut bytes: Vec<u8> = Vec::new();

        bvh_nodes.iter().for_each(|bvh_node| {
            let (min, max) = match bvh_node {
                BvhNode::Internal {
                    left_child_index: _,
                    right_child_index: _,
                    bounding_box,
                } => (bounding_box.min, bounding_box.max),
                BvhNode::Leaf {
                    triangles_range_start: _,
                    triangles_range_end: _,
                    bounding_box,
                } => (bounding_box.min, bounding_box.max),
            };

            bytes.extend_from_slice(cast_slice(&[
                min.x, min.y, min.z, 0., max.x, max.y, max.z, 0.,
            ]));
        });

        Self { bytes }
    }
}

struct GpuImage {
    bytes: Vec<u8>,
}

impl GpuImage {
    fn build(buffer: &[Vec3]) -> Self {
        let mut bytes: Vec<u8> = Vec::new();

        buffer
            .iter()
            .for_each(|color| bytes.extend_from_slice(bytes_of(&[color.x, color.y, color.z, 0.])));

        Self { bytes }
    }
}

fn setup_gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance: wgpu::Instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter: wgpu::Adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;

    let downlevel_capabilities: wgpu::DownlevelCapabilities = adapter.get_downlevel_capabilities();
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        return None;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;

    let adapter_info: wgpu::AdapterInfo = adapter.get_info();
    println!("gpu adapter info:");
    println!("  - name: {:?}", adapter_info.name);
    println!("  - device type: {:?}", adapter_info.device_type);
    println!("  - backend: {:?}", adapter_info.backend);

    Some((device, queue))
}

fn load_triangles_from_obj(
    path: &str,
    position: Vec3,
    scale: Vec3,
    rotation: Mat3,
) -> Option<Vec<Triangle>> {
    let contents: String = fs::read_to_string(path).ok()?;

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<Triangle> = Vec::new();

    for line in contents.lines() {
        let line: &str = line.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() == 4 {
            if parts[0] == "v" {
                vertices.push(Vec3::new(
                    parts[1].parse::<f32>().ok()?,
                    parts[2].parse::<f32>().ok()?,
                    parts[3].parse::<f32>().ok()?,
                ));
            } else if parts[0] == "f" {
                triangles.push(Triangle {
                    v0: rotation
                        * *vertices.get(parts[1].split('/').next()?.parse::<usize>().ok()? - 1)?
                        * scale
                        + position,
                    v1: rotation
                        * *vertices.get(parts[2].split('/').next()?.parse::<usize>().ok()? - 1)?
                        * scale
                        + position,
                    v2: rotation
                        * *vertices.get(parts[3].split('/').next()?.parse::<usize>().ok()? - 1)?
                        * scale
                        + position,
                });
            }
        }
    }

    Some(triangles)
}

fn build_bvh(input_triangles: Vec<Triangle>) -> (Vec<Triangle>, Vec<BvhNode>, usize) {
    fn split(
        input_triangles: Vec<Triangle>,
        bvh_nodes: &mut Vec<BvhNode>,
        final_triangles: &mut Vec<Triangle>,
        depth: usize,
    ) -> usize {
        let input_triangles_len: usize = input_triangles.len();
        let mut bounding_box: BoundingBox = BoundingBox::empty();
        let mut centroid_bounds: BoundingBox = BoundingBox::empty();

        for triangle in &input_triangles {
            bounding_box.expand_to_bounds(triangle.min(), triangle.max());
            centroid_bounds.expand_to_point(triangle.centroid());
        }

        let bounding_box_area: f32 = bounding_box.area();

        if input_triangles_len <= 4 || depth >= 64 {
            let triangles_range_start: usize = final_triangles.len();
            final_triangles.extend(input_triangles);
            let triangles_range_end: usize = final_triangles.len();

            bvh_nodes.push(BvhNode::Leaf {
                triangles_range_start,
                triangles_range_end,
                bounding_box,
            });
        } else {
            let centroid_extent: Vec3 = centroid_bounds.max - centroid_bounds.min;
            let centroid_center: Vec3 = (centroid_bounds.min + centroid_bounds.max) * 0.5;

            let mut best_cost: f32 = f32::INFINITY;
            let mut best_sides: Option<Vec<bool>> = None;

            let mut left_indices: Vec<usize> = Vec::new();
            let mut right_indices: Vec<usize> = Vec::new();

            let test_count: i32 = 7;
            let half_tests: f32 = (test_count - 1) as f32 / 2.0;

            for axis in 0..3 {
                if centroid_extent[axis] <= 1e-6 {
                    continue;
                }

                for test in 0..test_count {
                    left_indices.clear();
                    right_indices.clear();

                    let offset: f32 = (test as f32 - half_tests) / half_tests;
                    let split_position: f32 =
                        centroid_center[axis] + offset * centroid_extent[axis] * 0.75;

                    for (i, triangle) in input_triangles.iter().enumerate() {
                        if triangle.centroid()[axis] < split_position {
                            left_indices.push(i);
                        } else {
                            right_indices.push(i);
                        }
                    }

                    if left_indices.is_empty() || right_indices.is_empty() {
                        continue;
                    }

                    let mut left_bounding_box: BoundingBox = BoundingBox::empty();
                    let mut right_bounding_box: BoundingBox = BoundingBox::empty();

                    for &i in &left_indices {
                        let t: &Triangle = &input_triangles[i];
                        left_bounding_box.expand_to_bounds(t.min(), t.max());
                    }
                    for &i in &right_indices {
                        let t: &Triangle = &input_triangles[i];
                        right_bounding_box.expand_to_bounds(t.min(), t.max());
                    }

                    let cost: f32 = (left_bounding_box.area() / bounding_box_area)
                        * (left_indices.len() as f32)
                        + (right_bounding_box.area() / bounding_box_area)
                            * (right_indices.len() as f32);

                    if cost < best_cost {
                        best_cost = cost;
                        let mut sides: Vec<bool> = vec![false; input_triangles_len];
                        for &i in &right_indices {
                            sides[i] = true;
                        }
                        best_sides = Some(sides);
                    }
                }
            }

            match best_sides {
                Some(best_sides) => {
                    let mut left_triangles: Vec<Triangle> = Vec::new();
                    let mut right_triangles: Vec<Triangle> = Vec::new();

                    for (side, triangle) in best_sides.into_iter().zip(input_triangles) {
                        if side {
                            right_triangles.push(triangle);
                        } else {
                            left_triangles.push(triangle);
                        }
                    }

                    let left_child_index: usize =
                        split(left_triangles, bvh_nodes, final_triangles, depth + 1);
                    let right_child_index: usize =
                        split(right_triangles, bvh_nodes, final_triangles, depth + 1);

                    bvh_nodes.push(BvhNode::Internal {
                        left_child_index,
                        right_child_index,
                        bounding_box,
                    });
                }
                None => {
                    let triangles_range_start: usize = final_triangles.len();
                    final_triangles.extend(input_triangles);
                    let triangles_range_end: usize = final_triangles.len();

                    bvh_nodes.push(BvhNode::Leaf {
                        triangles_range_start,
                        triangles_range_end,
                        bounding_box,
                    });
                }
            }
        }

        bvh_nodes.len() - 1
    }

    let mut bvh_nodes: Vec<BvhNode> = Vec::new();
    let mut final_triangles: Vec<Triangle> = Vec::new();

    let bvh_root_node_index: usize =
        split(input_triangles, &mut bvh_nodes, &mut final_triangles, 0);

    (final_triangles, bvh_nodes, bvh_root_node_index)
}

pub fn load_image_from_hdr(path: &str, strength: f32) -> Option<(Vec<Vec3>, (usize, usize))> {
    let img: DynamicImage = ImageReader::open(path).ok()?.decode().ok()?;
    let rgb32f: image::ImageBuffer<image::Rgb<f32>, Vec<f32>> = match img {
        DynamicImage::ImageRgb32F(img) => img,
        other => other.to_rgb32f(),
    };
    let (width, height) = rgb32f.dimensions();
    let resolution: (usize, usize) = (width as usize, height as usize);
    let buffer: Vec<Vec3> = rgb32f
        .pixels()
        .map(|p: &image::Rgb<f32>| Vec3::new(p[0] * strength, p[1] * strength, p[2] * strength))
        .collect();
    Some((buffer, resolution))
}

fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resolution: (usize, usize),
    num_samples: usize,
    num_bounces: usize,
    camera_position: Vec3,
    camera_rotation: Mat3,
    camera_fov: f32,
    triangles: &[Triangle],
    bvh_nodes: &[BvhNode],
    bvh_root_node_index: usize,
    environment_image_buffer: &[Vec3],
    environment_image_resolution: (usize, usize),
) -> Vec<Vec3> {
    let shader_module: wgpu::ShaderModule =
        device.create_shader_module(wgpu::include_wgsl!("renderer.wgsl"));

    let gpu_uniforms: GpuUniforms = GpuUniforms::build(
        resolution,
        num_samples,
        num_bounces,
        bvh_root_node_index,
        camera_position,
        camera_rotation,
        camera_fov,
        environment_image_resolution,
    );
    let gpu_triangles_section0: GpuTrianglesSection0 = GpuTrianglesSection0::build(triangles);
    let gpu_triangles_section1: GpuTrianglesSection1 = GpuTrianglesSection1::build(triangles);
    let gpu_bvh_nodes_section0: GpuBvhNodesSection0 = GpuBvhNodesSection0::build(bvh_nodes);
    let gpu_bvh_nodes_section1: GpuBvhNodesSection1 = GpuBvhNodesSection1::build(bvh_nodes);
    let gpu_environment_image: GpuImage = GpuImage::build(environment_image_buffer);

    let triangles_section0_buffer: wgpu::Buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &gpu_triangles_section0.bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let triangles_section1_buffer: wgpu::Buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &gpu_triangles_section1.bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let bvh_nodes_section0_buffer: wgpu::Buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &gpu_bvh_nodes_section0.bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let bvh_nodes_section1_buffer: wgpu::Buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &gpu_bvh_nodes_section1.bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let environment_image_buffer_buffer: wgpu::Buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &gpu_environment_image.bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let uniforms_buffer: wgpu::Buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &gpu_uniforms.bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let output_pixel_count: u32 = (resolution.0 * resolution.1) as u32;
    let output_pixel_bytes: u64 = std::mem::size_of::<[f32; 4]>() as u64;
    let output_buffer_bytes: u64 = output_pixel_bytes * output_pixel_count as u64;

    let output_buffer: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let download_buffer: wgpu::Buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout: wgpu::BindGroupLayout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline_layout: wgpu::PipelineLayout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline: wgpu::ComputePipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group: wgpu::BindGroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: triangles_section0_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: triangles_section1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bvh_nodes_section0_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: bvh_nodes_section1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: environment_image_buffer_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let num_workgroups: u32 = output_pixel_count.div_ceil(64);

    let mut encoder: wgpu::CommandEncoder = device.create_command_encoder(&Default::default());

    let mut compute_pass: wgpu::ComputePass<'_> =
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
    drop(compute_pass);

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &download_buffer, 0, output_buffer.size());

    let command_buffer: wgpu::CommandBuffer = encoder.finish();

    queue.submit([command_buffer]);

    let buffer_slice: wgpu::BufferSlice<'_> = download_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    let _ = device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let buffer_view: wgpu::BufferView = buffer_slice.get_mapped_range();
    let raw_values: &[[f32; 4]] = cast_slice(&buffer_view);
    let buffer: Vec<Vec3> = raw_values
        .iter()
        .map(|raw_pixel| Vec3::new(raw_pixel[0], raw_pixel[1], raw_pixel[2]))
        .collect();
    drop(buffer_view);
    download_buffer.unmap();

    buffer
}

pub enum ViewTransform {
    None,
    Aces,
}

impl ViewTransform {
    fn aces(color: &Vec3) -> Vec3 {
        const A: f32 = 2.51;
        const B: f32 = 0.03;
        const C: f32 = 2.43;
        const D: f32 = 0.59;
        const E: f32 = 0.14;
        const F: f32 = 0.75;

        (color * (A * color + B)) / (color * (C * color + D) + E) * F
    }

    pub fn apply(&self, color: &mut Vec3) {
        match self {
            ViewTransform::None => {}
            ViewTransform::Aces => *color = Self::aces(color),
        }
    }
}

pub enum Colorspace {
    None,
    Srgb,
}

impl Colorspace {
    fn srgb_component(linear: f32) -> f32 {
        if linear <= 0.0031308 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1. / 2.4) - 0.055
        }
    }

    fn srgb(color: &Vec3) -> Vec3 {
        Vec3::new(
            Self::srgb_component(color.x),
            Self::srgb_component(color.y),
            Self::srgb_component(color.z),
        )
    }

    pub fn apply(&self, color: &mut Vec3) {
        match self {
            Colorspace::None => {}
            Colorspace::Srgb => *color = Self::srgb(color),
        }
    }
}

fn color_manage(
    buffer: &mut Vec<Vec3>,
    exposure_value: f32,
    view_transform: &ViewTransform,
    colorspace: &Colorspace,
) {
    let exposure_multiplier: f32 = exposure_value.exp2();
    buffer.iter_mut().for_each(|color| {
        *color *= exposure_multiplier;
        view_transform.apply(color);
        colorspace.apply(color);
    });
}

fn save_buffer_to_file(
    resolution: (usize, usize),
    buffer: &[Vec3],
    path: &str,
) -> Result<(), ImageError> {
    #[inline]
    const fn to_u8(v: f32) -> u8 {
        (v.clamp(0., 1.) * 255. + 0.5) as u8
    }

    let byte_buffer: Vec<u8> = buffer
        .iter()
        .map(|c| [to_u8(c.x), to_u8(c.y), to_u8(c.z)])
        .collect::<Vec<[u8; 3]>>()
        .iter()
        .flatten()
        .copied()
        .collect();

    save_buffer(
        path,
        &byte_buffer,
        resolution.0 as u32,
        resolution.1 as u32,
        ExtendedColorType::Rgb8,
    )
}

fn main() {
    let (device, queue) = setup_gpu().expect("failed to setup gpu");

    // let resolution: (usize, usize) = (1920, 1080);
    let resolution: (usize, usize) = (2560, 1600);
    // let resolution: (usize, usize) = (256, 160);

    let mut input_triangles: Vec<Triangle> = Vec::new();
    input_triangles.extend(
        load_triangles_from_obj(
            "res/plane.obj",
            Vec3::new(0., -0.275, 0.),
            Vec3::splat(3.),
            Mat3::from_euler(EulerRot::XYZ, 0., 0., 0.),
        )
        .expect("failed to load obj file"),
    );
    input_triangles.extend(
        load_triangles_from_obj(
            "res/dragon.obj",
            Vec3::new(0., 0., 0.),
            Vec3::ONE,
            Mat3::from_euler(EulerRot::XYZ, 0., 2., 0.),
        )
        .expect("failed to load obj file"),
    );

    let (triangles, bvh_nodes, bvh_root_node_index) = build_bvh(input_triangles);

    let (environment_image_buffer, environment_image_resolution) =
        load_image_from_hdr("res/sky.hdr", 0.2).expect("failed to load environment from hdr");

    let mut buffer: Vec<Vec3> = render(
        &device,
        &queue,
        resolution,
        64,
        6,
        Vec3::new(0., 0., 1.),
        Mat3::from_euler(glam::EulerRot::XYZ, 0., 0., 0.),
        1.,
        &triangles,
        &bvh_nodes,
        bvh_root_node_index,
        &environment_image_buffer,
        environment_image_resolution,
    );

    color_manage(&mut buffer, 0., &ViewTransform::Aces, &Colorspace::Srgb);

    save_buffer_to_file(resolution, &buffer, "render.png").expect("failed to save buffer to file");
}
