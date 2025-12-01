@group(0) @binding(0)
var<storage, read> triangles_section0: array<TriangleSection0>;

@group(0) @binding(1)
var<storage, read> triangles_section1: array<TriangleSection1>;

@group(0) @binding(2)
var<storage, read> bvh_nodes_section0: array<BvhSection0>;

@group(0) @binding(3)
var<storage, read> bvh_nodes_section1: array<BvhSection1>;

@group(0) @binding(4)
var<storage, read> materials: array<Material>;

@group(0) @binding(5)
var<storage, read> environment_buffer : array<vec3<f32>>;

@group(0) @binding(6)
var<uniform> render_data: RenderData;

@group(0) @binding(7)
var<storage, read_write> output_buffer : array<vec3<f32>>;

struct TriangleSection0 {
    v0_uu: vec4<f32>,
    u_uv: vec4<f32>,
    v_vv: vec4<f32>,
    normal_inv_d1: vec4<f32>,
}

struct TriangleSection1 {
    material_index: u32,
    normal: vec3<f32>,
}

struct BvhSection0 {
    a_b_node_type: vec3<u32>,
}

struct BvhSection1 {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct Material {
    albedo: vec3<f32>,
}

struct RenderData {
    resolution_x_num_samples_num_bounces_bvh_root_node_index: vec4<u32>,
    camera_position_camera_fov: vec4<f32>,
    camera_rotation_axis_x_inverse_num_samples: vec4<f32>,
    camera_rotation_axis_y_aspect_ratio: vec4<f32>,
    camera_rotation_axis_z_camera_fov_scale: vec4<f32>,
    inverse_resolution_environment_resolution: vec4<f32>,
}

struct HitData {
    hit_distance: f32,
    triangle_index: u32,
}

const PI = 3.141592653589793;
const TAU = PI * 2.;
const INV_PI: f32 = 1. / PI;
const INV_TAU: f32 = 1. / TAU;
const NO_HIT: f32 = 1e30;

var<private> stack: array<u32, 64>;

fn intersect_ray_bounding_box(
    ray_origin: vec3<f32>,
    ray_inverse_direction: vec3<f32>,
    bounding_box_min: vec3<f32>,
    bounding_box_max: vec3<f32>,
    nearest_hit_distance: f32,
) -> f32 {
    let t0 = (bounding_box_min - ray_origin) * ray_inverse_direction;
    let t1 = (bounding_box_max - ray_origin) * ray_inverse_direction;
    let tmin_v = min(t0, t1);
    let tmax_v = max(t0, t1);
    let tmin = max(max(max(tmin_v.x, tmin_v.y), tmin_v.z), 0.);
    let tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);

    return select(NO_HIT, tmin, (tmin <= tmax) && (tmin < nearest_hit_distance));
}

fn intersect_ray_triangle(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    triangle_section0: TriangleSection0,
    nearest_hit_distance: f32,
) -> f32 {
    let normal_inv_d1: vec4<f32> = triangle_section0.normal_inv_d1;
    let v0: vec3<f32> = triangle_section0.v0_uu.xyz;
    let uv: f32 = triangle_section0.u_uv.w;

    let den: f32 = dot(normal_inv_d1.xyz, ray_direction);

    if den >= 0. {
        return NO_HIT;
    }

    let d0: f32 = -dot(v0, normal_inv_d1.xyz);
    let distance: f32 = -(dot(normal_inv_d1.xyz, ray_origin) + d0) / den;

    if distance < 0. || distance > nearest_hit_distance {
        return NO_HIT;
    }

    let w: vec3<f32> = ray_origin + ray_direction * distance - v0;
    let wu: f32 = dot(w, triangle_section0.u_uv.xyz);
    let wv: f32 = dot(w, triangle_section0.v_vv.xyz);
    let s: f32 = (triangle_section0.v_vv.w * wu - uv * wv) * normal_inv_d1.w;
    let t: f32 = (triangle_section0.v0_uu.w * wv - uv * wu) * normal_inv_d1.w;

    return select(distance, NO_HIT, s < 0. || t < 0. || (s + t) > 1.);
}

fn intersect_ray_triangles(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    ray_inverse_direction: vec3<f32>,
) -> HitData {
    let bvh_root_node_index = render_data.resolution_x_num_samples_num_bounces_bvh_root_node_index.w;

    var nearest_hit_distance: f32 = NO_HIT;
    var nearest_hit_data: HitData = HitData(NO_HIT, 0);

    let bvh_root_node_section1: BvhSection1 = bvh_nodes_section1[bvh_root_node_index];
    let bvh_root_node_hit_distance: f32 = intersect_ray_bounding_box(
        ray_origin,
        ray_inverse_direction,
        bvh_root_node_section1.min,
        bvh_root_node_section1.max,
        nearest_hit_distance,
    );

    if bvh_root_node_hit_distance >= NO_HIT {
        return nearest_hit_data;
    }

    var stack_index: u32 = 1;
    stack[0] = bvh_root_node_index;

    loop {
        if stack_index == 0 {
            break;
        }

        stack_index--;
        let node_index: u32 = stack[stack_index];
        let node_section0: BvhSection0 = bvh_nodes_section0[node_index];

        let a: u32 = node_section0.a_b_node_type.x;
        let b: u32 = node_section0.a_b_node_type.y;

        if node_section0.a_b_node_type.z == 0 {
            let left_bvh_node_section1: BvhSection1 = bvh_nodes_section1[a];
            let right_bvh_node_section1: BvhSection1 = bvh_nodes_section1[b];
            let left_hit_distance: f32 = intersect_ray_bounding_box(
                ray_origin,
                ray_inverse_direction,
                left_bvh_node_section1.min,
                left_bvh_node_section1.max,
                nearest_hit_distance
            );
            let right_hit_distance: f32 = intersect_ray_bounding_box(
                ray_origin,
                ray_inverse_direction,
                right_bvh_node_section1.min,
                right_bvh_node_section1.max,
                nearest_hit_distance
            );
            let left_mask: u32 = u32(left_hit_distance < NO_HIT);
            let right_mask: u32 = u32(right_hit_distance < NO_HIT);

            if (left_mask | right_mask) == 0 {
                continue;
            }

            let swap: bool = left_hit_distance > right_hit_distance;
            let first_index: u32 = select(a, b, swap);
            let second_index: u32 = select(b, a, swap);
            let first_mask: u32 = select(left_mask, right_mask, swap);
            let second_mask: u32 = select(right_mask, left_mask, swap);

            stack[stack_index] = second_index;
            stack_index += second_mask;
            stack[stack_index] = first_index;
            stack_index += first_mask;
        } else {
            for (var triangle_index: u32 = a; triangle_index < b; triangle_index++) {
                let hit_distance = intersect_ray_triangle(
                    ray_origin,
                    ray_direction,
                    triangles_section0[triangle_index],
                    nearest_hit_distance,
                );

                if hit_distance < NO_HIT {
                    nearest_hit_distance = hit_distance;
                    nearest_hit_data = HitData(
                        nearest_hit_distance,
                        triangle_index,
                    );
                }
            }
        }
    }

    return nearest_hit_data;
}

fn rand_f32(seed: u32) -> f32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32((word >> 22u) ^ word) * (1. / 4294967296.);
}

fn sample_disk(seed: u32) -> vec2<f32> {
    let r: f32 = sqrt(rand_f32(seed * 2));
    let t: f32 = rand_f32(seed * 2 + 1) * TAU;
    return vec2<f32>(r * cos(t), r * sin(t));
}

fn sample_sphere(seed: u32) -> vec3<f32> {
    let z: f32 = rand_f32(seed * 2) * 2. - 1.;
    let t: f32 = rand_f32(seed * 2 + 1) * 6.2831855;
    let r: f32 = sqrt(1. - z * z);
    return vec3<f32>(r * cos(t), r * sin(t), z);
}

fn render(index: u32) -> vec3<f32> {
    let resolution_x: u32 = render_data.resolution_x_num_samples_num_bounces_bvh_root_node_index.x;
    let inverse_resolution_environment_resolution: vec4<f32> = render_data.inverse_resolution_environment_resolution;
    let camera_rotation: mat3x3<f32> = mat3x3<f32>(
        render_data.camera_rotation_axis_x_inverse_num_samples.xyz,
        render_data.camera_rotation_axis_y_aspect_ratio.xyz,
        render_data.camera_rotation_axis_z_camera_fov_scale.xyz,
    );
    let aspect_ratio: f32 = render_data.camera_rotation_axis_y_aspect_ratio.w;
    let camera_fov_scale: f32 = render_data.camera_rotation_axis_z_camera_fov_scale.w;
    let camera_position: vec3<f32> = render_data.camera_position_camera_fov.xyz;
    let num_samples: u32 = render_data.resolution_x_num_samples_num_bounces_bvh_root_node_index.y;
    let num_bounces: u32 = render_data.resolution_x_num_samples_num_bounces_bvh_root_node_index.z;
    let inverse_num_samples: f32 = render_data.camera_rotation_axis_x_inverse_num_samples.w;

    let x: f32 = f32(index % resolution_x);
    let y: f32 = f32(index / resolution_x);
    let coord: vec2<f32> = vec2<f32>(
        (x + 0.5) * inverse_resolution_environment_resolution.x * 2. - 1.,
        1. - (y + 0.5) * inverse_resolution_environment_resolution.y * 2.,
    );
    let unscaled_focal_plane: vec3<f32> =
        camera_rotation[0] * coord.x * aspect_ratio * camera_fov_scale
            + camera_rotation[1] * coord.y * camera_fov_scale
            + camera_rotation[2] * -1.;
    let base_ray_origin: vec3<f32> = camera_position;
    let base_ray_direction: vec3<f32> = normalize(unscaled_focal_plane);

    var total_incoming_light: vec3<f32> = vec3<f32>(0.);

    for (var sample_index: u32 = 0; sample_index < num_samples; sample_index++) {
        var ray_origin = base_ray_origin + vec3<f32>(sample_disk(index * num_samples + sample_index) * inverse_resolution_environment_resolution.xy, 0.);
        var ray_direction = base_ray_direction;

        var incoming_light: vec3<f32> = vec3<f32>(0.);
        var path_color: vec3<f32> = vec3<f32>(1.);

        for (var bounce_index: u32 = 0; bounce_index <= num_bounces; bounce_index++) {
            let ray_inverse_direction = 1. / ray_direction;

            let hit_data = intersect_ray_triangles(
                ray_origin,
                ray_direction,
                ray_inverse_direction,
            );

            if hit_data.hit_distance < NO_HIT {
                let triangle_section1: TriangleSection1 = triangles_section1[hit_data.triangle_index];
                let material: Material = materials[triangle_section1.material_index];

                ray_origin += ray_direction * (hit_data.hit_distance - 0.0000001);
                ray_direction = normalize(triangle_section1.normal + sample_sphere((index * num_samples + sample_index) * (num_bounces + 1) + bounce_index));

                path_color *= material.albedo;
            } else {
                let azimuth: f32 = atan2(ray_direction.z, ray_direction.x) + PI;
                let polar: f32 = acos(ray_direction.y);
                let x: u32 =
                    u32(azimuth * INV_TAU * (inverse_resolution_environment_resolution.z - 1));
                let y: u32 =
                    u32(polar * INV_PI * (inverse_resolution_environment_resolution.w - 1));

                incoming_light += path_color * environment_buffer[y * u32(inverse_resolution_environment_resolution.z) + x];

                break;
            }
        }

        total_incoming_light += incoming_light;
    }

    return total_incoming_light * inverse_num_samples;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output_buffer[global_id.x] = render(global_id.x);
}
