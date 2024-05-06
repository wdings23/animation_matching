struct Locals {
    transform: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> r_locals: Locals;

@group(0) @binding(1)
var<uniform> skinning_matrices0: array<mat4x4<f32>, 16>;

@group(0) @binding(2)
var<uniform> skinning_matrices1: array<mat4x4<f32>, 16>;

@group(0) @binding(3)
var<uniform> skinning_matrices2: array<mat4x4<f32>, 16>;

@group(0) @binding(4)
var<uniform> skinning_matrices3: array<mat4x4<f32>, 16>;

@group(0) @binding(5)
var<uniform> skinning_matrices4: array<mat4x4<f32>, 16>;

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) color : vec4<f32>,
    @location(3) joint_weights : vec4<f32>,
    @location(4) joint_indices : vec4<f32>
};
struct VertexOutput {
    @location(0) texcoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) color: vec4<f32>
};
struct FragmentOutput {
    @location(0) color_output : vec4f,
};


@vertex
fn vs_main(in: VertexInput) -> VertexOutput {

    let joint_index0: u32 = u32(floor(in.joint_indices.x + 0.5));
    let joint_index1: u32 = u32(floor(in.joint_indices.y + 0.5));
    let joint_index2: u32 = u32(floor(in.joint_indices.z + 0.5));
    let joint_index3: u32 = u32(floor(in.joint_indices.w + 0.5)); 

    let weight0: f32 = in.joint_weights.x;
    let weight1: f32 = in.joint_weights.y;
    let weight2: f32 = in.joint_weights.z;
    let weight3: f32 = in.joint_weights.w;

    let matrix_index0: u32 = joint_index0 / 16;
    let matrix_index1: u32 = joint_index1 / 16;
    let matrix_index2: u32 = joint_index2 / 16;
    let matrix_index3: u32 = joint_index3 / 16;

    let index0: u32 = joint_index0 % 16;
    let index1: u32 = joint_index1 % 16;
    let index2: u32 = joint_index2 % 16;
    let index3: u32 = joint_index3 % 16;

    var anim_matrix0: mat4x4<f32> = mat4x4<f32>();
    if(matrix_index0 == 0)
    {
        anim_matrix0 = skinning_matrices0[index0];
    }
    else if(matrix_index0 == 1)
    {
        anim_matrix0 = skinning_matrices1[index0];
    }
    else if(matrix_index0 == 2)
    {
        anim_matrix0 = skinning_matrices2[index0];
    }
    else if(matrix_index0 == 3)
    {
        anim_matrix0 = skinning_matrices3[index0];
    }
    else if(matrix_index0 == 4)
    {
        anim_matrix0 = skinning_matrices4[index0];
    }

    var anim_matrix1: mat4x4<f32> = mat4x4<f32>();
    if(matrix_index1 == 0)
    {
        anim_matrix1 = skinning_matrices0[index1];
    }
    else if(matrix_index1 == 1)
    {
        anim_matrix1 = skinning_matrices1[index1];
    }
    else if(matrix_index1 == 2)
    {
        anim_matrix1 = skinning_matrices2[index1];
    }
    else if(matrix_index1 == 3)
    {
        anim_matrix1 = skinning_matrices3[index1];
    }
    else if(matrix_index1 == 4)
    {
        anim_matrix1 = skinning_matrices4[index1];
    }

    var anim_matrix2: mat4x4<f32> = mat4x4<f32>();
    if(matrix_index2 == 0)
    {
        anim_matrix2 = skinning_matrices0[index2];
    }
    else if(matrix_index2 == 1)
    {
        anim_matrix2 = skinning_matrices1[index2];
    }
    else if(matrix_index2 == 2)
    {
        anim_matrix2 = skinning_matrices2[index2];
    }
    else if(matrix_index2 == 3)
    {
        anim_matrix2 = skinning_matrices3[index2];
    }
    else if(matrix_index2 == 4)
    {
        anim_matrix2 = skinning_matrices4[index2];
    }

    var anim_matrix3: mat4x4<f32> = mat4x4<f32>();
    if(matrix_index3 == 0)
    {
        anim_matrix3 = skinning_matrices0[index3];
    }
    else if(matrix_index3 == 1)
    {
        anim_matrix3 = skinning_matrices1[index3];
    }
    else if(matrix_index3 == 2)
    {
        anim_matrix3 = skinning_matrices2[index3];
    }
    else if(matrix_index3 == 3)
    {
        anim_matrix3 = skinning_matrices3[index3];
    }
    else if(matrix_index3 == 4)
    {
        anim_matrix3 = skinning_matrices4[index3];
    }

    let vert0: vec4<f32> = (vec4<f32>(in.pos.xyz, 1.0) * anim_matrix0) * weight0;
    let vert1: vec4<f32> = (vec4<f32>(in.pos.xyz, 1.0) * anim_matrix1) * weight1;
    let vert2: vec4<f32> = (vec4<f32>(in.pos.xyz, 1.0) * anim_matrix2) * weight2;
    let vert3: vec4<f32> = (vec4<f32>(in.pos.xyz, 1.0) * anim_matrix3) * weight3;

    //let vert0: vec4<f32> = (anim_matrix0 * vec4<f32>(in.pos.xyz, 1.0)) * weight0;
    //let vert1: vec4<f32> = (anim_matrix1 * vec4<f32>(in.pos.xyz, 1.0)) * weight1;
    //let vert2: vec4<f32> = (anim_matrix2 * vec4<f32>(in.pos.xyz, 1.0)) * weight2;
    //let vert3: vec4<f32> = (anim_matrix3 * vec4<f32>(in.pos.xyz, 1.0)) * weight3;

    let total_xform: vec4<f32> = vec4<f32>(vert0.xyz + vert1.xyz + vert2.xyz + vert3.xyz, 1.0);

    var out: VertexOutput;
    out.pos = total_xform * r_locals.transform;
    out.texcoord = in.texcoord;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    var dp: f32 = dot(in.color.xyz, normalize(vec3<f32>(-1.0, 1.0, -1.0)));
    if(dp < 0.0)
    {
        dp = 0.0;
    }

    var ambientShade: f32 = 0.2;
    out.color_output = vec4<f32>(dp + ambientShade, dp + ambientShade, dp + ambientShade, 1.0);

    return out;
}