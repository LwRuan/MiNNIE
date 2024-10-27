#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(binding = 0) uniform GlobalUniformData {
    mat4 proj;
    mat4 view;
    vec3 ambient;
    vec3 directional;
    vec3 light_dir;
} global_data;

layout(binding = 1) uniform ModelUniformData {
  vec4 Ka_d_;
  vec4 Kd_;
  vec4 Ks_Ns_;
  mat4 model_; // don't use
} model_data;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = global_data.proj * global_data.view * vec4(inPosition, 1.0);
    float diff = max(dot(inNormal, -global_data.light_dir), 0.0);
    fragColor = (global_data.ambient + global_data.directional * diff) * model_data.Ka_d_.rgb;
}