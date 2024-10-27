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

layout(location = 0) out vec3 viewPosition;

void main() {
    gl_Position = global_data.proj * global_data.view * vec4(inPosition, 1.0);
    viewPosition = (global_data.view * vec4(inPosition, 1.0)).xyz;
}