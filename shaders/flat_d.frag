#version 450

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

layout(location = 0) in vec3 viewPosition;
layout(location = 0) out vec4 outColor;

void main() {
  vec3 tx = dFdx(viewPosition);
  vec3 ty = dFdy(viewPosition);
  vec3 faceNormal = normalize(cross(ty, tx));
  float diff = max(dot(faceNormal, -global_data.light_dir), 0.0);
  vec3 fragColor = (global_data.ambient + global_data.directional * diff) * model_data.Ka_d_.rgb;
  outColor = vec4(fragColor, 1.0);
}