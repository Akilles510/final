#version 330 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 2) in vec2 vertexUV;
layout(location = 3) in vec3 vertexNormal;

uniform mat4 MVP;
uniform mat4 Model;
uniform mat4 lightSpaceMatrix;

out vec2 uv;
out vec3 worldPos;
out vec3 worldNormal;
out vec4 fragPosLightSpace;

void main() {
    uv = vertexUV;

    vec4 wp = Model * vec4(vertexPosition, 1.0);
    worldPos = wp.xyz;

    worldNormal = mat3(transpose(inverse(Model))) * vertexNormal;

    fragPosLightSpace = lightSpaceMatrix * wp;

    gl_Position = MVP * vec4(vertexPosition, 1.0);
}
