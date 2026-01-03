#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTexCoords; // Use location 2 for UVs

out vec2 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    TexCoords = aTexCoords;
    // Remove translation from the view matrix so the camera never moves "away" from the skybox
    vec4 pos = projection * mat4(mat3(view)) * vec4(aPos, 1.0);
    // This trick ensures the skybox is always rendered at the maximum depth
    gl_Position = pos.xyww;
}