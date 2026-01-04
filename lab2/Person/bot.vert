#version 330 core

// --- INPUTS ---
// These locations must match the "vaa" logic in person.h
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;      // <--- Added (Required for texture)
layout(location = 3) in vec4 jointIndices;  // Changed to vec4 for compatibility
layout(location = 4) in vec4 jointWeights;

// --- OUTPUTS ---
// These go to the Fragment Shader
out vec3 vPosition;
out vec3 vNormal;
out vec2 vTexCoord;                         // <--- Added

// --- UNIFORMS ---
uniform mat4 MVP;
uniform mat4 jointMatrices[50]; // 50 joints max

void main() {
    // 1. Skinning Initialization
    // We start with zero and accumulate the influence of each bone
    vec4 totalLocalPos = vec4(0.0);
    vec4 totalNormal   = vec4(0.0);

    // 2. Iterate through the 4 joints that influence this vertex
    for(int i = 0; i < 4; i++){
        // Convert the joint index to an integer
        int jointIndex = int(jointIndices[i]);
        float weight   = jointWeights[i];

        // Skip if the joint index is invalid (some models use -1)
        if(jointIndex >= 0) {
            // Get the transform matrix for this specific joint
            mat4 jointTransform = jointMatrices[jointIndex];

            // Accumulate Position: (Matrix * Position) * Weight
            vec4 posePosition = jointTransform * vec4(position, 1.0);
            totalLocalPos += posePosition * weight;

            // Accumulate Normal: (Matrix * Normal) * Weight
            // Note: We use 0.0 for w because normals are directions, not positions
            vec4 poseNormal = jointTransform * vec4(normal, 0.0);
            totalNormal += poseNormal * weight;
        }
    }

    // 3. Final Position Calculation
    gl_Position = MVP * totalLocalPos;

    // 4. Pass Data to Fragment Shader
    vPosition = totalLocalPos.xyz;      // World position for lighting
    vNormal   = normalize(totalNormal.xyz); // Normal for lighting
    vTexCoord = texCoord;               // UV coordinates for texturing
}