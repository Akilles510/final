#version 330 core

in vec2 uv;
in vec3 worldPos;
in vec3 worldNormal;
in vec4 fragPosLightSpace;

uniform sampler2D textureSampler;
uniform sampler2D shadowMap;

uniform vec3 lightPosition;
uniform vec3 lightIntensity;

out vec3 finalColor;

float ShadowFactor(vec4 fragPosLS, vec3 normal, vec3 lightDir)
{
    // Perspective divide
    vec3 projCoords = fragPosLS.xyz / fragPosLS.w;
    // To [0,1]
    projCoords = projCoords * 0.5 + 0.5;

    // Outside shadow map -> treat as lit
    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0 ||
        projCoords.z < 0.0 || projCoords.z > 1.0)
        return 1.0;

    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;

    // Bias reduces shadow acne; slope-scaled bias helps
    float bias = max(0.002 * (1.0 - dot(normal, lightDir)), 0.0008);

    // Simple hard shadow
    return (currentDepth - bias > closestDepth) ? 0.25 : 1.0;
}

void main() {
    vec3 albedo = texture(textureSampler, uv).rgb;

    vec3 N = normalize(worldNormal);
    vec3 L = normalize(lightPosition - worldPos);
    float NdotL = max(dot(N, L), 0.0);

    float shadow = ShadowFactor(fragPosLightSpace, N, L);

    vec3 ambient = 0.35 * albedo;
    vec3 diffuse = 0.85 * albedo * NdotL * shadow;

    finalColor = ambient + diffuse;
}
