#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <glm/gtc/type_ptr.hpp>   // glm::value_ptr, glm::make_mat4/make_vec3/make_quat
#include <glm/gtx/quaternion.hpp> // glm::quat, glm::slerp, toMat4, etc.

#include "render/shader.h"

//#include <tinygltf-2.9.3/tiny_gltf.h>
// TinyGLTF + stb (LOAD ONLY)

// Stop tinygltf from pulling its bundled stb headers
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE

// Also disable stb_image_write usage inside tinygltf
#define TINYGLTF_NO_STB_IMAGE_WRITE

// Provide ONE stb_image implementation (for texture loading)
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

// TinyGLTF implementation
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
// ------------------------------------------------------------


#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef BUFFER_OFFSET
#define BUFFER_OFFSET(i) ((char*)nullptr + (i))
#endif

#include "Person/person.h"


// ---------- Shadow map globals ----------
static GLuint depthMapFBO = 0;
static GLuint depthMap = 0;
static const unsigned int SHADOW_W = 2048;
static const unsigned int SHADOW_H = 2048;

// Depth-only shader program
static GLuint depthProgram = 0;
static GLuint depthLightSpaceID = 0;
static GLuint depthModelID = 0;

// Forward declaration
static void initShadowMap();

static GLFWwindow* window;
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

// ------------------------
// Camera (walk + turn)
// ------------------------
static glm::vec3 eye_center;
static glm::vec3 lookat;
static glm::vec3 up(0, 1, 0);

// Player-style movement (so the foreground can feel "infinite")
static glm::vec3 playerPos(0.0f, 0.0f, 0.0f);
static float yaw = 0.0f; // radians, turn left/right

// ------------------------
// Texture loaders
// ------------------------
static GLuint LoadTextureTileBox(const char* texture_file_path) {
    int w, h, channels;
    uint8_t* img = stbi_load(texture_file_path, &w, &h, &channels, 3);
    if (!img) {
        std::cout << "Failed to load texture " << texture_file_path << std::endl;
        return 0;
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Tile on box
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(img);
    return texture;
}

static GLuint LoadSkyboxTexture(const char* texture_file_path) {
    int w, h, channels;
    // Lab2 skybox often flips; keep consistent with your lab2_skybox.cpp
    stbi_set_flip_vertically_on_load(true);
    uint8_t* img = stbi_load(texture_file_path, &w, &h, &channels, 3);
    if (!img) {
        std::cout << "Failed to load skybox texture " << texture_file_path << std::endl;
        return 0;
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Avoid seams
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img);

    stbi_image_free(img);
    return texture;
}

static void initShadowMap() {
    glGenFramebuffers(1, &depthMapFBO);

    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                 SHADOW_W, SHADOW_H, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Important to reduce border artifacts
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


// ============================================================
// Building (Lab2 city)  +  (Point 3) normals buffer added
// ============================================================
struct Building {
    glm::vec3 position;
    glm::vec3 scale;

    // Vertex definition for a box on the XZ plane (same as your Lab 2)
    GLfloat vertex_buffer_data[72] = {
        // Front face
        -1.0f, 0.0f,  1.0f,
         1.0f, 0.0f,  1.0f,
         1.0f, 2.0f,  1.0f,
        -1.0f, 2.0f,  1.0f,

        // Back face
         1.0f, 0.0f, -1.0f,
        -1.0f, 0.0f, -1.0f,
        -1.0f, 2.0f, -1.0f,
         1.0f, 2.0f, -1.0f,

        // Left face
        -1.0f, 0.0f, -1.0f,
        -1.0f, 0.0f,  1.0f,
        -1.0f, 2.0f,  1.0f,
        -1.0f, 2.0f, -1.0f,

        // Right face
         1.0f, 0.0f,  1.0f,
         1.0f, 0.0f, -1.0f,
         1.0f, 2.0f, -1.0f,
         1.0f, 2.0f,  1.0f,

        // Top face
        -1.0f, 2.0f,  1.0f,
         1.0f, 2.0f,  1.0f,
         1.0f, 2.0f, -1.0f,
        -1.0f, 2.0f, -1.0f,

        // Bottom face
        -1.0f, 0.0f, -1.0f,
         1.0f, 0.0f, -1.0f,
         1.0f, 0.0f,  1.0f,
        -1.0f, 0.0f,  1.0f,
    };

    // Neutral color (kept for your current shader)
    GLfloat color_buffer_data[72] = {
        0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f,  0.5f, 0.5f, 0.5f
    };

    // UVs (same idea as Lab 2; tile V a bit)
    GLfloat uv_buffer_data[48] = {
        // Front
        0.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,
        // Back
        0.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,
        // Left
        0.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,
        // Right
        0.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,
        // Top (not used much)
        0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
        // Bottom (not used much)
        0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,  0.0f, 0.0f,
    };

    // (Point 3) Normals per face (24 vertices)
    // Front:  (0,0,1), Back: (0,0,-1), Left: (-1,0,0), Right: (1,0,0), Top: (0,1,0), Bottom: (0,-1,0)
    GLfloat normal_buffer_data[72] = {
        // Front
         0.f, 0.f,  1.f,   0.f, 0.f,  1.f,   0.f, 0.f,  1.f,   0.f, 0.f,  1.f,
        // Back
         0.f, 0.f, -1.f,   0.f, 0.f, -1.f,   0.f, 0.f, -1.f,   0.f, 0.f, -1.f,
        // Left
        -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,  -1.f, 0.f,  0.f,
        // Right
         1.f, 0.f,  0.f,   1.f, 0.f,  0.f,   1.f, 0.f,  0.f,   1.f, 0.f,  0.f,
        // Top
         0.f, 1.f,  0.f,   0.f, 1.f,  0.f,   0.f, 1.f,  0.f,   0.f, 1.f,  0.f,
        // Bottom
         0.f,-1.f,  0.f,   0.f,-1.f,  0.f,   0.f,-1.f,  0.f,   0.f,-1.f,  0.f,
    };

    GLuint index_buffer_data[36] = {
        0, 1, 2,   0, 2, 3,        // Front
        4, 5, 6,   4, 6, 7,        // Back
        8, 9, 10,  8, 10, 11,      // Left
        12, 13, 14, 12, 14, 15,    // Right
        16, 17, 18, 16, 18, 19,    // Top
        20, 21, 22, 20, 22, 23     // Bottom
    };

    GLuint vertexArrayID = 0;
    GLuint vertexBufferID = 0;
    GLuint colorBufferID  = 0;
    GLuint uvBufferID     = 0;
    GLuint normalBufferID = 0; // (Point 3)
    GLuint indexBufferID  = 0;

    GLuint programID = 0;
    GLuint mvpMatrixID = 0;
    GLuint textureSamplerID = 0;
    GLuint textureID = 0;
    GLuint modelMatrixID;
    GLuint lightPosID;
    GLuint lightIntID;
    GLuint lightSpaceMatrixID;
    GLuint shadowMapID;

    void initialize(glm::vec3 position, glm::vec3 scale, const char* texture_path) {
        this->position = position;
        this->scale    = scale;

        glGenVertexArrays(1, &vertexArrayID);
        glBindVertexArray(vertexArrayID);

        glGenBuffers(1, &vertexBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);

        glGenBuffers(1, &colorBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(color_buffer_data), color_buffer_data, GL_STATIC_DRAW);

        // Tile V coordinate (same idea as Lab 2)
        for (int i = 0; i < 24; ++i) uv_buffer_data[2 * i + 1] *= 5;

        glGenBuffers(1, &uvBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, uvBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uv_buffer_data), uv_buffer_data, GL_STATIC_DRAW);

        // (Point 3) Normal buffer
        glGenBuffers(1, &normalBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, normalBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(normal_buffer_data), normal_buffer_data, GL_STATIC_DRAW);

        glGenBuffers(1, &indexBufferID);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index_buffer_data), index_buffer_data, GL_STATIC_DRAW);

        // Same shader as your Lab 2 buildings (texture mapping)
        programID = LoadShadersFromFile("lab2/lit_box.vert", "lab2/lit_box.frag");
        mvpMatrixID = glGetUniformLocation(programID, "MVP");
        modelMatrixID = glGetUniformLocation(programID, "Model");
        textureSamplerID = glGetUniformLocation(programID, "textureSampler");
        lightPosID = glGetUniformLocation(programID, "lightPosition");
        lightIntID = glGetUniformLocation(programID, "lightIntensity");
        lightSpaceMatrixID = glGetUniformLocation(programID, "lightSpaceMatrix");
        shadowMapID = glGetUniformLocation(programID, "shadowMap");


        textureID = LoadTextureTileBox(texture_path);

        glBindVertexArray(0);
    }

    void render(const glm::mat4& vp, const glm::mat4& lightSpaceMatrix, GLuint depthMapTex) {
        glUseProgram(programID);
        glUniform3fv(glGetUniformLocation(programID, "cameraPos"), 1, &eye_center[0]);

        glUniform3f(glGetUniformLocation(programID, "fogColor"),
                    0.08f, 0.10f, 0.14f);   // tweak if you want

        glUniform1f(glGetUniformLocation(programID, "fogDensity"),
                    0.000001f);           // tweak if too strong/weak

        glBindVertexArray(vertexArrayID);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model = glm::scale(model, scale);
        glm::mat4 mvp = vp * model;

        glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, &mvp[0][0]);
        glUniformMatrix4fv(modelMatrixID, 1, GL_FALSE, &model[0][0]);
        glUniformMatrix4fv(lightSpaceMatrixID, 1, GL_FALSE, &lightSpaceMatrix[0][0]);


        glm::vec3 lightPos(0.0f, 500.0f, 0.0f);
        glm::vec3 lightIntensity(50.0f, 50.0f, 50.0f);
        glUniform3f(lightPosID, lightPos.x, lightPos.y, lightPos.z);
        glUniform3f(lightIntID, lightIntensity.x, lightIntensity.y, lightIntensity.z);

        // Position
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        // Color
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        // UV
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, uvBufferID);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        // (Point 3) Normals attribute (location 3)
        // Not used by Lab 2 shader yet, but ready for Lab 3 lighting merge.
        glEnableVertexAttribArray(3);
        glBindBuffer(GL_ARRAY_BUFFER, normalBufferID);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        // Indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);

        // Texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(textureSamplerID, 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthMapTex);
        glUniform1i(shadowMapID, 1);


        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (void*)0);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);

        glBindVertexArray(0);
    }

    void renderDepth(GLuint depthProgram, GLuint depthModelID) {
        glUseProgram(depthProgram);
        glBindVertexArray(vertexArrayID);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, position);
        model = glm::scale(model, scale);

        glUniformMatrix4fv(depthModelID, 1, GL_FALSE, &model[0][0]);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (void*)0);

        glDisableVertexAttribArray(0);
        glBindVertexArray(0);
    }


    void cleanup() {
        glDeleteBuffers(1, &vertexBufferID);
        glDeleteBuffers(1, &colorBufferID);
        glDeleteBuffers(1, &uvBufferID);
        glDeleteBuffers(1, &normalBufferID);
        glDeleteBuffers(1, &indexBufferID);
        glDeleteVertexArrays(1, &vertexArrayID);
        glDeleteProgram(programID);
        glDeleteTextures(1, &textureID);
    }
};

// ============================================================
// Skybox (from lab2_skybox.cpp, integrated)
// ============================================================
struct Skybox {
    GLfloat vertex_buffer_data[72] = {
        // Front (+Z)
        -1.0f,-1.0f, 1.0f,  1.0f,-1.0f, 1.0f,  1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
        // Back (-Z)
         1.0f,-1.0f,-1.0f, -1.0f,-1.0f,-1.0f, -1.0f, 1.0f,-1.0f,  1.0f, 1.0f,-1.0f,
        // Left (-X)
        -1.0f,-1.0f,-1.0f, -1.0f,-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,-1.0f,
        // Right (+X)
         1.0f,-1.0f, 1.0f,  1.0f,-1.0f,-1.0f,  1.0f, 1.0f,-1.0f,  1.0f, 1.0f, 1.0f,
        // Top (+Y)
        -1.0f, 1.0f, 1.0f,  1.0f, 1.0f, 1.0f,  1.0f, 1.0f,-1.0f, -1.0f, 1.0f,-1.0f,
        // Bottom (-Y)
        -1.0f,-1.0f,-1.0f,  1.0f,-1.0f,-1.0f,  1.0f,-1.0f, 1.0f, -1.0f,-1.0f, 1.0f,
    };

    GLuint index_buffer_data[36] = {
        0,1,2, 0,2,3,  4,5,6, 4,6,7,  8,9,10, 8,10,11,
        12,13,14, 12,14,15,  16,17,18, 16,18,19,  20,21,22, 20,22,23
    };

    // UVs mapped for your skybox layout (keep same as your lab2_skybox.cpp)
    GLfloat uv_buffer_data[48] = {
        // Front (+Z)
        0.25f, 0.6667f,  0.50f, 0.6667f,  0.50f, 0.3333f,  0.25f, 0.3333f,
        // Back (-Z)
        0.75f, 0.6667f,  1.00f, 0.6667f,  1.00f, 0.3333f,  0.75f, 0.3333f,
        // Left (-X)
        0.00f, 0.6667f,  0.25f, 0.6667f,  0.25f, 0.3333f,  0.00f, 0.3333f,
        // Right (+X)
        0.50f, 0.6667f,  0.75f, 0.6667f,  0.75f, 0.3333f,  0.50f, 0.3333f,
        // Top (+Y)
        0.25f, 1.0000f,  0.50f, 1.0000f,  0.50f, 0.6667f,  0.25f, 0.6667f,
        // Bottom (-Y)
        0.25f, 0.3333f,  0.50f, 0.3333f,  0.50f, 0.0000f,  0.25f, 0.0000f
    };

    GLuint vertexArrayID = 0;
    GLuint vertexBufferID = 0;
    GLuint uvBufferID = 0;
    GLuint indexBufferID = 0;

    GLuint programID = 0;
    GLuint viewMatrixID = 0;
    GLuint projMatrixID = 0;
    GLuint textureSamplerID = 0;
    GLuint textureID = 0;

    void initialize(const char* sky_texture_path) {
        glGenVertexArrays(1, &vertexArrayID);
        glBindVertexArray(vertexArrayID);

        glGenBuffers(1, &vertexBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);

        glGenBuffers(1, &uvBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, uvBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uv_buffer_data), uv_buffer_data, GL_STATIC_DRAW);

        glGenBuffers(1, &indexBufferID);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index_buffer_data), index_buffer_data, GL_STATIC_DRAW);

        programID = LoadShadersFromFile("lab2/skybox.vert", "lab2/skybox.frag");
        viewMatrixID = glGetUniformLocation(programID, "view");
        projMatrixID = glGetUniformLocation(programID, "projection");
        textureSamplerID = glGetUniformLocation(programID, "textureSampler");

        textureID = LoadSkyboxTexture(sky_texture_path);

        glBindVertexArray(0);
    }

    void render(const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix) {
        glDepthFunc(GL_LEQUAL);

        glUseProgram(programID);

        // Remove translation from view so skybox stays centered
        glm::mat4 viewNoTranslate = glm::mat4(glm::mat3(viewMatrix));

        glUniformMatrix4fv(viewMatrixID, 1, GL_FALSE, &viewNoTranslate[0][0]);
        glUniformMatrix4fv(projMatrixID, 1, GL_FALSE, &projectionMatrix[0][0]);

        glBindVertexArray(vertexArrayID);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, uvBufferID);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(textureSamplerID, 0);

        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (void*)0);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(2);

        glBindVertexArray(0);

        glDepthFunc(GL_LESS);
    }

    void cleanup() {
        glDeleteBuffers(1, &vertexBufferID);
        glDeleteBuffers(1, &uvBufferID);
        glDeleteBuffers(1, &indexBufferID);
        glDeleteVertexArrays(1, &vertexArrayID);
        glDeleteProgram(programID);
        glDeleteTextures(1, &textureID);
    }
};

// ============================================================
// Helpers (random + camera)
// ============================================================
static float rand01() {
    return float(rand()) / float(RAND_MAX);
}

static void update_camera_walk() {
    // Simple "walk + turn" camera.
    glm::vec3 forward(cos(yaw), 0.0f, sin(yaw));
    eye_center = playerPos + glm::vec3(0.0f, 120.0f, 0.0f);
    lookat = eye_center + forward * 250.0f;
}

int main() {
    if (!glfwInit()) {
        std::cout << "Failed to init GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(1024, 768, "Final (Lab2 city + skybox + normals-ready)", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        std::cerr << "Failed to initialize OpenGL context." << std::endl;
        return -1;
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    MyBot robot;
    robot.initialize();


    // Shadow map setup
    initShadowMap();
    depthProgram = LoadShadersFromFile("lab2/shadow_depth.vert", "lab2/shadow_depth.frag");
    depthLightSpaceID = glGetUniformLocation(depthProgram, "lightSpaceMatrix");
    depthModelID = glGetUniformLocation(depthProgram, "Model");

    // Random seed (for procedural placement)
    srand(12345);

    // Initial camera/player state
    playerPos = glm::vec3(0.0f, 0.0f, 0.0f);
    yaw = 0.0f;
    update_camera_walk();

    // --- Create skybox ---
    Skybox sky;
    // Adjust the path if your sky texture is different:
    sky.initialize("lab2/skyNeb.png");

    // --- Create buildings (example: keep your procedural city logic here) ---
    std::vector<Building> buildings;

    // Example layout (replace with your existing Lab 2 generation)
    const char* facades[] = {
        "lab2/skin.png",
        "lab2/skin2.png",
        "lab2/skin3.png",
        "lab2/skin4.png"
    };

    // "Minecraft-ish" infinite illusion: keep a fixed pool of buildings
    // and recycle them around the player as they move.
    const int BUILDING_COUNT = 220;
    const float SPAWN_RADIUS = 900.0f;   // initial scatter radius
    const float BASE_SIZE = 16.0f;

    buildings.reserve(BUILDING_COUNT);
    for (int i = 0; i < BUILDING_COUNT; ++i) {
        Building b;
        float angle = rand01() * 2.0f * float(M_PI);
        float r = rand01() * SPAWN_RADIUS;
        float x = cos(angle) * r;
        float z = sin(angle) * r;

        float h = 35.0f + rand01() * 120.0f;
        int texIdx = int(rand01() * 4.0f) % 4;

        b.initialize(glm::vec3(x, 0.0f, z),
                     glm::vec3(BASE_SIZE, h, BASE_SIZE),
                     facades[texIdx]);
        buildings.push_back(b);
    }

    // Ground plane (separate, so we don't recycle it)
    Building ground;
    ground.initialize(glm::vec3(0.0f, 0.0f, 0.0f),
                      glm::vec3(4000.0f, 2.0f, 4000.0f),
                      "lab2/facade0.jpg");

    double lastTime = glfwGetTime();

    // Infinite illusion parameters
    const float ACTIVE_RADIUS = 900.0f;   // if a building is outside this radius, recycle it
    const float AHEAD_MIN = 700.0f;       // how far ahead to respawn recycled buildings
    const float AHEAD_RAND = 500.0f;
    const float SIDE_RANGE = 800.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ------------------------------------------------------------
        // Time step + input (walk + turn)
        // ------------------------------------------------------------
        double now = glfwGetTime();
        float dt = float(now - lastTime);
        lastTime = now;

        const float moveSpeed = 300.0f;   // world units per second
        const float turnSpeed = 1.8f;     // radians per second

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) yaw -= turnSpeed * dt;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) yaw += turnSpeed * dt;

        glm::vec3 forward(cos(yaw), 0.0f, sin(yaw));
        glm::vec3 right(-forward.z, 0.0f, forward.x);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) playerPos += forward * (moveSpeed * dt);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) playerPos -= forward * (moveSpeed * dt);

        // Optional: strafe with Q/E (handy for demo videos)
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) playerPos -= right * (moveSpeed * dt);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) playerPos += right * (moveSpeed * dt);

        update_camera_walk();

        // ------------------------------------------------------------
        // Recycle buildings to create an "infinite" foreground
        // ------------------------------------------------------------
        for (auto &b : buildings) {
            glm::vec2 d(b.position.x - playerPos.x, b.position.z - playerPos.z);
            if (glm::length(d) > ACTIVE_RADIUS) {
                float ahead = AHEAD_MIN + rand01() * AHEAD_RAND;
                float side = (rand01() * 2.0f - 1.0f) * SIDE_RANGE;
                b.position = playerPos + forward * ahead + right * side;

                // Change the height a bit to avoid repeating patterns
                b.scale.y = 35.0f + rand01() * 120.0f;
            }
        }
        // ---------- PASS A: render depth map ----------
        glm::vec3 lightPos(200.0f, 600.0f, 200.0f);
        glm::vec3 lightTarget(0.0f, 0.0f, 0.0f);

        glm::mat4 lightView = glm::lookAt(lightPos, lightTarget, glm::vec3(0, 1, 0));
        float orthoSize = 600.0f;
        glm::mat4 lightProj = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 1.0f, 2000.0f);
        glm::mat4 lightSpaceMatrix = lightProj * lightView;

        glViewport(0, 0, SHADOW_W, SHADOW_H);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);

        glUseProgram(depthProgram);
        glUniformMatrix4fv(depthLightSpaceID, 1, GL_FALSE, &lightSpaceMatrix[0][0]);

        for (auto& b : buildings) {
            b.renderDepth(depthProgram, depthModelID);
        }
        ground.renderDepth(depthProgram, depthModelID);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);


        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Camera matrices
        glm::mat4 viewMatrix = glm::lookAt(eye_center, lookat, up);
        glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 2000.0f);
        glm::mat4 vp = projectionMatrix * viewMatrix;

        glm::mat4 modelMatrix = glm::mat4(1.0f);                 // identity
        modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, 0.0f, 0.0f));
        // modelMatrix = glm::scale(modelMatrix, glm::vec3(1.0f)); // optional

        glm::vec3 lightInt = glm::vec3(50.0f, 50.0f, 50.0f);

        // NO robot.update(time) needed for static pose
        modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, 0.0f, 0.0f));
        modelMatrix = glm::scale(modelMatrix, glm::vec3(50.0f));  // BIG

        robot.render(viewMatrix, projectionMatrix, modelMatrix, lightPos, glm::vec3(50.0f));

        //robot.render(viewMatrix, projectionMatrix, modelMatrix, lightPos, lightInt);


        // Render skybox first (as background)
        sky.render(viewMatrix, projectionMatrix);

        // Render buildings + ground
        for (auto& b : buildings) b.render(vp, lightSpaceMatrix, depthMap);
        ground.render(vp, lightSpaceMatrix, depthMap);
        


        glfwSwapBuffers(window);
    }

    for (auto& b : buildings) b.cleanup();
    ground.cleanup();
    sky.cleanup();
    robot.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// ------------------------
// Input (same controls as your lab2_building style)
// ------------------------
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    (void)scancode; (void)mode;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    // Escape to quit
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GL_TRUE);
        return;
    }
}
