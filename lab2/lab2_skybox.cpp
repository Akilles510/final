#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <render/shader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

static GLFWwindow *window;
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);

// OpenGL camera view parameters
static glm::vec3 eye_center(0.0f, 0.0f, 0.0f);
static glm::vec3 lookat(0, 0, 0);
static glm::vec3 up(0, 1, 0);

// View control for FPS-style camera
static float viewAzimuth = -M_PI / 2.0f;
static float viewPolar = M_PI / 2.0f;

static GLuint LoadSkyboxTexture(const char *texture_file_path) {
    int w, h, channels;
    // Load image, and flip it vertically to match OpenGL's coordinate system
    stbi_set_flip_vertically_on_load(true);
    uint8_t* img = stbi_load(texture_file_path, &w, &h, &channels, 0);
    stbi_set_flip_vertically_on_load(false); // Set back to default

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // For a skybox, we clamp to the edge to prevent seams at the corners
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if (img) {
        GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
        glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, img);
    } else {
        std::cout << "Failed to load texture " << texture_file_path << std::endl;
    }
    stbi_image_free(img);

    return textureID;
}

struct Skybox {
    // Vertex data for a cube
    GLfloat vertex_buffer_data[72] = {
       // Front face (+Z)
       -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
       // Back face (-Z)
       1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
       // Left face (-X)
       -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
       // Right face (+X)
        1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
       // Top face (+Y)
       -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
       // Bottom face (-Y)
       -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    };

    // Index data remains the same
    GLuint index_buffer_data[36] = {
       0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11,
       12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    };

    // UV coordinates carefully mapped to your sky.png layout
    // The image seems to be a 4x3 grid of squares.
    GLfloat uv_buffer_data[48] = {
        // Front face (+Z)
        0.25f, 0.334f, 0.50f, 0.334f, 0.50f, 0.667f, 0.25f, 0.667f,
        // Back face (-Z) - Fully Corrected
        1.00f, 0.334f, 0.75f, 0.334f, 0.75f, 0.667f, 1.00f, 0.667f,
        // Left face (-X)
        0.25f, 0.334f, 0.00f, 0.334f, 0.00f, 0.667f, 0.25f, 0.667f,
        // Right face (+X)
        0.50f, 0.334f, 0.75f, 0.334f, 0.75f, 0.667f, 0.50f, 0.667f,
        // Top face (+Y)
        0.25f, 0.667f, 0.50f, 0.667f, 0.50f, 1.000f, 0.25f, 1.000f,
        // Bottom face (-Y)
        0.25f, 0.000f, 0.50f, 0.000f, 0.50f, 0.334f, 0.25f, 0.334f,
    };

    GLuint vertexArrayID, vertexBufferID, indexBufferID, uvBufferID, textureID;
    GLuint programID;
    GLuint viewMatrixID, projectionMatrixID, textureSamplerID;

    void initialize() {
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

       programID = LoadShadersFromFile("../lab2/skybox.vert", "../lab2/skybox.frag");

       viewMatrixID = glGetUniformLocation(programID, "view");
       projectionMatrixID = glGetUniformLocation(programID, "projection");
       textureSamplerID = glGetUniformLocation(programID, "skybox");

       textureID = LoadSkyboxTexture("../lab2/sky.png");
    }

    void render(glm::mat4 viewMatrix, glm::mat4 projectionMatrix) {
       // Change depth function to pass when depth is less or equal
       // This allows the skybox (drawn at max depth) to be the background
       glDepthFunc(GL_LEQUAL);

       glUseProgram(programID);

       glEnableVertexAttribArray(0); // Vertex positions
       glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
       glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

       glEnableVertexAttribArray(2); // Texture Coordinates
       glBindBuffer(GL_ARRAY_BUFFER, uvBufferID);
       glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);

       glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);

       glUniformMatrix4fv(viewMatrixID, 1, GL_FALSE, &viewMatrix[0][0]);
       glUniformMatrix4fv(projectionMatrixID, 1, GL_FALSE, &projectionMatrix[0][0]);

       glActiveTexture(GL_TEXTURE0);
       glBindTexture(GL_TEXTURE_2D, textureID);
       glUniform1i(textureSamplerID, 0);

       glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (void*)0);

       glDisableVertexAttribArray(0);
       glDisableVertexAttribArray(2);

       // Set depth function back to default
       glDepthFunc(GL_LESS);
    }

    void cleanup() {
       glDeleteBuffers(1, &vertexBufferID);
       glDeleteBuffers(1, &uvBufferID);
       glDeleteBuffers(1, &indexBufferID);
       glDeleteTextures(1, &textureID);
       glDeleteVertexArrays(1, &vertexArrayID);
       glDeleteProgram(programID);
    }
};

int main(void)
{
    if (!glfwInit()) {
       std::cerr << "Failed to initialize GLFW." << std::endl;
       return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1024, 768, "Lab 2 - Skybox", NULL, NULL);
    if (window == NULL) {
       std::cerr << "Failed to open a GLFW window." << std::endl;
       glfwTerminate();
       return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetKeyCallback(window, key_callback);

    if (gladLoadGL(glfwGetProcAddress) == 0) {
       std::cerr << "Failed to initialize OpenGL context." << std::endl;
       return -1;
    }

    glClearColor(0.2f, 0.2f, 0.25f, 0.0f);

    glEnable(GL_DEPTH_TEST);
    // No need to cull faces or change front face for skybox
    // The cube is rendered normally, and the shader trick places it "around" the camera

    Skybox skybox;
    skybox.initialize();

    glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);

    do {
       glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

       glm::vec3 direction;
       direction.x = cos(viewAzimuth) * sin(viewPolar);
       direction.y = cos(viewPolar);
       direction.z = sin(viewAzimuth) * sin(viewPolar);
       lookat = eye_center + glm::normalize(direction);

       glm::mat4 viewMatrix = glm::lookAt(eye_center, lookat, up);

       skybox.render(viewMatrix, projectionMatrix);

       glfwSwapBuffers(window);
       glfwPollEvents();
    } while (!glfwWindowShouldClose(window));

    skybox.cleanup();
    glfwTerminate();
    return 0;
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
       viewAzimuth = -M_PI / 2.0f;
       viewPolar = M_PI / 2.0f;
       std::cout << "Camera Reset." << std::endl;
    }

    if (key == GLFW_KEY_UP && (action == GLFW_REPEAT || action == GLFW_PRESS)) viewPolar -= 0.05f;
    if (key == GLFW_KEY_DOWN && (action == GLFW_REPEAT || action == GLFW_PRESS)) viewPolar += 0.05f;
    if (key == GLFW_KEY_LEFT && (action == GLFW_REPEAT || action == GLFW_PRESS)) viewAzimuth -= 0.05f;
    if (key == GLFW_KEY_RIGHT && (action == GLFW_REPEAT || action == GLFW_PRESS)) viewAzimuth += 0.05f;

    // Clamp polar angle to prevent camera flipping
    const float polar_margin = 0.01f;
    if (viewPolar < polar_margin) viewPolar = polar_margin;
    if (viewPolar > M_PI - polar_margin) viewPolar = M_PI - polar_margin;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
       glfwSetWindowShouldClose(window, GL_TRUE);
}