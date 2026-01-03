#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

// GLTF model loader
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include <render/shader.h>

#include <vector>
#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <math.h>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

static GLFWwindow *window;
static int windowWidth = 1024;
static int windowHeight = 768;

// Input callbacks
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);

// ----------------------------------------------------------------------------
// GLOBAL VARIABLES
// ----------------------------------------------------------------------------
// Camera - GLOBAL so key_callback can see it
static glm::vec3 eye_center(0.0f, 150.0f, 600.0f); // Start position
static glm::vec3 lookat(0.0f, 50.0f, 0.0f);        // Look at the robot's waist
static glm::vec3 up(0.0f, 1.0f, 0.0f);

static float FoV = 45.0f;
static float zNear = 10.0f;     
static float zFar = 2000.0f;    

// Lighting  
static glm::vec3 lightIntensity(5e6f, 5e6f, 5e6f);
static glm::vec3 lightPosition(-275.0f, 500.0f, 800.0f);

// Animation State
static bool playAnimation = true;
static float playbackSpeed = 2.0f;

// ----------------------------------------------------------------------------
// SKYBOX STRUCT
// ----------------------------------------------------------------------------
static GLuint LoadSkyboxTexture(const char *texture_file_path) {
    int w, h, channels;
    stbi_set_flip_vertically_on_load(true);
    uint8_t* img = stbi_load(texture_file_path, &w, &h, &channels, 0);
    stbi_set_flip_vertically_on_load(false);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if (img) {
        GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
        glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, img);
    } else {
        std::cout << "FAILED to load texture: " << texture_file_path << std::endl;
    }
    stbi_image_free(img);

    return textureID;
}

struct Skybox {
    GLfloat vertex_buffer_data[72] = {
       -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
        1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
       -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
        1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
       -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
       -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    };

    GLuint index_buffer_data[36] = {
       0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11,
       12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23,
    };

    GLfloat uv_buffer_data[48] = {
        0.25f, 0.333f, 0.50f, 0.333f, 0.50f, 0.666f, 0.25f, 0.666f,
        0.75f, 0.333f, 1.00f, 0.333f, 1.00f, 0.666f, 0.75f, 0.666f,
        0.00f, 0.333f, 0.25f, 0.333f, 0.25f, 0.666f, 0.00f, 0.666f,
        0.50f, 0.333f, 0.75f, 0.333f, 0.75f, 0.666f, 0.50f, 0.666f,
        0.25f, 0.666f, 0.50f, 0.666f, 0.50f, 1.000f, 0.25f, 1.000f,
        0.25f, 0.000f, 0.50f, 0.000f, 0.50f, 0.333f, 0.25f, 0.333f,
    };

    GLuint vertexArrayID, vertexBufferID, indexBufferID, uvBufferID, textureID;
    GLuint programID, viewMatrixID, projectionMatrixID, textureSamplerID;

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

       programID = LoadShadersFromFile("../lab4/shader/skybox.vert", "../lab4/shader/skybox.frag");
       
       viewMatrixID = glGetUniformLocation(programID, "view");
       projectionMatrixID = glGetUniformLocation(programID, "projection");
       textureSamplerID = glGetUniformLocation(programID, "skybox");

       textureID = LoadSkyboxTexture("../lab4/model/sky.png");
    }

    void render(glm::mat4 viewMatrix, glm::mat4 projectionMatrix) {
       glDepthFunc(GL_LEQUAL); 
       glUseProgram(programID);

       glEnableVertexAttribArray(0); 
       glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
       glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

       glEnableVertexAttribArray(2); 
       glBindBuffer(GL_ARRAY_BUFFER, uvBufferID);
       glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);

       glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);

       glm::mat4 view = glm::mat4(glm::mat3(viewMatrix)); 
       
       glUniformMatrix4fv(viewMatrixID, 1, GL_FALSE, &view[0][0]);
       glUniformMatrix4fv(projectionMatrixID, 1, GL_FALSE, &projectionMatrix[0][0]);

       glActiveTexture(GL_TEXTURE0);
       glBindTexture(GL_TEXTURE_2D, textureID);
       glUniform1i(textureSamplerID, 0);

       glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (void*)0);

       glDisableVertexAttribArray(0);
       glDisableVertexAttribArray(2);
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

// ----------------------------------------------------------------------------
// FLOOR GRID STRUCT
// ----------------------------------------------------------------------------
struct Grid {
    std::vector<GLfloat> vertices;
    GLuint vao, vbo;
    GLuint programID, mvpMatrixID;

    void initialize() {
        int size = 1000; 
        int step = 50;  

        for (int i = -size; i <= size; i += step) {
            vertices.push_back((float)-size); vertices.push_back(0); vertices.push_back((float)i);
            vertices.push_back((float)size);  vertices.push_back(0); vertices.push_back((float)i);
            vertices.push_back((float)i); vertices.push_back(0); vertices.push_back((float)-size);
            vertices.push_back((float)i); vertices.push_back(0); vertices.push_back((float)size);
        }

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

        const char* vs = "#version 330 core\nlayout(location=0) in vec3 pos; uniform mat4 MVP; void main(){gl_Position=MVP*vec4(pos,1);}";
        const char* fs = "#version 330 core\nout vec3 color; void main(){color=vec3(0.6, 0.6, 0.6);}"; 
        
        programID = LoadShadersFromString(vs, fs); 
        mvpMatrixID = glGetUniformLocation(programID, "MVP");
    }

    void render(glm::mat4 vp) {
        glUseProgram(programID);
        glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, &vp[0][0]);
        
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        
        glDrawArrays(GL_LINES, 0, vertices.size() / 3);
        glBindVertexArray(0);
    }
    
    void cleanup() {
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteProgram(programID);
    }
};

// ----------------------------------------------------------------------------
// MYBOT STRUCT
// ----------------------------------------------------------------------------
struct MyBot {
	GLuint mvpMatrixID;
	GLuint jointMatricesID;
	GLuint lightPositionID;
	GLuint lightIntensityID;
	GLuint programID;

	tinygltf::Model model;

	struct PrimitiveObject {
		GLuint vao;
		std::map<int, GLuint> vbos;
	};
	std::vector<PrimitiveObject> primitiveObjects;

	struct SkinObject {
		std::vector<glm::mat4> inverseBindMatrices;  
		std::vector<glm::mat4> globalJointTransforms;
		std::vector<glm::mat4> jointMatrices;
	};
	std::vector<SkinObject> skinObjects;

	struct SamplerObject {
		std::vector<float> input;
		std::vector<glm::vec4> output;
		int interpolation;
	};
	struct AnimationObject {
		std::vector<SamplerObject> samplers;
	};
	std::vector<AnimationObject> animationObjects;

	glm::mat4 getNodeTransform(const tinygltf::Node& node) {
		glm::mat4 transform(1.0f); 
		if (node.matrix.size() == 16) {
			transform = glm::make_mat4(node.matrix.data());
		} else {
			if (node.translation.size() == 3) {
				transform = glm::translate(transform, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
			}
			if (node.rotation.size() == 4) {
				glm::quat q(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
				transform *= glm::mat4_cast(q);
			}
			if (node.scale.size() == 3) {
				transform = glm::scale(transform, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
			}
		}
		return transform;
	}

	void computeLocalNodeTransform(const tinygltf::Model& model, int nodeIndex, std::vector<glm::mat4> &localTransforms) {
        const tinygltf::Node& node = model.nodes[nodeIndex];
        localTransforms[nodeIndex] = getNodeTransform(node);
        for (int childIndex : node.children) {
            computeLocalNodeTransform(model, childIndex, localTransforms);
        }
    }

	void computeGlobalNodeTransform(const tinygltf::Model& model, const std::vector<glm::mat4> &localTransforms, int nodeIndex, const glm::mat4& parentTransform, std::vector<glm::mat4> &globalTransforms) {
        glm::mat4 currentGlobalTransform = parentTransform * localTransforms[nodeIndex];
        globalTransforms[nodeIndex] = currentGlobalTransform;
        const tinygltf::Node& node = model.nodes[nodeIndex];
        for (int childIndex : node.children) {
            computeGlobalNodeTransform(model, localTransforms, childIndex, currentGlobalTransform, globalTransforms);
        }
    }

	std::vector<SkinObject> prepareSkinning(const tinygltf::Model &model) {
		std::vector<SkinObject> skinObjects;
		for (size_t i = 0; i < model.skins.size(); i++) {
			SkinObject skinObject;
			const tinygltf::Skin &skin = model.skins[i];
			const tinygltf::Accessor &accessor = model.accessors[skin.inverseBindMatrices];
			const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
			const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
			const float *ptr = reinterpret_cast<const float *>(buffer.data.data() + accessor.byteOffset + bufferView.byteOffset);
			
			skinObject.inverseBindMatrices.resize(accessor.count);
			for (size_t j = 0; j < accessor.count; j++) {
				float m[16];
				memcpy(m, ptr + j * 16, 16 * sizeof(float));
				skinObject.inverseBindMatrices[j] = glm::make_mat4(m);
			}
			skinObject.globalJointTransforms.resize(skin.joints.size());
			skinObject.jointMatrices.resize(skin.joints.size());

            std::vector<glm::mat4> localTransforms(model.nodes.size());
            std::vector<glm::mat4> globalTransforms(model.nodes.size());

            const tinygltf::Scene &scene = model.scenes[model.defaultScene];
            for (size_t k = 0; k < scene.nodes.size(); ++k) {
                computeLocalNodeTransform(model, scene.nodes[k], localTransforms);
            }
            for (size_t k = 0; k < scene.nodes.size(); ++k) {
                computeGlobalNodeTransform(model, localTransforms, scene.nodes[k], glm::mat4(1.0f), globalTransforms);
            }
            for (size_t j = 0; j < skin.joints.size(); ++j) {
                int nodeIndex = skin.joints[j];
                skinObject.globalJointTransforms[j] = globalTransforms[nodeIndex];
                skinObject.jointMatrices[j] = globalTransforms[nodeIndex] * skinObject.inverseBindMatrices[j];
            }
			skinObjects.push_back(skinObject);
		}
		return skinObjects;
	}

	int findKeyframeIndex(const std::vector<float>& times, float animationTime) {
		int left = 0;
		int right = times.size() - 1;
		while (left <= right) {
			int mid = (left + right) / 2;
			if (mid + 1 < times.size() && times[mid] <= animationTime && animationTime < times[mid + 1]) {
				return mid;
			} else if (times[mid] > animationTime) {
				right = mid - 1;
			} else {
				left = mid + 1;
			}
		}
		return times.size() - 2;
	}

	std::vector<AnimationObject> prepareAnimation(const tinygltf::Model &model) {
		std::vector<AnimationObject> animationObjects;
		for (const auto &anim : model.animations) {
			AnimationObject animationObject;
			for (const auto &sampler : anim.samplers) {
				SamplerObject samplerObject;
				const tinygltf::Accessor &inputAccessor = model.accessors[sampler.input];
				const tinygltf::BufferView &inputBufferView = model.bufferViews[inputAccessor.bufferView];
				const tinygltf::Buffer &inputBuffer = model.buffers[inputBufferView.buffer];
				samplerObject.input.resize(inputAccessor.count);
				const unsigned char *inputPtr = &inputBuffer.data[inputBufferView.byteOffset + inputAccessor.byteOffset];
				int stride = inputAccessor.ByteStride(inputBufferView);
				for (size_t i = 0; i < inputAccessor.count; ++i) {
					samplerObject.input[i] = *reinterpret_cast<const float*>(inputPtr + i * stride);
				}
				const tinygltf::Accessor &outputAccessor = model.accessors[sampler.output];
				const tinygltf::BufferView &outputBufferView = model.bufferViews[outputAccessor.bufferView];
				const tinygltf::Buffer &outputBuffer = model.buffers[outputBufferView.buffer];
				const unsigned char *outputPtr = &outputBuffer.data[outputBufferView.byteOffset + outputAccessor.byteOffset];
				samplerObject.output.resize(outputAccessor.count);
				for (size_t i = 0; i < outputAccessor.count; ++i) {
					if (outputAccessor.type == TINYGLTF_TYPE_VEC3) {
						memcpy(&samplerObject.output[i], outputPtr + i * 3 * sizeof(float), 3 * sizeof(float));
					} else if (outputAccessor.type == TINYGLTF_TYPE_VEC4) {
						memcpy(&samplerObject.output[i], outputPtr + i * 4 * sizeof(float), 4 * sizeof(float));
					}
				}
				animationObject.samplers.push_back(samplerObject);			
			}
			animationObjects.push_back(animationObject);
		}
		return animationObjects;
	}

	void updateAnimation(const tinygltf::Model &model, const tinygltf::Animation &anim, const AnimationObject &animationObject, float time, std::vector<glm::vec3> &translations, std::vector<glm::quat> &rotations, std::vector<glm::vec3> &scales) {
        for (const auto &channel : anim.channels) {
            int targetNodeIndex = channel.target_node;
            const auto &sampler = anim.samplers[channel.sampler];
            const tinygltf::Accessor &outputAccessor = model.accessors[sampler.output];
            const tinygltf::BufferView &outputBufferView = model.bufferViews[outputAccessor.bufferView];
            const tinygltf::Buffer &outputBuffer = model.buffers[outputBufferView.buffer];
            const std::vector<float> &times = animationObject.samplers[channel.sampler].input;
            float animationTime = fmod(time, times.back());
            int keyframeIndex = findKeyframeIndex(times, animationTime);
            const unsigned char *outputPtr = &outputBuffer.data[outputBufferView.byteOffset + outputAccessor.byteOffset];
            float t1 = times[keyframeIndex];
            float t2 = times[keyframeIndex + 1];
            float factor = (animationTime - t1) / (t2 - t1);

            if (channel.target_path == "translation") {
                glm::vec3 translation0, translation1;
                memcpy(&translation0, outputPtr + keyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
                memcpy(&translation1, outputPtr + (keyframeIndex + 1) * 3 * sizeof(float), 3 * sizeof(float));
                glm::vec3 translation = glm::mix(translation0, translation1, factor);
                translations[targetNodeIndex] = translation;
            } else if (channel.target_path == "rotation") {
                glm::quat rotation0, rotation1;
                memcpy(&rotation0, outputPtr + keyframeIndex * 4 * sizeof(float), 4 * sizeof(float));
                memcpy(&rotation1, outputPtr + (keyframeIndex + 1) * 4 * sizeof(float), 4 * sizeof(float));
                glm::quat rotation = glm::slerp(rotation0, rotation1, factor);
                rotations[targetNodeIndex] = rotation;
            } else if (channel.target_path == "scale") {
                glm::vec3 scale0, scale1;
                memcpy(&scale0, outputPtr + keyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
                memcpy(&scale1, outputPtr + (keyframeIndex + 1) * 3 * sizeof(float), 3 * sizeof(float));
                glm::vec3 scale = glm::mix(scale0, scale1, factor);
                scales[targetNodeIndex] = scale;
            }
        }
    }

	void updateSkinning(const std::vector<glm::mat4> &nodeTransforms) {
        std::vector<glm::mat4> globalTransforms(model.nodes.size());
        const tinygltf::Scene &scene = model.scenes[model.defaultScene];
        for (size_t i = 0; i < scene.nodes.size(); ++i) {
            computeGlobalNodeTransform(model, nodeTransforms, scene.nodes[i], glm::mat4(1.0f), globalTransforms);
        }
        for (SkinObject &skinObject : skinObjects) {
             const tinygltf::Skin &skin = model.skins[0];
             for (size_t j = 0; j < skin.joints.size(); ++j) {
                 int jointNodeIndex = skin.joints[j];
                 skinObject.globalJointTransforms[j] = globalTransforms[jointNodeIndex];
                 skinObject.jointMatrices[j] = globalTransforms[jointNodeIndex] * skinObject.inverseBindMatrices[j];
             }
        }
    }

	void update(float time) {
        if (model.animations.size() > 0) {
            const tinygltf::Animation &animation = model.animations[0];
            const AnimationObject &animationObject = animationObjects[0];
            std::vector<glm::vec3> translations(model.nodes.size());
            std::vector<glm::quat> rotations(model.nodes.size());
            std::vector<glm::vec3> scales(model.nodes.size());

            for (size_t i = 0; i < model.nodes.size(); ++i) {
                const tinygltf::Node &node = model.nodes[i];
                if (node.translation.size() == 3) translations[i] = glm::make_vec3(node.translation.data());
                else translations[i] = glm::vec3(0.0f);
                if (node.rotation.size() == 4) rotations[i] = glm::make_quat(node.rotation.data());
                else rotations[i] = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
                if (node.scale.size() == 3) scales[i] = glm::make_vec3(node.scale.data());
                else scales[i] = glm::vec3(1.0f);
            }
            updateAnimation(model, animation, animationObject, time, translations, rotations, scales);
            std::vector<glm::mat4> nodeTransforms(model.nodes.size());
            for (size_t i = 0; i < model.nodes.size(); ++i) {
                glm::mat4 t = glm::translate(glm::mat4(1.0f), translations[i]);
                glm::mat4 r = glm::mat4_cast(rotations[i]);
                glm::mat4 s = glm::scale(glm::mat4(1.0f), scales[i]);
                nodeTransforms[i] = t * r * s;
            }
            updateSkinning(nodeTransforms);
        }
    }

	bool loadModel(tinygltf::Model &model, const char *filename) {
		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;
		bool res = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
		if (!warn.empty()) std::cout << "WARN: " << warn << std::endl;
		if (!err.empty()) std::cout << "ERR: " << err << std::endl;
		if (!res) std::cout << "Failed to load glTF: " << filename << std::endl;
		else std::cout << "Loaded glTF: " << filename << std::endl;
		return res;
	}

	void initialize() {
		if (!loadModel(model, "../lab4/model/bot/bot.gltf")) {
			return;
		}
		primitiveObjects = bindModel(model);
		skinObjects = prepareSkinning(model);
		animationObjects = prepareAnimation(model);

		programID = LoadShadersFromFile("../lab4/shader/bot.vert", "../lab4/shader/bot.frag");
		if (programID == 0) std::cerr << "Failed to load shaders." << std::endl;

		mvpMatrixID = glGetUniformLocation(programID, "MVP");
		lightPositionID = glGetUniformLocation(programID, "lightPosition");
		lightIntensityID = glGetUniformLocation(programID, "lightIntensity");
		jointMatricesID = glGetUniformLocation(programID, "jointMatrices");
	}

	void bindMesh(std::vector<PrimitiveObject> &primitiveObjects, tinygltf::Model &model, tinygltf::Mesh &mesh) {
		std::map<int, GLuint> vbos;
		for (size_t i = 0; i < model.bufferViews.size(); ++i) {
			const tinygltf::BufferView &bufferView = model.bufferViews[i];
			int target = bufferView.target;
			if (bufferView.target == 0) continue;
			const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
			GLuint vbo;
			glGenBuffers(1, &vbo);
			glBindBuffer(target, vbo);
			glBufferData(target, bufferView.byteLength, &buffer.data.at(0) + bufferView.byteOffset, GL_STATIC_DRAW);
			vbos[i] = vbo;
		}
		for (size_t i = 0; i < mesh.primitives.size(); ++i) {
			tinygltf::Primitive primitive = mesh.primitives[i];
			GLuint vao;
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);
			for (auto &attrib : primitive.attributes) {
				tinygltf::Accessor accessor = model.accessors[attrib.second];
				int byteStride = accessor.ByteStride(model.bufferViews[accessor.bufferView]);
				glBindBuffer(GL_ARRAY_BUFFER, vbos[accessor.bufferView]);
				int size = 1;
				if (accessor.type != TINYGLTF_TYPE_SCALAR) size = accessor.type;
				int vaa = -1;
				if (attrib.first.compare("POSITION") == 0) vaa = 0;
				if (attrib.first.compare("NORMAL") == 0) vaa = 1;
				if (attrib.first.compare("TEXCOORD_0") == 0) vaa = 2;
				if (attrib.first.compare("JOINTS_0") == 0) vaa = 3;
				if (attrib.first.compare("WEIGHTS_0") == 0) vaa = 4;
				if (vaa > -1) {
					glEnableVertexAttribArray(vaa);
					glVertexAttribPointer(vaa, size, accessor.componentType, accessor.normalized ? GL_TRUE : GL_FALSE, byteStride, BUFFER_OFFSET(accessor.byteOffset));
				}
			}
			PrimitiveObject primitiveObject;
			primitiveObject.vao = vao;
			primitiveObject.vbos = vbos;
			primitiveObjects.push_back(primitiveObject);
			glBindVertexArray(0);
		}
	}

	void bindModelNodes(std::vector<PrimitiveObject> &primitiveObjects, tinygltf::Model &model, tinygltf::Node &node) {
		if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
			bindMesh(primitiveObjects, model, model.meshes[node.mesh]);
		}
		for (size_t i = 0; i < node.children.size(); i++) {
			bindModelNodes(primitiveObjects, model, model.nodes[node.children[i]]);
		}
	}

	std::vector<PrimitiveObject> bindModel(tinygltf::Model &model) {
		std::vector<PrimitiveObject> primitiveObjects;
		const tinygltf::Scene &scene = model.scenes[model.defaultScene];
		for (size_t i = 0; i < scene.nodes.size(); ++i) {
			bindModelNodes(primitiveObjects, model, model.nodes[scene.nodes[i]]);
		}
		return primitiveObjects;
	}

	void drawMesh(const std::vector<PrimitiveObject> &primitiveObjects, tinygltf::Model &model, tinygltf::Mesh &mesh) {
		for (size_t i = 0; i < mesh.primitives.size(); ++i) {
			GLuint vao = primitiveObjects[i].vao;
			std::map<int, GLuint> vbos = primitiveObjects[i].vbos;
			glBindVertexArray(vao);
			tinygltf::Primitive primitive = mesh.primitives[i];
			tinygltf::Accessor indexAccessor = model.accessors[primitive.indices];
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos.at(indexAccessor.bufferView));
			glDrawElements(primitive.mode, indexAccessor.count, indexAccessor.componentType, BUFFER_OFFSET(indexAccessor.byteOffset));
			glBindVertexArray(0);
		}
	}

	void drawModelNodes(const std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model &model, tinygltf::Node &node) {
		if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
			drawMesh(primitiveObjects, model, model.meshes[node.mesh]);
		}
		for (size_t i = 0; i < node.children.size(); i++) {
			drawModelNodes(primitiveObjects, model, model.nodes[node.children[i]]);
		}
	}
	void drawModel(const std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model &model) {
		const tinygltf::Scene &scene = model.scenes[model.defaultScene];
		for (size_t i = 0; i < scene.nodes.size(); ++i) {
			drawModelNodes(primitiveObjects, model, model.nodes[scene.nodes[i]]);
		}
	}

	void render(glm::mat4 projectionViewMatrix) {
		glUseProgram(programID);
		
		// Model Matrix is identity because the CAMERA is moving, not the bot
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        
        // Combine into MVP
        glm::mat4 mvp = projectionViewMatrix * modelMatrix;
		glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, &mvp[0][0]);

		// Send animation data
		if (!skinObjects.empty()) {
             glUniformMatrix4fv(jointMatricesID, (GLsizei)skinObjects[0].jointMatrices.size(), GL_FALSE, glm::value_ptr(skinObjects[0].jointMatrices[0]));
        }
		
		// Set light data 
		glUniform3fv(lightPositionID, 1, &lightPosition[0]);
		glUniform3fv(lightIntensityID, 1, &lightIntensity[0]);

		// Draw
		drawModel(primitiveObjects, model);
	}

	void cleanup() {
		glDeleteProgram(programID);
	}
}; 

// ----------------------------------------------------------------------------
// MAIN FUNCTION
// ----------------------------------------------------------------------------
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

	window = glfwCreateWindow(windowWidth, windowHeight, "Lab 4 - Wonderland", NULL, NULL);
	if (window == NULL) {
		std::cerr << "Failed to open a GLFW window." << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetKeyCallback(window, key_callback);

	int version = gladLoadGL(glfwGetProcAddress);
	if (version == 0) {
		std::cerr << "Failed to initialize OpenGL context." << std::endl;
		return -1;
	}

	glClearColor(0.2f, 0.2f, 0.25f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// Initialize Objects
	MyBot bot;
	bot.initialize();

	Skybox skybox;
	skybox.initialize();
    
    Grid grid;
    grid.initialize();

    glm::mat4 viewMatrix, projectionMatrix;
	projectionMatrix = glm::perspective(glm::radians(FoV), (float)windowWidth / windowHeight, zNear, zFar);

	static double lastTime = glfwGetTime();
	float time = 0.0f;			
	float fTime = 0.0f;			
	unsigned long frames = 0;

	// Loop
	do
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		double currentTime = glfwGetTime();
        float deltaTime = float(currentTime - lastTime);
		lastTime = currentTime;

		if (playAnimation) {
			time += deltaTime * playbackSpeed;
			bot.update(time);
		} 

        // -----------------------------------------------------------
        // RENDER LOOP
        // -----------------------------------------------------------
        
        // 1. Calculate View Matrix (Camera)
		viewMatrix = glm::lookAt(eye_center, lookat, up);
        
        // 2. Render Skybox (Remove translation from view for skybox)
        skybox.render(viewMatrix, projectionMatrix);

        // 3. Calculate MVP
		glm::mat4 vp = projectionMatrix * viewMatrix;
        
        // 4. Render Grid
        grid.render(vp);

        // 5. Render Bot
		bot.render(vp);

		// FPS
		frames++;
		fTime += deltaTime;
		if (fTime > 2.0f) {		
			float fps = frames / fTime;
			frames = 0;
			fTime = 0;
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << "FPS: " << fps;
			glfwSetWindowTitle(window, stream.str().c_str());
		}

		glfwSwapBuffers(window);
		glfwPollEvents();

	} while (!glfwWindowShouldClose(window));

	bot.cleanup();
	skybox.cleanup();
    grid.cleanup();
	glfwTerminate();

	return 0;
}

// ----------------------------------------------------------------------------
// KEY CALLBACK (MOVES CAMERA)
// ----------------------------------------------------------------------------
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode)
{
    float moveSpeed = 20.0f; 
    float rotateSpeed = 0.05f;

    // 1. Calculate the vector from the camera to the center (The "Forward" direction)
    // We only care about X and Z for movement on the floor
    glm::vec3 forward = lookat - eye_center;
    forward.y = 0; // Keep movement flat on the ground
    forward = glm::normalize(forward); // Make it a unit vector (length 1)

    // 2. ROTATE CAMERA (Orbit around center)
    if (key == GLFW_KEY_A && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
        float radius = glm::length(glm::vec2(eye_center.x, eye_center.z));
        float angle = atan2(eye_center.z, eye_center.x);
        angle -= rotateSpeed; 
        eye_center.x = radius * cos(angle);
        eye_center.z = radius * sin(angle);
    }
    if (key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
        float radius = glm::length(glm::vec2(eye_center.x, eye_center.z));
        float angle = atan2(eye_center.z, eye_center.x);
        angle += rotateSpeed;
        eye_center.x = radius * cos(angle);
        eye_center.z = radius * sin(angle);
    }

    // 3. MOVE CAMERA (Zoom In / Out)
    // Instead of changing Z directly, we add/subtract the Forward vector
    if (key == GLFW_KEY_W && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
        eye_center += forward * moveSpeed; 
    }
    if (key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS)) {
        eye_center -= forward * moveSpeed;
    }
    
    // Toggle Animation
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		playAnimation = !playAnimation;
	}

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}