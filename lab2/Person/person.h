struct MyBot {

   // Pick a valid scene index (glTF defaultScene may be -1)
   static int safeSceneIndex(const tinygltf::Model &model) {
      if (model.scenes.empty()) return -1;
      int si = model.defaultScene;
      if (si < 0 || si >= (int)model.scenes.size()) si = 0;
      return si;
   }

    // Shader variable IDs
    GLuint mvpMatrixID;
    GLuint jointMatricesID;
    GLuint lightPositionID;
    GLuint lightIntensityID;
    GLuint programID;

    tinygltf::Model model;

   std::vector<GLuint> textureIDs;
   GLuint textureSamplerID; // Handle for the texture uniform

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
    struct ChannelObject {
       int sampler;
       std::string targetPath;
       int targetNode;
    };
    struct AnimationObject {
       std::vector<SamplerObject> samplers;
    };
    std::vector<AnimationObject> animationObjects;

   void loadTextures(tinygltf::Model &model) {
      textureIDs.resize(model.textures.size());

      for (size_t i = 0; i < model.textures.size(); i++) {
         tinygltf::Texture &tex = model.textures[i];
         if (tex.source > -1) {
            GLuint texID;
            glGenTextures(1, &texID);

            tinygltf::Image &image = model.images[tex.source];

            glBindTexture(GL_TEXTURE_2D, texID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            GLenum format = GL_RGBA;
            if (image.component == 3) format = GL_RGB;

            glTexImage2D(GL_TEXTURE_2D, 0, format, image.width, image.height,
                         0, format, GL_UNSIGNED_BYTE, &image.image.at(0));
            glGenerateMipmap(GL_TEXTURE_2D);

            textureIDs[i] = texID;
         }
      }
   }

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
        const auto& currentNode = model.nodes[nodeIndex];
        localTransforms[nodeIndex] = getNodeTransform(currentNode);
        for (size_t i = 0; i < currentNode.children.size(); ++i) {
            computeLocalNodeTransform(model, currentNode.children[i], localTransforms);
        }
    }

    void computeGlobalNodeTransform(const tinygltf::Model& model, const std::vector<glm::mat4> &localTransforms, int nodeIndex, const glm::mat4& parentTransform, std::vector<glm::mat4> &globalTransforms) {
        glm::mat4 accumulatedMatrix = parentTransform * localTransforms[nodeIndex];
        globalTransforms[nodeIndex] = accumulatedMatrix;
        const auto& currentNode = model.nodes[nodeIndex];
        for (int childID : currentNode.children) {
            computeGlobalNodeTransform(model, localTransforms, childID, accumulatedMatrix, globalTransforms);
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

          std::vector<glm::mat4> nodeLocals(model.nodes.size());
          std::vector<glm::mat4> nodeGlobals(model.nodes.size());
          const tinygltf::Scene &activeScene = model.scenes[model.defaultScene];
          for (int rootNode : activeScene.nodes) computeLocalNodeTransform(model, rootNode, nodeLocals);
          glm::mat4 identity(1.0f);
          for (int rootNode : activeScene.nodes) computeGlobalNodeTransform(model, nodeLocals, rootNode, identity, nodeGlobals);

          for (size_t j = 0; j < skin.joints.size(); ++j) {
             int nodeID = skin.joints[j];
             glm::mat4 globalMtx = nodeGlobals[nodeID];
             skinObject.globalJointTransforms[j] = globalMtx;
             skinObject.jointMatrices[j] = globalMtx * skinObject.inverseBindMatrices[j];
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
          if (mid + 1 < times.size() && times[mid] <= animationTime && animationTime < times[mid + 1]) return mid;
          else if (times[mid] > animationTime) right = mid - 1;
          else left = mid + 1;
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
             for (size_t i = 0; i < inputAccessor.count; ++i) samplerObject.input[i] = *reinterpret_cast<const float*>(inputPtr + i * stride);

             const tinygltf::Accessor &outputAccessor = model.accessors[sampler.output];
             const tinygltf::BufferView &outputBufferView = model.bufferViews[outputAccessor.bufferView];
             const tinygltf::Buffer &outputBuffer = model.buffers[outputBufferView.buffer];
             const unsigned char *outputPtr = &outputBuffer.data[outputBufferView.byteOffset + outputAccessor.byteOffset];
             samplerObject.output.resize(outputAccessor.count);
             for (size_t i = 0; i < outputAccessor.count; ++i) {
                if (outputAccessor.type == TINYGLTF_TYPE_VEC3) memcpy(&samplerObject.output[i], outputPtr + i * 3 * sizeof(float), 3 * sizeof(float));
                else if (outputAccessor.type == TINYGLTF_TYPE_VEC4) memcpy(&samplerObject.output[i], outputPtr + i * 4 * sizeof(float), 4 * sizeof(float));
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
                translations[targetNodeIndex] = glm::mix(translation0, translation1, factor);
            } else if (channel.target_path == "rotation") {
                glm::quat rotation0, rotation1;
                memcpy(&rotation0, outputPtr + keyframeIndex * 4 * sizeof(float), 4 * sizeof(float));
                memcpy(&rotation1, outputPtr + (keyframeIndex + 1) * 4 * sizeof(float), 4 * sizeof(float));
                rotations[targetNodeIndex] = glm::slerp(rotation0, rotation1, factor);
            } else if (channel.target_path == "scale") {
                glm::vec3 scale0, scale1;
                memcpy(&scale0, outputPtr + keyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
                memcpy(&scale1, outputPtr + (keyframeIndex + 1) * 3 * sizeof(float), 3 * sizeof(float));
                scales[targetNodeIndex] = glm::mix(scale0, scale1, factor);
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
       std::string err, warn;
       bool res = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
       if (!warn.empty()) std::cout << "WARN: " << warn << std::endl;
       if (!err.empty()) std::cout << "ERR: " << err << std::endl;
       return res;
    }

   void initialize() {
      if (!loadModel(model, "lab2/Person/bot.gltf")) {
         std::cerr << "Failed to load bot.gltf! Check path." << std::endl;
         return;
      }

      // --- ADD THIS LINE ---
      loadTextures(model);
      // --------------------

      primitiveObjects = bindModel(model);
      skinObjects = prepareSkinning(model);
      animationObjects = prepareAnimation(model);

      programID = LoadShadersFromFile("../lab2/Person/bot.vert", "../lab2/Person/bot.frag");

      mvpMatrixID = glGetUniformLocation(programID, "MVP");
      lightPositionID = glGetUniformLocation(programID, "lightPosition");
      lightIntensityID = glGetUniformLocation(programID, "lightIntensity");
      jointMatricesID = glGetUniformLocation(programID, "jointMatrices");

      // --- ADD THIS LINE ---
      textureSamplerID = glGetUniformLocation(programID, "diffuseTexture");
      // --------------------
   }

    void bindMesh(std::vector<PrimitiveObject> &primitiveObjects, tinygltf::Model &model, tinygltf::Mesh &mesh) {
       std::map<int, GLuint> vbos;
       for (size_t i = 0; i < model.bufferViews.size(); ++i) {
          const tinygltf::BufferView &bufferView = model.bufferViews[i];
          if (bufferView.target == 0) continue;
          const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
          GLuint vbo;
          glGenBuffers(1, &vbo);
          glBindBuffer(bufferView.target, vbo);
          glBufferData(bufferView.target, bufferView.byteLength, &buffer.data.at(0) + bufferView.byteOffset, GL_STATIC_DRAW);
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
       if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) bindMesh(primitiveObjects, model, model.meshes[node.mesh]);
       for (size_t i = 0; i < node.children.size(); i++) bindModelNodes(primitiveObjects, model, model.nodes[node.children[i]]);
    }

    std::vector<PrimitiveObject> bindModel(tinygltf::Model &model) {
       std::vector<PrimitiveObject> primitiveObjects;
       int sceneIndex = safeSceneIndex(model);
       if (sceneIndex < 0) { std::cerr << "MyBot: model has no scenes\n"; return primitiveObjects; }
       const tinygltf::Scene &scene = model.scenes[sceneIndex];
       for (size_t i = 0; i < scene.nodes.size(); ++i) {
          int nodeIndex = scene.nodes[i];
          if (nodeIndex < 0 || nodeIndex >= (int)model.nodes.size()) continue;
          bindModelNodes(primitiveObjects, model, model.nodes[nodeIndex]);
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

         // --- TEXTURE BINDING LOGIC ---
         if (primitive.material >= 0) {
            tinygltf::Material &mat = model.materials[primitive.material];
            // Look for baseColorTexture in PBR metallic roughness
            int texIndex = mat.pbrMetallicRoughness.baseColorTexture.index;
            if (texIndex >= 0 && texIndex < textureIDs.size()) {
               glActiveTexture(GL_TEXTURE0);
               glBindTexture(GL_TEXTURE_2D, textureIDs[texIndex]);
               glUniform1i(textureSamplerID, 0); // Tell shader to use texture unit 0
            }
         }
         // -----------------------------

         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos.at(indexAccessor.bufferView));
         glDrawElements(primitive.mode, indexAccessor.count, indexAccessor.componentType, BUFFER_OFFSET(indexAccessor.byteOffset));
         glBindVertexArray(0);
      }
   }

    void drawModelNodes(const std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model &model, tinygltf::Node &node) {
       if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) drawMesh(primitiveObjects, model, model.meshes[node.mesh]);
       for (size_t i = 0; i < node.children.size(); i++) drawModelNodes(primitiveObjects, model, model.nodes[node.children[i]]);
    }
    void drawModel(const std::vector<PrimitiveObject>& primitiveObjects, tinygltf::Model &model) {
       int sceneIndex = safeSceneIndex(model);
       if (sceneIndex < 0) { std::cerr << "MyBot: model has no scenes\n"; return; }
       const tinygltf::Scene &scene = model.scenes[sceneIndex];
       for (size_t i = 0; i < scene.nodes.size(); ++i) {
         int nodeIndex = scene.nodes[i];
          if (nodeIndex < 0 || nodeIndex >= (int)model.nodes.size()) continue;
          drawModelNodes(primitiveObjects, model, model.nodes[nodeIndex]);
       }
    }

    // UPDATED RENDER FUNCTION for the world
    void render(glm::mat4 view, glm::mat4 projection, glm::mat4 modelMatrix, glm::vec3 lightPos, glm::vec3 lightInt) {
       glUseProgram(programID);

       glm::mat4 mvp = projection * view * modelMatrix;
       glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, &mvp[0][0]);

       if (skinObjects.size() > 0) {
            auto& joints = skinObjects[0].jointMatrices;
            glUniformMatrix4fv(jointMatricesID, (GLsizei)joints.size(), GL_FALSE, glm::value_ptr(joints[0]));
       }

       glUniform3fv(lightPositionID, 1, &lightPos[0]);
       glUniform3fv(lightIntensityID, 1, &lightInt[0]);

       drawModel(primitiveObjects, model);
    }

    void cleanup() {
       glDeleteProgram(programID);
    }
};