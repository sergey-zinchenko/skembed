//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <mutex>
#include "abstract/IModel.h"
#include "llama/llama.h"
#include "llama/common.h"

class Model: public IModel {
public:
     void LoadModel() override;
     void UnloadModel() override;
     std::vector<float_t> Embeddings(std::string text) override;
     explicit Model(gpt_params *params);
private:
    gpt_params *params_;
    std::mutex modelStateMutex_;
    int64_t modelLoadedCount_ = 0;
    llama_context *ctx_{};
    llama_model *model_{};
    int32_t n_ctx_train{};
    uint32_t n_ctx{};
};
