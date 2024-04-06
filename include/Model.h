//
// Created by Sergei on 4/6/2024.
//

#pragma once

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
    llama_context *ctx_ {};
    llama_model *model_ {};
    gpt_params *params_;
};
