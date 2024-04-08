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
     std::vector<std::vector<float_t>> Embeddings(const std::vector<std::string> &prompts) override;
     explicit Model(gpt_params *params);
private:
    void batchDecode(llama_batch & batch, float * output);
    static void batchAddSeq(llama_batch & batch, const std::vector<int32_t> & tokens, int seq_id);
    gpt_params *params_;
    std::mutex modelStateMutex_;
    int64_t modelLoadedCount_ = 0;
    llama_context *ctx_{};
    llama_model *model_{};
    int32_t nBatch{};
    int32_t nEmbd{};
    llama_token eosToken{};
};