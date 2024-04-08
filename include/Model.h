//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <mutex>
#include "abstract/IModel.h"
#include "llama/llama.h"
#include "llama/common.h"
#include "spdlog/spdlog.h"

class Model: public IModel {
public:
     void LoadModel() override;
     void UnloadModel() override;
     std::vector<std::vector<float_t>> Embeddings(const std::vector<std::string> &prompts) override;
     explicit Model(gpt_params *params,
                    std::shared_ptr<spdlog::logger> logger);
private:
    void batchDecode(llama_batch & batch, float * output);
    static void batchAddSeq(llama_batch & batch, const std::vector<int32_t> & tokens, int seq_id);
    gpt_params *params_;
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex modelStateMutex_;
    int64_t modelLoadedCount_ = 0;
    llama_context *ctx_{};
    llama_model *model_{};
    int32_t nBatch{};
    int32_t nEmbed{};
    llama_token eosToken{};
};