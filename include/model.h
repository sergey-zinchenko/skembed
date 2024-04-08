//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <mutex>
#include "abstract/abstract_model.h"
#include "llama/llama.h"
#include "llama/common.h"
#include "spdlog/spdlog.h"

class model: public abstract_model {
public:
     void load_model() override;
     void unload_model() override;
     std::vector<std::vector<float_t>> embeddings(const std::vector<std::string> &prompts) override;
     model(gpt_params params,
           std::shared_ptr<spdlog::logger> logger);
private:
    void batch_decode(llama_batch & batch, float * output);
    static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, int seq_id);
    gpt_params params_;
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex model_state_mutex_;
    int64_t model_loaded_count_ = 0;
    llama_context *ctx_{};
    llama_model *model_{};
    int32_t n_batch_{};
    int32_t n_embed_{};
    llama_token eos_token_{};
};