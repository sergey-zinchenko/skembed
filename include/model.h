//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <shared_mutex>
#include "abstract/abstract_model.h"
#include "abstract/abstract_model_backend.h"
#include "llama/llama.h"
#include "llama/common.h"
#include "spdlog/spdlog.h"

class model : public abstract_model {
public:
    model(gpt_params params,
          const std::shared_ptr<abstract_model_backend> &model_backend,
          std::shared_ptr<spdlog::logger> logger);

    ~model() override;

    std::shared_ptr<abstract_embedding_context> create_embedding_context() override;

private:
    [[nodiscard]] std::vector<std::vector<int32_t>> tokenize_and_trim(const std::vector<std::string> &prompts) const;

    [[nodiscard]] std::vector<float_t>
    process_tokenized_prompts(const std::vector<std::vector<int32_t>> &tokenized_prompts) const;

    [[nodiscard]] std::vector<std::vector<float_t>> reshape_embeddings(const std::vector<float> &flat_embeddings) const;

    [[nodiscard]] static struct llama_model *init_model_from_gpt_params(const gpt_params &params);

    void batch_decode(llama_batch &batch, float *output) const;

    static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, int seq_id);

    gpt_params params_;
    std::shared_ptr<abstract_model_backend> model_backend_;
    std::shared_ptr<spdlog::logger> logger_;
    std::shared_mutex mutex_;
    llama_context *ctx_{};
    llama_model *model_{};
    int32_t n_batch_{};
    int32_t n_embed_{};
    llama_token eos_token_{};
};