//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <mutex>
#include "abstract/abstract_model.h"
#include "abstract/abstract_model_backend.h"
#include "llama/llama.h"
#include "llama/common.h"
#include "spdlog/spdlog.h"
#include "flat_embed.h"

class model : public abstract_model {
public:
    [[nodiscard]] flat_embed
    embed(const std::vector<std::string> &prompts) override;

    model(const gpt_params &params,
          std::function<flat_embed(size_t, size_t)> embed_factory,
          std::shared_ptr<abstract_model_backend> model_backend,
          std::shared_ptr<spdlog::logger> logger);

    ~model() override;

private:
    [[nodiscard]] std::vector<std::vector<int32_t>> tokenize_and_trim(const std::vector<std::string> &prompts) const;

    [[nodiscard]] flat_embed
    process_tokenized_prompts(const std::vector<std::vector<int32_t>> &tokenized_prompts) const;

    void batch_decode(llama_batch &batch, float *output) const;

    static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, int seq_id);

    gpt_params params_;
    std::function<flat_embed(size_t, size_t)> embed_factory_;
    std::shared_ptr<abstract_model_backend> model_backend_;
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex mutex_;
    llama_context *ctx_{};
    llama_model *model_{};
    int32_t n_batch_{};
    int32_t n_embed_{};
    llama_token eos_token_{};
};