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
    [[nodiscard]] auto embed(const std::vector<std::string> &prompts) -> flat_embed override;

    model(gpt_params params,
          std::function<flat_embed(size_t, size_t)> embed_factory,
          std::shared_ptr<abstract_model_backend> model_backend,
          std::shared_ptr<spdlog::logger> logger);

    ~model() override;

private:
    [[nodiscard]] auto
    tokenize_and_trim(const std::vector<std::string> &prompts) const -> std::vector<std::vector<int32_t>>;

    [[nodiscard]] auto
    process_tokenized_prompts(const std::vector<std::vector<int32_t>> &tokenized_prompts) const -> flat_embed;

    void batch_decode(llama_batch &batch, float *output) const;

    void log_embeddings(const std::vector<std::string> &prompts, const flat_embed &embeds) const;

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
	llama_token sep_token {};
};