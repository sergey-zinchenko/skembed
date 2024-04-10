//
// Created by Sergei on 4/10/2024.
//

#pragma once


#include <mutex>
#include "abstract/abstract_embedding_context.h"
#include "llama/llama.h"
#include "llama/common.h"
#include "spdlog/logger.h"
#include "abstract/abstract_model.h"

template<typename FlatEmbedType>
class llamacpp_embedding_context : public abstract_embedding_context {
    static_assert(std::is_base_of<abstract_flat_embed, FlatEmbedType>::value, "FlatEmbedType must be a subclass of abstract_flat_embed");
public:
    llamacpp_embedding_context(const gpt_params &params,
                               const struct llama_model *model,
                               const std::function<std::shared_ptr<FlatEmbedType>()>& flat_embed_factory,
                               const std::shared_ptr<abstract_model> &a_model,
                               const std::shared_ptr<spdlog::logger> &logger);

    [[nodiscard]] std::shared_ptr<abstract_flat_embed>
    embed(const std::vector<std::string>::iterator &start, const std::vector<std::string>::iterator &end) override;

    ~llamacpp_embedding_context() override;
private:
    [[nodiscard]] std::vector<std::vector<int32_t>> tokenize_and_trim(const std::vector<std::string>::iterator &prompts_start,
                                                                      const std::vector<std::string>::iterator &prompts_end) const;

    [[nodiscard]] std::shared_ptr<FlatEmbedType>
    process_tokenized_prompts(const std::vector<std::vector<int32_t>> &tokenized_prompts) const;


    void batch_decode(llama_batch &batch, float *output) const;

    static void batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, int seq_id);

    static struct llama_context *init_context_from_gpt_params(gpt_params &params, struct llama_model *model);

    gpt_params params_;
    std::function<std::shared_ptr<FlatEmbedType>()> flat_embed_factory_;
    llama_context *ctx_{};
    int32_t n_batch_{};
    int32_t n_embed_{};
    llama_token eos_token_{};
    std::mutex mutex_;
    std::shared_ptr<spdlog::logger> logger_;
    std::shared_ptr<abstract_model> model_;
};

