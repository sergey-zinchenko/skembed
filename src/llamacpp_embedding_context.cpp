//
// Created by Sergei on 4/10/2024.
//

#include "llamacpp_embedding_context.h"

template<typename FlatEmbedType>
std::shared_ptr<abstract_flat_embed> llamacpp_embedding_context<FlatEmbedType>::embed(const std::vector<std::string>::iterator &prompts_start,
                                                                        const std::vector<std::string>::iterator &prompts_end) {
    logger_->trace("Embedding {} prompts");
    std::lock_guard lock(mutex_);
    auto tokenized_prompts = tokenize_and_trim(prompts_start, prompts_end);

    auto embeddings = process_tokenized_prompts(tokenized_prompts);
    logger_->debug("Prompts embedded");
    return embeddings;
}

template<typename FlatEmbedType>
std::vector<std::vector<int32_t>> llamacpp_embedding_context<FlatEmbedType>::tokenize_and_trim(const std::vector<std::string>::iterator &prompts_start,
                                                                                const std::vector<std::string>::iterator &prompts_end) const {
    logger_->trace("Tokenizing prompts");
    std::vector<std::vector<int32_t>> tokenized_prompts;
    for (auto prompt = prompts_start; prompt != prompts_end; ++prompt) {
        auto tokenized_elem = ::llama_tokenize(ctx_, *prompt, true, false);
        if (tokenized_elem.size() > n_batch_) {
            throw std::runtime_error(
                    "Number of tokens in input line exceeds batch size, increase batch size and re-run");
        }
        if (tokenized_elem.empty() || tokenized_elem.back() != eos_token_) {
            tokenized_elem.push_back(eos_token_);
        }
        tokenized_prompts.push_back(tokenized_elem);
    }
    logger_->debug("Prompts tokenized");
    return tokenized_prompts;
}

template<typename FlatEmbedType>
std::shared_ptr<FlatEmbedType>
llamacpp_embedding_context<FlatEmbedType>::process_tokenized_prompts(const std::vector<std::vector<int32_t>> &tokenized_prompts) const {
    logger_->trace("Processing tokenized prompts");
    auto embeddings = flat_embed_factory_();
    auto batch = llama_batch_init(n_batch_, 0, 1);
    auto p_emb = embeddings.data();
    auto p = 0, s = 0;
    for (const auto &tokenized_prompt: tokenized_prompts) {
        if (batch.n_tokens + tokenized_prompt.size() > n_batch_) {
            batch_decode(batch, p_emb + p * n_embed_);
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }
        batch_add_seq(batch, tokenized_prompt, s++);
    }
    batch_decode(batch, p_emb + p * n_embed_);
    llama_batch_clear(batch);
    llama_batch_free(batch);
    logger_->debug("Tokenized prompts prompts processed");
    return embeddings;
}

template<typename FlatEmbedType>
void llamacpp_embedding_context<FlatEmbedType>::batch_decode(llama_batch &batch, float *output) const {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx_);

    // run model
    auto decode_result = llama_decode(ctx_, batch);
    if (decode_result == 1)
        logger_->warn(
                "could not find a KV slot for the batch (try reducing the size of the batch or increase the context)");
    else if (decode_result < 0)
        throw std::runtime_error("error decoding batch");

    for (auto i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        auto embed = llama_get_embeddings_seq(ctx_, batch.seq_id[i][0]);
        if (!embed) {
            embed = llama_get_embeddings_ith(ctx_, i);
        }

        if (embed) {
            auto out = output + batch.seq_id[i][0] * n_embed_;
            llama_embd_normalize(embed, out, n_embed_);
        }
    }
}

template<typename FlatEmbedType>
void llamacpp_embedding_context<FlatEmbedType>::batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, int seq_id) {
    for (auto i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, {seq_id}, i == tokens.size() - 1);
    }
}

template<typename FlatEmbedType>
llamacpp_embedding_context<FlatEmbedType>::~llamacpp_embedding_context() {
    logger_->trace("Freeing context");
    llama_free(ctx_);
    logger_->info("Context freed");
}

template<typename FlatEmbedType>
llamacpp_embedding_context<FlatEmbedType>::llamacpp_embedding_context(const gpt_params &params,
                                                                      const struct llama_model *model,
                                                                      const std::function<std::shared_ptr<FlatEmbedType>()> &flat_embed_factory,
                                                                      const std::shared_ptr<abstract_model> &a_model,
                                                                      const std::shared_ptr<spdlog::logger> &logger) {

}


template<typename FlatEmbedType>
struct llama_context *
llamacpp_embedding_context<FlatEmbedType>::init_context_from_gpt_params(gpt_params &params, struct llama_model *model) {
    auto cparams = llama_context_params_from_gpt_params(params);

    llama_context * lctx = llama_new_context_with_model(model, cparams);
    if (!lctx)
        throw std::runtime_error("failed to create context" + params.model);


    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_n_layer(model);

        const auto cvec = llama_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            llama_free(lctx);
            throw std::runtime_error("failed to load control vectors");
        }

        int err = llama_control_vector_apply(lctx,
                                             cvec.data.data(),
                                             cvec.data.size(),
                                             cvec.n_embd,
                                             params.control_vector_layer_start,
                                             params.control_vector_layer_end);
        if (err) {
            llama_free(lctx);
            throw std::runtime_error("failed to apply control vectors");
        }
    }

    if (params.ignore_eos) {
        params.sparams.logit_bias[llama_token_eos(model)] = -INFINITY;
    }

    {
        //warming up the model with an empty run
        std::vector<llama_token> tmp = { llama_token_bos(model), llama_token_eos(model), };
        llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(static_cast<int32_t>(tmp.size()), params.n_batch), 0, 0));
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_reset_timings(lctx);
    }

    return lctx;
}
