//
// Created by Sergei on 4/6/2024.
//

#include "model.h"

model::model(const gpt_params &params, std::function<flat_embed(size_t, size_t)> embed_factory,
             std::shared_ptr<abstract_model_backend> model_backend,
             std::shared_ptr<spdlog::logger> logger) :
        params_(std::move(params)),
        embed_factory_(std::move(embed_factory)),
        model_backend_(std::move(model_backend)),
        logger_(std::move(logger)) {
    logger_->trace("Loading model");
    params_.embedding = true;
    params_.n_ubatch = params_.n_batch;
    std::tie(model_, ctx_) = llama_init_from_gpt_params(params_);
    if (model_ == nullptr) {
        throw std::runtime_error("Unable to load model");
    }
    std::mt19937 rng(params_.seed);
    if (params_.random_prompt) {
        params_.prompt = gpt_random_prompt(rng);
    }
    auto n_ctx_train = llama_n_ctx_train(model_);
    auto n_ctx = llama_n_ctx(ctx_);
    if (n_ctx > n_ctx_train) {
        logger_->warn("Model was trained on only {} context tokens ({} specified)", n_ctx_train, n_ctx);
    }
    n_batch_ = params_.n_batch;
    if (params_.n_batch < params_.n_ctx) {
        throw std::runtime_error("Batch size must be greater than context size");
    }
    n_embed_ = llama_n_embd(model_);
    eos_token_ = llama_token_eos(model_);
    logger_->info("Model loaded");
}


std::vector<std::vector<int32_t>> model::tokenize_and_trim(const std::vector<std::string> &prompts) const {
    std::vector<std::vector<int32_t>> tokenized_prompts;
    for (const auto &prompt: prompts) {
        auto tokenized_elem = ::llama_tokenize(ctx_, prompt, true, false);
        if (tokenized_elem.size() > n_batch_) {
            throw std::runtime_error(
                    "Number of tokens in input line exceeds batch size, increase batch size and re-run");
        }
        if (tokenized_elem.back() != eos_token_)
            tokenized_elem.emplace_back(eos_token_);

        tokenized_prompts.emplace_back(tokenized_elem);
    }
    return tokenized_prompts;
}

flat_embed model::process_tokenized_prompts(const std::vector<std::vector<int32_t>> &tokenized_prompts) const {
    auto embeddings = embed_factory_(n_embed_, tokenized_prompts.size());
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
    return embeddings;
}

void model::batch_decode(llama_batch &batch, float *output) const {
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

void model::batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, int seq_id) {
    for (auto i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, {seq_id}, i == tokens.size() - 1);
    }
}

model::~model() {
    logger_->trace("Freeing model");
    llama_free(ctx_);
    llama_free_model(model_);
    logger_->info("Model freed");
}

flat_embed model::embed(const std::vector<std::string> &prompts) {
    logger_->trace("Embedding prompts");
    std::lock_guard lock(mutex_);
    auto tokenized_prompts = tokenize_and_trim(prompts);
    auto embeddings = process_tokenized_prompts(tokenized_prompts);
    logger_->trace("Prompts embedded");
    return embeddings;
}
