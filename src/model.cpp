//
// Created by Sergei on 4/6/2024.
//

#include "model.h"

model::model(gpt_params params, std::shared_ptr<spdlog::logger> logger) :
        params_(std::move(params)),
        logger_(std::move(logger)) {
}

void model::load_model() {
    std::unique_lock lock(mutex_);
    if (model_loaded_count_++ > 0)
        return;
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
        throw std::runtime_error("Model was trained on only " + std::to_string(n_ctx_train) +
                                 " context tokens (" + std::to_string(n_ctx) + " specified)");
    }
    n_batch_ = params_.n_batch;
    if (params_.n_batch < params_.n_ctx) {
        throw std::runtime_error("Batch size must be greater than context size");
    }
    n_embed_ = llama_n_embd(model_);
    eos_token_ = llama_token_eos(model_);
}

void model::unload_model() {
    std::unique_lock lock(mutex_);
    if (model_loaded_count_ <= 0)
        throw std::runtime_error("Model::unload_model() called without initialization");
    if (model_loaded_count_-- > 0)
        return;
    llama_free(ctx_);
    llama_free_model(model_);
}

std::vector<std::vector<float_t>> model::embeddings(const std::vector<std::string> &prompts) {
    std::shared_lock lock(mutex_);
    if (model_loaded_count_ <= 0)
        throw std::runtime_error("Model::embeddings() called without initialization");
    std::vector<std::vector<int32_t>> inputs = tokenize_and_trim(prompts);
    auto batch = llama_batch_init(n_batch_, 0, 1);
    std::vector<float> embeddings(inputs.size() * n_embed_, 0);
    process_batches(inputs, batch, embeddings.data());
    return reshape_embeddings(embeddings, inputs.size());
}

std::vector<std::vector<int32_t>> model::tokenize_and_trim(const std::vector<std::string> &prompts) {
    std::vector<std::vector<int32_t>> tokenized_prompts;
    for (const auto &prompt: prompts) {
        auto tokenized_elem = ::llama_tokenize(ctx_, prompt, true, false);
        if (tokenized_elem.size() > n_batch_) {
            throw std::runtime_error(
                    "Number of tokens in input line exceeds batch size, increase batch size and re-run");
        }
        if (tokenized_elem.empty() || tokenized_elem.back() != eos_token_) {
            tokenized_elem.push_back(eos_token_);
        }
        tokenized_prompts.push_back(tokenized_elem);
    }
    return tokenized_prompts;
}

void model::process_batches(const std::vector<std::vector<int32_t>> &inputs, llama_batch &batch, float *emb) const {
    auto p = 0, s = 0;
    for (const auto &inp: inputs) {
        if (batch.n_tokens + inp.size() > n_batch_) {
            batch_decode(batch, emb + p * n_embed_);
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }
        batch_add_seq(batch, inp, s++);
    }
    batch_decode(batch, emb + p * n_embed_);
}

std::vector<std::vector<float_t>> model::reshape_embeddings(const std::vector<float> &embeddings, size_t n_prompts) {
    return std::vector<std::vector<float_t>>(n_prompts, std::vector<float_t>(embeddings.begin(), embeddings.end()));
}


void model::batch_decode(llama_batch &batch, float *output) const {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx_);

    // run model
    llama_decode(ctx_, batch);

    for (auto i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float *embed = llama_get_embeddings_seq(ctx_, batch.seq_id[i][0]);
        if (embed == nullptr) {
            embed = llama_get_embeddings_ith(ctx_, i);
            if (embed == nullptr) {
                continue;
            }
        }

        float *out = output + batch.seq_id[i][0] * n_embed_;
        llama_embd_normalize(embed, out, n_embed_);
    }
}

void model::batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, int seq_id) {
    for (auto i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, {seq_id}, i == tokens.size() - 1);
    }
}

model::~model() {
    std::unique_lock lock(mutex_);
    if (model_loaded_count_ <= 0)
        return;
    model_loaded_count_ = 0;
    llama_free(ctx_);
    llama_free_model(model_);
}
