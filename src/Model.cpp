//
// Created by Sergei on 4/6/2024.
//

#include "Model.h"

Model::Model(gpt_params *params, std::shared_ptr<spdlog::logger> logger) :
        params_(params),
        logger_(std::move(logger)){
}

void Model::LoadModel() {
    std::lock_guard<std::mutex> lock(modelStateMutex_);
    if (modelLoadedCount_++ > 0)
        return;
    params_->embedding = true;
    params_->n_ubatch = params_->n_batch;
    std::tie(model_, ctx_) = llama_init_from_gpt_params(*params_);
    if (model_ == nullptr) {
        throw std::runtime_error("Unable to load model");
    }
    std::mt19937 rng(params_->seed);
    if (params_->random_prompt) {
        params_->prompt = gpt_random_prompt(rng);
    }
    auto n_ctx_train = llama_n_ctx_train(model_);
    auto n_ctx = llama_n_ctx(ctx_);
    if (n_ctx > n_ctx_train) {
        throw std::runtime_error("Model was trained on only " + std::to_string(n_ctx_train) +
                                 " context tokens (" + std::to_string(n_ctx) + " specified)");
    }
    nBatch = params_->n_batch;
    if (params_->n_batch < params_->n_ctx) {
        throw std::runtime_error("Batch size must be greater than context size");
    }
    nEmbed = llama_n_embd(model_);
    eosToken = llama_token_eos(model_);
}

void Model::UnloadModel() {
    std::lock_guard<std::mutex> lock(modelStateMutex_);
    if (modelLoadedCount_ <= 0)
        throw std::runtime_error("Model::UnloadModel() called without initialization");
    if (modelLoadedCount_-- > 0)
        return;
    llama_free(ctx_);
    llama_free_model(model_);
}

std::vector<std::vector<float_t>> Model::Embeddings(const std::vector<std::string> &prompts) {
    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> inputs;
    for (const auto &prompt: prompts) {
        auto inp = ::llama_tokenize(ctx_, prompt, true, false);
        if (inp.size() > nBatch) {
            throw std::runtime_error(
                    "Number of tokens in input line exceeds batch size, increase batch size and re-run");
        }
        inputs.push_back(inp);
    }

    // add eos if not present
    for (auto &inp: inputs) {
        if (inp.empty() || inp.back() != eosToken) {
            inp.push_back(eosToken);
        }
    }

    // initialize batch
    auto n_prompts = inputs.size();
    auto batch = llama_batch_init(nBatch, 0, 1);

    // allocate output
    std::vector<float> embeddings(n_prompts * nEmbed, 0);
    auto emb = embeddings.data();

    // break into batches
    auto p = 0; // number of prompts processed already
    auto s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto &inp = inputs[k];

        auto n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > nBatch) {
            auto out = emb + p * nEmbed;
            batchDecode(batch, out);
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        batchAddSeq(batch, inp, s);
        s += 1;
    }

    // final batch
    float *out = emb + p * nEmbed;
    batchDecode(batch, out);

    return std::vector<std::vector<float_t>>(n_prompts, std::vector<float_t>(emb, emb + n_prompts * nEmbed));
}

void Model::batchDecode(llama_batch &batch, float *output) {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx_);

    // run model
    llama_decode(ctx_, batch);

    for (auto i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float *embd = llama_get_embeddings_seq(ctx_, batch.seq_id[i][0]);
        if (embd == nullptr) {
            embd = llama_get_embeddings_ith(ctx_, i);
            if (embd == nullptr) {
                continue;
            }
        }

        float *out = output + batch.seq_id[i][0] * nEmbed;
        llama_embd_normalize(embd, out, nEmbed);
    }
}

void Model::batchAddSeq(llama_batch &batch, const std::vector<int32_t> &tokens, int seq_id) {
    for (auto i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, {seq_id}, i == tokens.size() - 1);
    }
}

