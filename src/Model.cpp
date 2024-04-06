//
// Created by Sergei on 4/6/2024.
//

#include "Model.h"

Model::Model(gpt_params *params) :
        params_(params) {
    params_->embedding = true;
    params_->n_ubatch = params_->n_batch;
}

void Model::LoadModel() {
    std::lock_guard<std::mutex> lock(modelStateMutex_);
    if (modelLoadedCount_++ > 0)
        return;
    std::tie(model_, ctx_) = llama_init_from_gpt_params(*params_);
    if (model_ == nullptr) {
        throw std::runtime_error("Unable to load model");
    }
    n_ctx_train = llama_n_ctx_train(model_);
    n_ctx = llama_n_ctx(ctx_);
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

std::vector<float_t> Model::Embeddings(std::string text) {
    std::vector<int32_t> tokens = ::llama_tokenize(ctx_, text, true, false);
    const float *embd = llama_get_embeddings_seq(ctx_, tokens[0]);
    if (embd == nullptr) {
        throw std::runtime_error("Failed to get embeddings");
    }
    std::vector<float_t> embeddings(embd, embd + llama_n_embd(model_));
    return embeddings;
}

