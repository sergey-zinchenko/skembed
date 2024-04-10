//
// Created by Sergei on 4/6/2024.
//

#include "llamacpp_model.h"

llamacpp_model::llamacpp_model(gpt_params params, const std::shared_ptr<abstract_backend> &model_backend,
                               std::shared_ptr<spdlog::logger> logger) :
        model_backend_(model_backend),
        logger_(std::move(logger)) {
    logger_->trace("Loading model");
    params.embedding = true;
    params.n_ubatch = params.n_batch;
    model_ = init_model_from_gpt_params(params);

//    std::mt19937 rng(params_.seed);
//    if (params_.random_prompt) {
//        params_.prompt = gpt_random_prompt(rng);
//    }
//    auto n_ctx_train = llama_n_ctx_train(model_);
//    auto n_ctx = llama_n_ctx(ctx_);
//    if (n_ctx > n_ctx_train) {
//        logger_->warn("Model was trained on only {} context tokens ({} specified)", n_ctx_train, n_ctx);
//    }
//    n_batch_ = params_.n_batch;
//    if (params_.n_batch < params_.n_ctx) {
//        throw std::runtime_error("Batch size must be greater than context size");
//    }
//    n_embed_ = llama_n_embd(model_);
//    eos_token_ = llama_token_eos(model_);
    logger_->info("Model loaded");
}


//std::vector<std::vector<float_t>> model::embeddings(const std::vector<std::string> &prompts) {
//    logger_->trace("Embedding {} prompts", prompts.size());
//    std::shared_lock lock(mutex_);
//    auto tokenized_prompts = tokenize_and_trim(prompts);
//    auto flat_embeddings = process_tokenized_prompts(tokenized_prompts);
//    auto embeddings = reshape_embeddings(flat_embeddings);
//    logger_->trace("Prompts embedded");
////    for (const auto& vec : result) {
////        for (int i = 0; i < std::min(16, static_cast<int>(vec.size())); ++i) {
////            logger_->info("Embedding value {}: {}", i, vec[i]);
////        }
////    }
//    return embeddings;
//}


llamacpp_model::~llamacpp_model() {
    logger_->trace("Freeing model");
    llama_free_model(model_);
    logger_->info("Model freed");
}

std::shared_ptr<abstract_embedding_context> llamacpp_model::create_embedding_context() {
    return nullptr;
}

struct llama_model *llamacpp_model::init_model_from_gpt_params(const gpt_params &params) {
    llama_model *model;

    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        model = llama_load_model_from_hf(params.hf_repo.c_str(), params.hf_file.c_str(), params.model.c_str(),
                                         llama_model_params_from_gpt_params(params));
    } else if (!params.model_url.empty()) {
        model = llama_load_model_from_url(params.model_url.c_str(), params.model.c_str(),
                                          llama_model_params_from_gpt_params(params));
    } else {
        model = llama_load_model_from_file(params.model.c_str(), llama_model_params_from_gpt_params(params));
    }

    if (!model)
        throw std::runtime_error("Failed to load model " + params.model);

    for (unsigned int i = 0; i < params.lora_adapter.size(); ++i) {
        const std::string &lora_adapter = std::get<0>(params.lora_adapter[i]);
        float lora_scale = std::get<1>(params.lora_adapter[i]);
        int err = llama_model_apply_lora_from_file(model,
                                                   lora_adapter.c_str(),
                                                   lora_scale,
                                                   ((i > 0) || params.lora_base.empty())
                                                   ? nullptr
                                                   : params.lora_base.c_str(),
                                                   params.n_threads);
        if (err != 0) {
            llama_free_model(model);
            throw std::runtime_error("Failed to apply lora adapter");
        }
    }

    return model;
}
