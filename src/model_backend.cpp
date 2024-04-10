//
// Created by Sergei on 4/5/2024.
//

#include "model_backend.h"

#include <utility>

void model_backend::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_count_++ > 1)
        return;
    llama_backend_init();
    llama_numa_init(params_.numa);
}

void model_backend::finalize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_count_ <= 0)
        throw std::runtime_error("ModelInitializationHolder::finalize() called without initialization");
    if (initialized_count_-- > 0)
        return;
    llama_backend_free();
}

model_backend::model_backend(gpt_params params, std::shared_ptr<spdlog::logger> logger) :
        params_(std::move(params)),
        logger_(std::move(logger)) {}

