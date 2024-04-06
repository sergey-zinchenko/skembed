//
// Created by Sergei on 4/5/2024.
//

#include "ModelInitializationHolder.h"
#include <utility>
#include "llama/llama.h"

void ModelInitializationHolder::performInitialization() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initializedCount_++ > 1)
        return;
    llama_backend_init();
    llama_numa_init(params_->numa);
}

void ModelInitializationHolder::performFinalization() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initializedCount_ <= 0)
        throw std::runtime_error("ModelInitializationHolder::performFinalization() called without initialization");
    if (initializedCount_-- > 0)
        return;
    llama_backend_free();
}

ModelInitializationHolder::ModelInitializationHolder(std::shared_ptr<gpt_params> params):
    params_(std::move(params)){}

