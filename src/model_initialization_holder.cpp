//
// Created by Sergei on 4/5/2024.
//

#include "model_initialization_holder.h"

void model_initialization_holder::perform_initialization() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_count_++ > 1)
        return;
    llama_backend_init();
    llama_numa_init(params_->numa);
}

void model_initialization_holder::perform_finalization() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_count_ <= 0)
        throw std::runtime_error("ModelInitializationHolder::perform_finalization() called without initialization");
    if (initialized_count_-- > 0)
        return;
    llama_backend_free();
}

model_initialization_holder::model_initialization_holder(gpt_params *params):
    params_(params){}

