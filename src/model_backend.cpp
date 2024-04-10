//
// Created by Sergei on 4/5/2024.
//

#include "model_backend.h"

model_backend::model_backend(const gpt_params &params, std::shared_ptr<spdlog::logger> logger) :
        logger_(std::move(logger)) {
    logger_->trace("Initializing model backend and setting up numa");
    llama_backend_init();
    llama_numa_init(params.numa);
    logger_->trace("Model backend initialized");
}

model_backend::~model_backend() {
    logger_->trace("Freeing model backend");
    llama_backend_free();
    logger_->trace("Model backend freed");
}

