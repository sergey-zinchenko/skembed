//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include "abstract/abstract_model_backend.h"
#include "llama/common.h"
#include "spdlog/logger.h"

class model_backend : public abstract_model_backend {
public:
    explicit model_backend(const gpt_params &params, std::shared_ptr<spdlog::logger> logger);

    ~model_backend() override;

private:
    std::shared_ptr<spdlog::logger> logger_;
};