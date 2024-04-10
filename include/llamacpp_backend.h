//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include "abstract/abstract_backend.h"
#include "llama/common.h"
#include "spdlog/logger.h"

class llamacpp_backend : public abstract_backend {
public:
    explicit llamacpp_backend(const gpt_params &params, std::shared_ptr<spdlog::logger> logger);

    ~llamacpp_backend() override;

private:
    std::shared_ptr<spdlog::logger> logger_;
};