//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include <mutex>
#include "abstract/abstract_model_backend.h"
#include "llama/common.h"
#include "spdlog/logger.h"

class model_backend : public abstract_model_backend {
public:
    explicit model_backend(gpt_params params, std::shared_ptr<spdlog::logger> logger);
    void initialize() override;
    void finalize() override;
private:
    std::shared_ptr<spdlog::logger> logger_;
    int64_t initialized_count_ = 0;
    std::mutex mutex_;
    gpt_params params_;
};