//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include <mutex>
#include "abstract/abstract_model_initialization_holder.h"
#include "llama/common.h"

class model_initialization_holder : public abstract_model_initialization_holder {
public:
    explicit model_initialization_holder(gpt_params *params);
    ~model_initialization_holder() override = default;
    void perform_initialization() override;
    void perform_finalization() override;
private:
    int64_t initialized_count_ = 0;
    std::mutex mutex_;
    gpt_params *params_;
};