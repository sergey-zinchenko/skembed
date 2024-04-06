//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include <mutex>
#include "abstract/IModelInitializationHolder.h"
#include "llama/common.h"

class ModelInitializationHolder : public IModelInitializationHolder {
private:
    int64_t initializedCount_ = 0;
    std::mutex mutex_;
    std::shared_ptr<gpt_params> params_;
public:
    explicit ModelInitializationHolder(std::shared_ptr<gpt_params> params);
    ~ModelInitializationHolder() override = default;
    void PerformInitialization() override;
    void PerformFinalization() override;
};