//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include <mutex>
#include "abstract/IModelInitializationHolder.h"

class ModelInitializationHolder : public IModelInitializationHolder {
private:
    int initializedCount_ = 0;
    std::mutex mutex_;
public:
    ~ModelInitializationHolder() override = default;
    void performInitialization() override;
    void performFinalization() override;
};