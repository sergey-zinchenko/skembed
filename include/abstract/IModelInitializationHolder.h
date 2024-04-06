//
// Created by Sergei on 4/5/2024.
//

#pragma once

class IModelInitializationHolder {
public:
    ~IModelInitializationHolder() = default;
    virtual void performInitialization() = 0;
    virtual void performFinalization() = 0;
};
