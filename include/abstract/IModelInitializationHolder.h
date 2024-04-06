//
// Created by Sergei on 4/5/2024.
//

#pragma once

class IModelInitializationHolder {
public:
    virtual ~IModelInitializationHolder() = default;
    virtual void PerformInitialization() = 0;
    virtual void PerformFinalization() = 0;
};
