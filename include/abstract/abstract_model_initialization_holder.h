//
// Created by Sergei on 4/5/2024.
//

#pragma once

class abstract_model_initialization_holder {
public:
    virtual ~abstract_model_initialization_holder() = default;
    virtual void perform_initialization() = 0;
    virtual void perform_finalization() = 0;
};
