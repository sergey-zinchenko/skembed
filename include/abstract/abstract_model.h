//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <string>
#include <vector>
#include "abstract_embedding_context.h"

class abstract_model {
public:
    virtual ~abstract_model() = default;
    [[nodiscard]] virtual std::shared_ptr<abstract_embedding_context> create_embedding_context() = 0;
};