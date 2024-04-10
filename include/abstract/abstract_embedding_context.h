//
// Created by Sergei on 4/10/2024.
//

#pragma once

#include <string>
#include "abstract_flat_embed.h"

class abstract_embedding_context {
public:
    virtual ~abstract_embedding_context() = default;

    [[nodiscard]] virtual std::shared_ptr<abstract_flat_embed>
    embed(std::vector<std::string>::iterator start, std::vector<std::string>::iterator end) = 0;
};

