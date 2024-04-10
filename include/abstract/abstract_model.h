//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <string>
#include <vector>
#include "abstract_flat_embed.h"

class abstract_model {
public:
    virtual ~abstract_model() = default;
    [[nodiscard]] virtual std::shared_ptr<abstract_flat_embed>
    embed(const std::vector<std::string>::iterator &prompts_start, const std::vector<std::string>::iterator &prompts_end) = 0;
};