//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <string>
#include <vector>
#include "flat_embed.h"

struct abstract_model {
public:
    virtual ~abstract_model() = default;

    [[nodiscard]] virtual flat_embed
    embed(const std::vector<std::string> &prompts) = 0;
};