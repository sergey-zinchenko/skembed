//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include "flat_embed.h"
#include <string>
#include <vector>

struct abstract_model {
public:
    virtual ~abstract_model() = default;

    [[nodiscard]] virtual auto
    embed(const std::vector<std::string> &prompts) -> flat_embed = 0;
};