//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <string>
#include <vector>

class abstract_model {
public:
    virtual ~abstract_model() = default;
    virtual void load_model() = 0;
    virtual void unload_model() = 0;
    virtual std::vector<std::vector<float_t>> embeddings(const std::vector<std::string> &prompts) = 0;
};