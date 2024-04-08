//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <string>
#include <filesystem>

class abstract_index_with_model {
public:
    virtual ~abstract_index_with_model() = default;
    virtual void add(std::string key, std::string value) = 0;
    virtual std::string search(std::string key) = 0;
    virtual void save(std::filesystem::path indexPath) = 0;
    virtual void load(std::filesystem::path indexPath) = 0;
};
