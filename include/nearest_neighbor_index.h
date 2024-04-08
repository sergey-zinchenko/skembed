//
// Created by Sergei on 4/8/2024.
//

#pragma once

#include <string>
#include <filesystem>
#include "abstract/abstract_index.h"

class nearest_neighbor_index: public abstract_index<std::string, float_t> {
public:
    nearest_neighbor_index();
    void add(std::string key, float_t value) override;
    std::string search(float_t value) override;
    void save(std::filesystem::path indexPath) override;
    void load(std::filesystem::path indexPath) override;
private:
};