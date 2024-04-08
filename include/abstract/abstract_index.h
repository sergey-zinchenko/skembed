//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <string>
#include <filesystem>

template<typename KeyType, typename ValueType>
class abstract_index {
public:
    virtual ~abstract_index() = default;
    virtual void add(KeyType key, ValueType value) = 0;
    virtual ValueType search(KeyType value) = 0;
    virtual void save(std::filesystem::path indexPath) = 0;
    virtual void load(std::filesystem::path indexPath) = 0;
};

