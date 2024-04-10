//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <filesystem>

template<typename KeyType, typename ValueType, typename ResultSizeType>
class abstract_index {
public:
    virtual ~abstract_index() = default;
    virtual void add(std::vector<KeyType> keys, ValueType values) = 0;
    virtual std::vector<std::vector<KeyType>> search(ValueType value, ResultSizeType number_of_extracted_results) = 0;
    virtual void save(std::filesystem::path indexPath) = 0;
    virtual void load(std::filesystem::path indexPath) = 0;
};

