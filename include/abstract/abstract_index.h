//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <filesystem>
#include <vector>

template<typename KeyType, typename ValueType, typename ResultSizeType>
struct abstract_index {
public:
    virtual ~abstract_index() = default;

    virtual void add(const std::vector<KeyType> &keys, const ValueType &values) = 0;

    [[nodiscard]] virtual auto search(const ValueType &value,
                                      ResultSizeType number_of_extracted_results) -> std::vector<std::vector<KeyType>> = 0;

    virtual void save(const std::filesystem::path &indexPath) = 0;

    virtual void load(const std::filesystem::path &indexPath) = 0;
};

