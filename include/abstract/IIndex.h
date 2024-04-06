//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <string>
#include <filesystem>

class IIndex {
public:
    ~IIndex() = default;
    virtual void Add(std::string key, std::string value) = 0;
    virtual std::string Search(std::string key) = 0;
    virtual void Save(std::filesystem::path indexPath) = 0;
    virtual void Load(std::filesystem::path indexPath) = 0;
};
