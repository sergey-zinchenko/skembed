//
// Created by Sergei on 4/8/2024.
//

#pragma once

#include <filesystem>
#include <shared_mutex>
#include "abstract/abstract_index.h"
#include "faiss/IndexFlat.h"
#include "faiss/MetricType.h"
#include "spdlog/logger.h"

class nearest_neighbor_index: public abstract_index<uint32_t, std::vector<float_t>> {
public:
    explicit nearest_neighbor_index(std::shared_ptr<spdlog::logger> logger);
    void add(uint32_t key, std::vector<float_t> value) override;
    uint32_t search(std::vector<float_t> value) override;
    void save(std::filesystem::path indexPath) override;
    void load(std::filesystem::path indexPath) override;
private:
    std::shared_ptr<spdlog::logger> logger_;
    std::unique_ptr<faiss::IndexFlatL2> index_;
    std::shared_mutex mutex_;
};