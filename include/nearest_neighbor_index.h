//
// Created by Sergei on 4/8/2024.
//

#pragma once

#include <shared_mutex>
#include "abstract/abstract_index.h"
#include "faiss/IndexFlat.h"
#include "faiss/MetricType.h"
#include "spdlog/logger.h"

class nearest_neighbor_index: public abstract_index<faiss::idx_t, std::vector<float_t>, faiss::idx_t> {
public:
    explicit nearest_neighbor_index(std::shared_ptr<spdlog::logger> logger);
    void add(std::vector<faiss::idx_t> keys, std::vector<std::vector<float_t>> values) override;
    std::vector<faiss::idx_t> search(std::vector<float_t> value, faiss::idx_t number_of_extracted_results) override;
    void save(std::filesystem::path indexPath) override;
    void load(std::filesystem::path indexPath) override;
private:
    std::shared_ptr<spdlog::logger> logger_;
    std::unique_ptr<faiss::IndexFlatL2> index_;
    std::shared_mutex mutex_;
};