//
// Created by Sergei on 4/8/2024.
//

#pragma once

#include <shared_mutex>
#include "abstract/abstract_index.h"
#include "abstract/abstract_flat_embed.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include "faiss/MetricType.h"
#include "spdlog/logger.h"

class nearest_neighbor_index: public abstract_index<faiss::idx_t, std::shared_ptr<abstract_flat_embed>, faiss::idx_t> {
public:
    explicit nearest_neighbor_index(std::shared_ptr<spdlog::logger> logger);
    void add(std::vector<faiss::idx_t> keys, std::shared_ptr<abstract_flat_embed> values) override;
    std::vector<std::vector<faiss::idx_t>> search(std::shared_ptr<abstract_flat_embed> values, faiss::idx_t number_of_extracted_results) override;
    void save(std::filesystem::path indexPath) override;
    void load(std::filesystem::path indexPath) override;
private:
    static std::vector<std::vector<faiss::idx_t>> reshape_vectors(const faiss::idx_t* flat, size_t total, int row_size);
    std::shared_ptr<spdlog::logger> logger_;
    std::shared_mutex mutex_;
    std::unique_ptr<faiss::IndexIDMap> index_{};
};