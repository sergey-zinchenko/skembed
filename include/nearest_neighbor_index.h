//
// Created by Sergei on 4/8/2024.
//

#pragma once

#include <shared_mutex>
#include "abstract/abstract_index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include "faiss/MetricType.h"
#include "spdlog/logger.h"
#include "flat_embed.h"

class nearest_neighbor_index : public abstract_index<faiss::idx_t, flat_embed, faiss::idx_t> {
public:
    explicit nearest_neighbor_index(std::shared_ptr<spdlog::logger> logger);

    void add(const std::vector<faiss::idx_t> &keys, const flat_embed &values) override;

    [[nodiscard]] auto
    search(const flat_embed &values,
           faiss::idx_t number_of_extracted_results) -> std::vector<std::vector<faiss::idx_t>> override;

    void save(const std::filesystem::path &indexPath) override;

    void load(const std::filesystem::path &indexPath) override;

private:
    [[nodiscard]] static auto reshape_vectors(const std::vector<faiss::idx_t> &flat,
                                              int row_size) -> std::vector<std::vector<faiss::idx_t>>;

    std::shared_ptr<spdlog::logger> logger_;
    std::shared_mutex mutex_;
    std::shared_ptr<faiss::Index> index_{};
};