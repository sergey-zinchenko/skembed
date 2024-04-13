//
// Created by Sergei on 4/8/2024.
//

#include <fstream>
#include <string>
#include <utility>
#include "nearest_neighbor_index.h"
#include "faiss/index_io.h"

nearest_neighbor_index::nearest_neighbor_index(std::shared_ptr<spdlog::logger> logger) :
        logger_(std::move(logger)) {
}

void nearest_neighbor_index::save(const std::filesystem::path &indexPath) {
    logger_->trace("Saving index to {}", indexPath.string());
    std::shared_lock lock(mutex_);
    faiss::write_index(index_.get(), indexPath.string().c_str());
    logger_->trace("Index saved to {}", indexPath.string());
}

void nearest_neighbor_index::load(const std::filesystem::path &indexPath) {
    logger_->trace("Loading index from {}", indexPath.string());
    std::unique_lock lock(mutex_);
    auto p_index = std::shared_ptr<faiss::Index>(faiss::read_index(indexPath.string().c_str()));
    auto p_index_id_map = std::dynamic_pointer_cast<faiss::IndexIDMap>(p_index);
    if (!p_index_id_map) {
        throw std::runtime_error("Failed to cast index to IndexIDMap");
    }
    auto *index_flat_l2 = dynamic_cast<faiss::IndexFlatL2 *>(p_index_id_map->index);
    if (index_flat_l2 == nullptr) {
        throw std::runtime_error("Failed to cast underlying index to IndexFlatL2");
    }
    index_ = std::move(p_index);
    logger_->trace("Index loaded with {} indexed vectors of {} dimensions", index_->ntotal, index_->d);
}

void nearest_neighbor_index::add(const std::vector<faiss::idx_t> &keys, const flat_embed &values) {
    logger_->trace("Adding {} vectors to index", keys.size());
    if (keys.size() != values.rows()) {
        throw std::runtime_error("Keys and values sizes are different");
    }
    if (keys.empty()) {
        return;
    }
    std::unique_lock lock(mutex_);
    if (index_) {
        if (index_->d != values.row_size()) {
            throw std::runtime_error("Index dimension is different from the dimension of the added vectors");
        }
    } else {
        auto *underlying_index_ = new faiss::IndexFlatL2(static_cast<faiss::idx_t>(values.row_size()));
        index_ = std::make_shared<faiss::IndexIDMap>(underlying_index_);
    }
    index_->add_with_ids(static_cast<faiss::idx_t>(values.rows()), values.data(), keys.data());
    logger_->trace("Added {} vectors to index. New index size is {}", keys.size(), index_->ntotal);
}

auto nearest_neighbor_index::search(const flat_embed &values,
                                    faiss::idx_t number_of_extracted_results) -> std::vector<std::vector<faiss::idx_t>> {
    logger_->trace("Querying index with {} values and asking for {} results", values.rows(),
                   number_of_extracted_results);
    std::shared_lock lock(mutex_);
    if (!index_) {
        throw std::runtime_error("Index is not initialized");
    }
    auto const result_size = number_of_extracted_results * values.rows();
    auto results_idxes = std::vector<faiss::idx_t>(result_size);
    auto result_distances = std::vector<float_t>(result_size);
    index_->search(static_cast<faiss::idx_t>(values.rows()), values.data(), number_of_extracted_results,
                   result_distances.data(), results_idxes.data());
    auto result = reshape_vectors(results_idxes, static_cast<int>(number_of_extracted_results));
    logger_->info("Querying index done");
    return result;
}

auto nearest_neighbor_index::reshape_vectors(const std::vector<faiss::idx_t> &flat,
                                             int row_size) -> std::vector<std::vector<faiss::idx_t>> {
    std::vector<std::vector<faiss::idx_t>> result;
    for (auto i = 0; i < flat.size(); i += row_size) {
        result.emplace_back(flat.begin() + i, flat.begin() + i + row_size);
    }
    return result;
}
