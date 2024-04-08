//
// Created by Sergei on 4/8/2024.
//

#include <fstream>
#include <string>
#include "nearest_neighbor_index.h"
#include "faiss/index_io.h"


void nearest_neighbor_index::add(std::vector<faiss::idx_t> keys, std::vector<std::vector<float_t>> values)
{
    logger_->trace("Adding {} vectors to index", keys.size());
    if (keys.size() != values.size())
        throw std::runtime_error("Keys and values sizes are different");
    std::unique_lock lock(mutex_);
    if (index_ == nullptr) {
        index_ = std::make_unique<faiss::IndexFlatL2>(values[0].size());
    }
    std::vector<float_t> flat_value;
    for (const auto& v : values) {
        flat_value.insert(flat_value.end(), v.begin(), v.end());
    }
    index_->add_with_ids(static_cast<faiss::idx_t>(keys.size()), flat_value.data(), keys.data());
    logger_->trace("Added {} vectors to index", keys.size());
}

nearest_neighbor_index::nearest_neighbor_index(std::shared_ptr<spdlog::logger> logger) :
        logger_(std::move(logger)) {
}

std::vector<faiss::idx_t> nearest_neighbor_index::search(std::vector<float_t> value, faiss::idx_t number_of_extracted_results) {
    logger_->trace("Querying index with value of size {} and asking for {} results", value.size(), number_of_extracted_results);
    std::shared_lock lock(mutex_);
    if (!index_)
        throw std::runtime_error("Index is not initialized");
    auto results_idxes = new faiss::idx_t[number_of_extracted_results];
    auto result_distances = new float_t[number_of_extracted_results];
    index_->search(1, value.data(), number_of_extracted_results,
                   result_distances, results_idxes);
    std::vector<faiss::idx_t> results(results_idxes, results_idxes + number_of_extracted_results);
    logger_->trace("Querying index finished. Nearest result index is {} with distance {}", results_idxes[0], result_distances[0]);
    return results;
}

void nearest_neighbor_index::save(std::filesystem::path indexPath) {
    logger_->trace("Saving index to {}", indexPath.string());
    std::shared_lock lock(mutex_);
    faiss::write_index(index_.get(), indexPath.string().c_str());
    logger_->trace("Index saved to {}", indexPath.string());
}

void nearest_neighbor_index::load(std::filesystem::path indexPath) {
    logger_->trace("Loading index from {}", indexPath.string());
    std::unique_lock lock(mutex_);
    auto p_index = faiss::read_index(indexPath.string().c_str());
    auto p_index_flat_l2 = dynamic_cast<faiss::IndexFlatL2 *>(p_index);
    if (!p_index_flat_l2) {
        delete p_index;
        throw std::runtime_error("Failed to cast index to IndexFlatL2");
    }
    index_ = std::unique_ptr<faiss::IndexFlatL2>(p_index_flat_l2);
    logger_->trace("Index loaded with {} indexed vectors of {} dimensions", index_->ntotal, index_->d);
}