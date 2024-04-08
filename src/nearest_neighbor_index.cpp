//
// Created by Sergei on 4/8/2024.
//

#include <fstream>
#include <string>
#include "nearest_neighbor_index.h"
#include "faiss/index_io.h"


void nearest_neighbor_index::add(uint32_t key, std::vector<float_t> value) {
    std::unique_lock lock(mutex_);
}

nearest_neighbor_index::nearest_neighbor_index(std::shared_ptr<spdlog::logger> logger) :
        logger_(std::move(logger)) {

}

uint32_t nearest_neighbor_index::search(std::vector<float_t> value) {
    std::shared_lock lock(mutex_);
    return 0;
}

void nearest_neighbor_index::save(std::filesystem::path indexPath) {
    std::shared_lock lock(mutex_);
    logger_->info("Saving index to {}", indexPath.string());
    faiss::write_index(index_.get(), indexPath.string().c_str());
    logger_->trace("Index saved to {}", indexPath.string());
}

void nearest_neighbor_index::load(std::filesystem::path indexPath) {
    std::unique_lock lock(mutex_);
    logger_->info("Loading index from {}", indexPath.string());
    auto p_index = faiss::read_index(indexPath.string().c_str());
    auto p_index_flat_l2 = dynamic_cast<faiss::IndexFlatL2 *>(p_index);
    if (!p_index_flat_l2) {
        delete p_index;
        throw std::runtime_error("Failed to cast index to IndexFlatL2");
    }
    index_ = std::unique_ptr<faiss::IndexFlatL2>(p_index_flat_l2);
    logger_->trace("Index loaded from {}", indexPath.string());
}