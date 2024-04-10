//
// Created by Sergei on 4/5/2024.
//

#include "index_of_embeddings.h"

void index_of_embeddings::add(std::vector<faiss::idx_t> keys, std::vector<std::string> values) {
    auto embeddings = model_->embed(values.begin(), values.end());
    index_delegate_->add(keys, embeddings);
}

void index_of_embeddings::save(std::filesystem::path indexPath) {
    index_delegate_->save(indexPath);
}

void index_of_embeddings::load(std::filesystem::path indexPath) {
    index_delegate_->load(indexPath);
}

index_of_embeddings::index_of_embeddings(std::shared_ptr<abstract_model> model,
                                         std::shared_ptr<abstract_index<faiss::idx_t, std::shared_ptr<abstract_flat_embed>, faiss::idx_t>> index_delegate,
                                         std::shared_ptr<spdlog::logger> logger) :
        model_(std::move(model)),
        index_delegate_(std::move(index_delegate)),
        logger_(std::move(logger)) {
}

std::vector<std::vector<faiss::idx_t>>
index_of_embeddings::search(std::vector<std::string> value, faiss::idx_t number_of_extracted_results) {
    auto embeddings = model_->embed(value.begin(), value.end());
    return index_delegate_->search(embeddings, number_of_extracted_results);
}

