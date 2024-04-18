//
// Created by Sergei on 4/5/2024.
//

#include "index_of_embeddings.h"

void index_of_embeddings::add(const std::vector<faiss::idx_t> &keys, const std::vector<std::string> &values) {
    auto embeddings = model_->embed(values);
    index_delegate_->add(keys, embeddings);
}

void index_of_embeddings::save(const std::filesystem::path &index_path) {
    index_delegate_->save(index_path);
}

void index_of_embeddings::load(const std::filesystem::path &index_path) {
    index_delegate_->load(index_path);
}

index_of_embeddings::index_of_embeddings(std::shared_ptr<abstract_model> model,
                                         std::shared_ptr<abstract_index<faiss::idx_t, flat_embed, faiss::idx_t>> index_delegate,
                                         std::shared_ptr<spdlog::logger> logger) :
        model_(std::move(model)),
        index_delegate_(std::move(index_delegate)),
        logger_(std::move(logger)) {
}

auto index_of_embeddings::search(const std::vector<std::string> &value,
                                 faiss::idx_t number_of_extracted_results) -> std::vector<std::vector<faiss::idx_t>> {
    auto embeddings = model_->embed(value);
    return index_delegate_->search(embeddings, number_of_extracted_results);
}

