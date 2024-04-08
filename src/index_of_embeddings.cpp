//
// Created by Sergei on 4/5/2024.
//

#include "index_of_embeddings.h"

void index_of_embeddings::add(std::vector<faiss::idx_t> keys, std::vector<std::string> values) {
    auto embeddings = model_->embeddings(values);
    index_delegate_->add(keys, embeddings);
}

void index_of_embeddings::save(std::filesystem::path indexPath) {
    index_delegate_->save(indexPath);
}

void index_of_embeddings::load(std::filesystem::path indexPath) {
    index_delegate_->load(indexPath);
}

index_of_embeddings::index_of_embeddings(std::shared_ptr<abstract_model_initialization_holder> modelInitializer,
                                         std::shared_ptr<abstract_model> model,
                                         std::shared_ptr<abstract_index<faiss::idx_t, std::vector<float_t>, faiss::idx_t>> index_delegate,
                                         std::shared_ptr<spdlog::logger> logger) :
        modelInitializationHolder_(std::move(modelInitializer)),
        model_(std::move(model)),
        index_delegate_(std::move(index_delegate)),
        logger_(std::move(logger)) {
    modelInitializationHolder_->perform_initialization();
    model_->load_model();
}

std::vector<faiss::idx_t> index_of_embeddings::search(std::string value, faiss::idx_t number_of_extracted_results) {
    auto embeddings = model_->embeddings(std::vector<std::string>{value});
    return index_delegate_->search(embeddings[0], number_of_extracted_results);
}

index_of_embeddings::~index_of_embeddings() {
    model_->unload_model();
    modelInitializationHolder_->perform_finalization();
}

