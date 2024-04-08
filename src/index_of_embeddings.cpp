//
// Created by Sergei on 4/5/2024.
//

#include "index_of_embeddings.h"

void index_of_embeddings::add(std::string key, std::string value) {

}

void index_of_embeddings::save(std::filesystem::path indexPath) {

}

void index_of_embeddings::load(std::filesystem::path indexPath) {

}

index_of_embeddings::index_of_embeddings(std::shared_ptr<abstract_model_initialization_holder> modelInitializer,
                                         std::shared_ptr<abstract_model> model):
        modelInitializationHolder_ (std::move(modelInitializer)),
        model_(std::move(model))
{
    modelInitializationHolder_->perform_initialization();
    model_->load_model();
}

std::string index_of_embeddings::search(std::string value) {
    return std::string();
}

index_of_embeddings::~index_of_embeddings() {
    model_->unload_model();
    modelInitializationHolder_->perform_finalization();
}

