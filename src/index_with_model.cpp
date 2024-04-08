//
// Created by Sergei on 4/5/2024.
//

#include "index_with_model.h"

void index_with_model::add(std::string key, std::string value) {

}

void index_with_model::save(std::filesystem::path indexPath) {

}

void index_with_model::load(std::filesystem::path indexPath) {

}

index_with_model::index_with_model(std::shared_ptr<abstract_model_initialization_holder> modelInitializer,
                                   std::shared_ptr<abstract_model> model):
        modelInitializationHolder_ (std::move(modelInitializer)),
        model_(std::move(model))
{
    modelInitializationHolder_->perform_initialization();
    model_->load_model();
}

std::string index_with_model::search(std::string value) {
    return std::string();
}

index_with_model::~index_with_model() {
    model_->unload_model();
    modelInitializationHolder_->perform_finalization();
}

