//
// Created by Sergei on 4/5/2024.
//

#include "index.h"

void index::add(std::string key, std::string value) {

}

void index::save(std::filesystem::path indexPath) {

}

void index::load(std::filesystem::path indexPath) {

}

index::index(std::shared_ptr<abstract_model_initialization_holder> modelInitializer,
                                   std::shared_ptr<abstract_model> model):
        modelInitializationHolder_ (std::move(modelInitializer)),
        model_(std::move(model))
{
    modelInitializationHolder_->perform_initialization();
    model_->load_model();
}

std::string index::search(std::string key) {
    return std::string();
}

index::~index() {
    model_->unload_model();
    modelInitializationHolder_->perform_finalization();
}

