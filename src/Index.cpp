//
// Created by Sergei on 4/5/2024.
//

#include "Index.h"

#include <utility>

void Index::Add(std::string key, std::string value) {

}

void Index::Save(std::filesystem::path indexPath) {

}

void Index::Load(std::filesystem::path indexPath) {

}

Index::Index(std::shared_ptr<IModelInitializationHolder> modelInitializer,
             std::shared_ptr<IModel> model):
        modelInitializationHolder_ (std::move(modelInitializer)),
        model_(std::move(model))
{
    modelInitializationHolder_->PerformInitialization();
    model_->LoadModel();
}

std::string Index::Search(std::string key) {
    return std::string();
}

Index::~Index() {
    model_->UnloadModel();
    modelInitializationHolder_->PerformFinalization();
}

