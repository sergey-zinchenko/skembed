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

void Index::initLlama(std::shared_ptr<gpt_params> &params) {
    llamaInitializationHolder_->performInitialization();
}

Index::Index(std::shared_ptr<IModelInitializationHolder> llamaInitializer):
        llamaInitializationHolder_ (std::move(llamaInitializer))
{
    llamaInitializationHolder_->performInitialization();
}

std::string Index::Search(std::string key) {
    return std::string();
}

Index::~Index() {
    llamaInitializationHolder_->performFinalization();
}

