//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include <string>
#include <filesystem>
#include <memory>
#include "llama/common.h"
#include "abstract/IModelInitializationHolder.h"
#include "abstract/IIndex.h"

class Index: public IIndex {
public:
    Index(std::shared_ptr<IModelInitializationHolder> llamaInitializer);
    ~Index() override;
    void Add(std::string key, std::string value) override;
    std::string Search(std::string key) override;
    void Save(std::filesystem::path indexPath) override;
    void Load(std::filesystem::path indexPath) override;
private:
    std::shared_ptr<IModelInitializationHolder> llamaInitializationHolder_;

    std::shared_ptr<llama_model> model_;
    std::shared_ptr<llama_context> ctx_;
    void initLlama(std::shared_ptr<gpt_params> &params);
};


