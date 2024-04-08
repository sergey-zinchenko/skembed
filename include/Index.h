//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include "abstract/IModelInitializationHolder.h"
#include <memory>
#include "abstract/IIndex.h"
#include "abstract/IModel.h"

class Index: public IIndex {
public:
    Index(std::shared_ptr<IModelInitializationHolder> modelInitializer,
          std::shared_ptr<IModel> model);
    ~Index() override;
    void Add(std::string key, std::string value) override;
    std::string Search(std::string key) override;
    void Save(std::filesystem::path indexPath) override;
    void Load(std::filesystem::path indexPath) override;
private:
    std::shared_ptr<IModelInitializationHolder> modelInitializationHolder_;
    std::shared_ptr<IModel> model_;
};