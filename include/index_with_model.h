//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include "abstract/abstract_model_initialization_holder.h"
#include <memory>
#include "abstract/abstract_index_with_model.h"
#include "abstract/abstract_model.h"

class index_with_model: public abstract_index_with_model {
public:
    index_with_model(std::shared_ptr<abstract_model_initialization_holder> modelInitializer,
                     std::shared_ptr<abstract_model> model);
    ~index_with_model() override;
    void add(std::string key, std::string value) override;
    std::string search(std::string key) override;
    void save(std::filesystem::path indexPath) override;
    void load(std::filesystem::path indexPath) override;
private:
    std::shared_ptr<abstract_model_initialization_holder> modelInitializationHolder_;
    std::shared_ptr<abstract_model> model_;
};