//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include "abstract/abstract_model_initialization_holder.h"
#include <memory>
#include "abstract/abstract_index.h"
#include "abstract/abstract_model.h"
#include "spdlog/logger.h"

class index_of_embeddings: public abstract_index<std::string, std::string> {
public:
    index_of_embeddings(std::shared_ptr<abstract_model_initialization_holder> modelInitializer,
                        std::shared_ptr<abstract_model> model,
                        std::shared_ptr<abstract_index<std::string, std::vector<float_t>>> index_delegate,
                        std::shared_ptr<spdlog::logger> logger);
    ~index_of_embeddings() override;
    void add(std::string key, std::string value) override;
    std::string search(std::string value) override;
    void save(std::filesystem::path indexPath) override;
    void load(std::filesystem::path indexPath) override;
private:
    std::shared_ptr<abstract_model_initialization_holder> modelInitializationHolder_;
    std::shared_ptr<abstract_model> model_;
    std::shared_ptr<abstract_index<std::string, std::vector<float_t>>> index_delegate_;
    std::shared_ptr<spdlog::logger> logger_;
};