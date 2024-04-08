//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include "abstract/abstract_model_initialization_holder.h"
#include "abstract/abstract_index.h"
#include "abstract/abstract_model.h"
#include "spdlog/logger.h"
#include "faiss/MetricType.h"

class index_of_embeddings: public abstract_index<faiss::idx_t, std::string, faiss::idx_t> {
public:
    index_of_embeddings(std::shared_ptr<abstract_model_initialization_holder> modelInitializer,
                        std::shared_ptr<abstract_model> model,
                        std::shared_ptr<abstract_index<faiss::idx_t, std::vector<float_t>, faiss::idx_t>> index_delegate,
                        std::shared_ptr<spdlog::logger> logger);
    ~index_of_embeddings() override;
    void add(std::vector<faiss::idx_t> key, std::vector<std::string> value) override;
    std::vector<faiss::idx_t> search(std::string value, faiss::idx_t number_of_extracted_results) override;
    void save(std::filesystem::path indexPath) override;
    void load(std::filesystem::path indexPath) override;
private:
    std::shared_ptr<abstract_model_initialization_holder> modelInitializationHolder_;
    std::shared_ptr<abstract_model> model_;
    std::shared_ptr<abstract_index<faiss::idx_t, std::vector<float_t>, faiss::idx_t>> index_delegate_;
    std::shared_ptr<spdlog::logger> logger_;
};