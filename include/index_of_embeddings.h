//
// Created by Sergei on 4/5/2024.
//

#pragma once

#include <vector>
#include "abstract/abstract_model_backend.h"
#include "abstract/abstract_index.h"
#include "abstract/abstract_model.h"
#include "spdlog/logger.h"
#include "faiss/MetricType.h"

class index_of_embeddings : public abstract_index<faiss::idx_t, std::vector<std::string>, faiss::idx_t> {
public:
    index_of_embeddings(std::shared_ptr<abstract_model> model,
                        std::shared_ptr<abstract_index<faiss::idx_t, std::shared_ptr<abstract_flat_embed>, faiss::idx_t>> index_delegate,
                        std::shared_ptr<spdlog::logger> logger);

    void add(std::vector<faiss::idx_t> key, std::vector<std::string> value) override;

    std::vector<std::vector<faiss::idx_t>> search(std::vector<std::string> value, faiss::idx_t number_of_extracted_results) override;

    void save(std::filesystem::path indexPath) override;

    void load(std::filesystem::path indexPath) override;

private:
    std::shared_ptr<abstract_model> model_;
    std::shared_ptr<abstract_index<faiss::idx_t, std::shared_ptr<abstract_flat_embed>, faiss::idx_t>> index_delegate_;
    std::shared_ptr<spdlog::logger> logger_;
};