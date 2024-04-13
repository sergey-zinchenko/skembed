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
                        std::shared_ptr<abstract_index<faiss::idx_t, flat_embed, faiss::idx_t>> index_delegate,
                        std::shared_ptr<spdlog::logger> logger);

    void add(const std::vector<faiss::idx_t> &key, const std::vector<std::string> &value) override;

    [[nodiscard]] std::vector<std::vector<faiss::idx_t>>
    search(const std::vector<std::string> &value, faiss::idx_t number_of_extracted_results) override;

    void save(const std::filesystem::path &indexPath) override;

    void load(const std::filesystem::path &indexPath) override;

private:
    std::shared_ptr<abstract_model> model_;
    std::shared_ptr<abstract_index<faiss::idx_t, flat_embed, faiss::idx_t>> index_delegate_;
    std::shared_ptr<spdlog::logger> logger_;
};