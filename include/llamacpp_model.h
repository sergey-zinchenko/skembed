//
// Created by Sergei on 4/6/2024.
//

#pragma once

#include <shared_mutex>
#include "abstract/abstract_model.h"
#include "abstract/abstract_backend.h"
#include "llama/llama.h"
#include "llama/common.h"
#include "spdlog/spdlog.h"

class llamacpp_model : public abstract_model {
public:
    llamacpp_model(gpt_params params,
                   const std::shared_ptr<abstract_backend> &model_backend,
                   std::shared_ptr<spdlog::logger> logger);

    ~llamacpp_model() override;

    std::shared_ptr<abstract_embedding_context> create_embedding_context() override;
private:
    [[nodiscard]] static struct llama_model *init_model_from_gpt_params(const gpt_params &params);
    std::shared_ptr<abstract_backend> model_backend_;
    std::shared_ptr<spdlog::logger> logger_;
    llama_model *model_{};
};