//
// Created by Sergei on 4/5/2024.
//

#include "Index.h"

#include <utility>
#include "llama/llama.h"

void Index::Add(std::string key, std::string value) {

}

void Index::Save(std::filesystem::path indexPath) {

}

void Index::Load(std::filesystem::path indexPath) {

}

void Index::initLlama(std::shared_ptr<gpt_params> &params) {
    params->embedding = true;
    params->n_ubatch = params->n_batch;
    if (params->seed == LLAMA_DEFAULT_SEED) {
        params->seed = time(nullptr);
    }
    std::mt19937 rng(params->seed);
    if (params->random_prompt) {
        params->prompt = gpt_random_prompt(rng);
    }
    llama_backend_init();
    llama_numa_init(params->numa);
}

Index::Index(std::shared_ptr<IModelInitializationHolder> llamaInitializer, std::filesystem::path modelPath) {
    llamaInitializer_ = std::move(llamaInitializer);
}

std::string Index::Search(std::string key) {
    return std::string();
}

