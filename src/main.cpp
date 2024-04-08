#include "index_of_embeddings.h"
#include "model.h"
#include "model_initialization_holder.h"
#include "nearest_neighbor_index.h"

#include "spdlog/logger.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

#include <boost/di.hpp>

namespace di = boost::di;

// create a logger with 2 targets, with different log levels and formats.
// The console will show only warnings or errors, while the file will log all.

std::shared_ptr<spdlog::logger> create_logger()
{
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::warn);
    console_sink->set_pattern("[multi_sink_example] [%^%l%$] %v");

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
    file_sink->set_level(spdlog::level::trace);

    auto logger = std::make_shared<spdlog::logger>("multi_sink", std::initializer_list<spdlog::sink_ptr>{ console_sink, file_sink });
    logger->set_level(spdlog::level::debug);
    return logger;
}

int main(int argc, char** argv) {

    gpt_params params;


    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    auto logger = create_logger();

    auto injector = di::make_injector(
        di::bind<abstract_model_initialization_holder>().to<model_initialization_holder>(),
        di::bind<abstract_model>().to<model>(),
        di::bind<abstract_index<faiss::idx_t, std::vector<float_t>, faiss::idx_t>>().to<nearest_neighbor_index>(),
        di::bind<gpt_params>().to(params),
        di::bind<spdlog::logger>().to(std::move(logger))
    );
    auto index = injector.create<std::shared_ptr<index_of_embeddings>>();
}