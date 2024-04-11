#ifdef _WIN32

#include <windows.h>

#endif

#include "index_of_embeddings.h"
#include "model.h"
#include "model_backend.h"
#include "nearest_neighbor_index.h"

#include "spdlog/logger.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "flat_embed.h"

#include <boost/di.hpp>

namespace di = boost::di;

std::shared_ptr<spdlog::logger> create_logger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
    console_sink->set_pattern("[%^%l%$] %v");

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
    file_sink->set_level(spdlog::level::trace);

    auto logger = std::make_shared<spdlog::logger>("multi_sink",
                                                   std::initializer_list<spdlog::sink_ptr>{console_sink, file_sink});
    logger->set_level(spdlog::level::warn);
    return logger;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    auto logger = create_logger();

    auto injector = di::make_injector(
            di::bind<abstract_model_backend>().to<model_backend>().in(di::singleton),
            di::bind<std::function<std::shared_ptr<abstract_flat_embed>(size_t, size_t)>>().to(
                    [](const auto &injector) {
                        return [](size_t row_size, size_t rows) {
                            return std::make_shared<flat_embed>(rows, row_size);
                        };
                    }
            ),
            boost::di::bind<abstract_model>().to<model>().in(di::singleton),
            di::bind<abstract_index<faiss::idx_t, std::shared_ptr<abstract_flat_embed>, faiss::idx_t>>().to<nearest_neighbor_index>(),
            di::bind<gpt_params>().to(params),
            di::bind<spdlog::logger>().to(logger)
    );
    try {
        std::unordered_map<faiss::idx_t, std::string> input_texts = {
                {1, "Космические корабли бараздят просторы большого театра"},
                {2, "Двигатель внутреннего сгорания сгорает изнутри"},
                {3, "Солнечный ветер толкает корабль к северному полюсу"},
                {4, "Если бы у рыбы были блохи то это была бы собака"},
                {5, "Чтобы приготовить яишницу надо найти гнездо курицы"},
                {6, "Мотыга капает землю лучше чем лопата"},
                {7, "Хуанг сказал что скоро видеокарты заменять программистов"}
        };

        std::vector<faiss::idx_t> keys;
        std::vector<std::string> values;
        for (const auto &pair: input_texts) {
            keys.push_back(pair.first);
            values.push_back(pair.second);
        }

        auto index = injector.create<std::unique_ptr<index_of_embeddings>>();
        index->add(keys, values);

        index->save("logs/index.idx");
        index->load("logs/index.idx");

        std::unordered_map<std::string, int> search_texts = {
                {"Приготовление яишничы на костре - хорошая прилюдия к завтраку", 1},
                {"Nvidia - компания которая проивзодит ускорители",               1},
                {"Анекдот про блох рассказал студент на экзамене по зоологии",    1},
                {"В фильме про шурика корабли бараздили просторы чего то там",    1},
                {"Эксковатор применяют для рытья котлованов в наши дня",          1},
                {"У субару опозитный мотор",                                      1},
                {"Вояджер вылетел за пределы солнечной системы",                  1}
        };

        for (const auto &pair: search_texts) {
            auto r1 = index->search({pair.first}, pair.second);
            for (const auto &result: r1) {
                for (const auto &i: result) {
                    logger->warn("{} = {}", pair.first, input_texts[i]);
                }
            }
        }


    }
    catch (const std::exception &e) {
        logger->error(e.what());
        return 1;
    }
}