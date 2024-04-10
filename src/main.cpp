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
    logger->set_level(spdlog::level::trace);
    return logger;
}

int main(int argc, char **argv) {

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
        auto index = injector.create<std::unique_ptr<index_of_embeddings>>();
        index->add({1, 2, 3, 4, 5, 6, 7}, {"Космические корабли бараздят просторы большого театра",
                                           "Двигатель внутреннего сгорания сгорает изнутри",
                                           "Солнечный ветер толкает корабль к северному полюсу",
                                           "Если бы у рыбы были блохи то это была бы собака",
                                           "Чтобы приготовить яишницу надо найти гнездо курицы",
                                           "Мотыга капает землю лучше чем лопата",
                                           "Хуанг сказал что скоро видеокарты заменять программистов"});
        index->save("logs/index.idx");
        index->load("logs/index.idx");
        auto r1 = index->search({"Приготовление яишничы на костре - хорошая прилюдия к завтраку",
                                 "Nvidia - компания которая проивзодит ускорители",
                                 "Анекдот про блох рассказал студент на экзамене по зоологии",
                                 "В фильме про шурика корабли бараздили просторы чего то там",
                                 "Эксковатор применяют для рытья котлованов в наши дня",
                                 "У субару опозитный мотор",
                                 "Вояджер вылетел за пределы солнечной системы"}, 2);
        for (const auto &v : r1) {
            for (auto i : v) {
                std::cout << i << " ";
            }
            std::cout << std::endl;

        }
    }
    catch (const std::exception &e) {
        logger->error(e.what());
        return 1;
    }
}