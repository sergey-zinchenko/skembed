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
#include "zxorm/orm/table.hpp"
#include "zxorm/orm/connection.hpp"

#include <boost/di.hpp>

struct skill {
    int id = 0;
    std::string name;
    std::string parent_skill_name;
    int node_level {};
    std::string aliases;
    std::string path;
    std::string tags;
};

namespace di = boost::di;

std::shared_ptr<spdlog::logger> create_logger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("[%^%l%$] %v");

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
    file_sink->set_level(spdlog::level::trace);

    auto logger = std::make_shared<spdlog::logger>("multi_sink",
                                                   std::initializer_list<spdlog::sink_ptr>{console_sink, file_sink});
    logger->set_level(spdlog::level::trace);
    return logger;
}


using SkillsTable = zxorm::Table<"Skills", skill,
        zxorm::Column<"id", &skill::id, zxorm::PrimaryKey<>>,
        zxorm::Column<"name", &skill::name>,
        zxorm::Column<"parent_skill_name", &skill::parent_skill_name>,
        zxorm::Column<"node_level", &skill::node_level>,
        zxorm::Column<"aliases", &skill::aliases>,
        zxorm::Column<"path", &skill::path>,
        zxorm::Column<"tags", &skill::tags>>;

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    auto logger = create_logger();

    try {
        auto connection = zxorm::Connection<SkillsTable>("C:/Users/sergei/Downloads/skills.sqlite");
        auto skills = connection.select_query<SkillsTable>().limit(50).many().exec();
        for (const auto &skill: skills) {
            logger->info("id = {}, name = {}, path = {}", skill.id, skill.name, skill.path);
        }

    } catch (const zxorm::ConnectionError &e) {
        logger->critical(e.what());
    }

    auto injector = di::make_injector(
            di::bind<abstract_model_backend>().to<model_backend>().in(di::singleton),
            di::bind<std::function<std::shared_ptr<abstract_flat_embed>(size_t, size_t)>>().to(
                    [](const auto &injector) {
                        return [](size_t row_size, size_t rows) {
                            return std::make_shared<flat_embed>(rows, row_size);
                        };
                    }
            ),
            di::bind<abstract_model>().to<model>().in(di::singleton),
            di::bind<abstract_index<faiss::idx_t, std::shared_ptr<abstract_flat_embed>, faiss::idx_t>>().to<nearest_neighbor_index>(),
            di::bind<gpt_params>().to(params),
            di::bind<spdlog::logger>().to(logger)
    );
    try {
        auto index = injector.create<std::unique_ptr<index_of_embeddings>>();

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
        index->add(keys, values);


        index->save("logs/index.idx");
        index->load("logs/index.idx");


        std::vector<std::string> search_texts = {
                "Приготовление яишничы на костре - хорошая прилюдия к завтраку",
                "Nvidia - компания которая проивзодит ускорители",
                "Анекдот про блох рассказал студент на экзамене по зоологии",
                "В фильме про шурика корабли бараздили просторы чего то там",
                "Эксковатор применяют для рытья котлованов в наши дня",
                "У субару опозитный мотор",
                "Вояджер вылетел за пределы солнечной системы"
        };

        auto r1 = index->search(search_texts, 1);
        for (size_t j = 0; j < r1.size(); ++j) {
            const auto &result = r1[j];
            for (const auto &i: result) {
                logger->warn("{} = {}", search_texts[j], input_texts[i]);
            }
        }
    }
    catch (const std::exception &e) {
        logger->error(e.what());
        return 1;
    }
}