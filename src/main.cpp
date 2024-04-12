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
#include <ranges>

struct skill {
    int id = 0;
    std::string name;
    std::string parent_skill_name;
    int node_level{};
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
        logger->info("Process start");
        logger->info("Creating index and db connection");
        auto index = injector.create<std::unique_ptr<index_of_embeddings>>();
        auto connection = zxorm::Connection<SkillsTable>("C:/Users/sergei/Downloads/skills.sqlite");
        logger->info("Querying skills");
        auto skills = connection.select_query<SkillsTable>()
                              .order_by < SkillsTable::field_t < "id" >> (zxorm::order_t::ASC)
                .where_many( SkillsTable::field_t<"path">().like("%.NET%") || SkillsTable::field_t<"path">().like("%C#%"))
                .exec().to_vector();
        if (skills.size() == 0) {
            logger->error("No skills found");
            return 1;
        }
        logger->info("Skills count = {}", skills.size());

        auto batch_size = skills.size() / 10;

        for (int i = 0; i < 10; ++i) {
            logger->info("Processing batch {}", i);
            logger->info("Skills in batch {}", i);
            auto skill_path_vector = std::vector<std::string>{};
            auto skill_id_vector = std::vector<faiss::idx_t>{};
            for (const auto &skill: skills | std::views::drop(i * batch_size) | std::views::take(batch_size)) {
                logger->info("id = {}, path = {}", skill.id, skill.path);
                skill_path_vector.push_back(skill.path);
                skill_id_vector.push_back(skill.id);
            }
            logger->info("Adding {} skills of batch {} to index", skill_path_vector.size(), i);
            index->add(skill_id_vector, skill_path_vector);
            logger->info("Saving index of batch {}", i);
            index->save("skills_" + std::to_string(i) + ".idx");
            logger->info("Index of batch {} saved", i);
        }
    }
    catch (const zxorm::Error &e) {
        logger->error(e.what());
        return 1;
    }
    catch (const std::exception &e) {
        logger->error(e.what());
        return 1;
    }
}