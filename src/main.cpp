#ifdef _WIN32

#include <windows.h>

#endif

#include "index_of_embeddings.h"
#include "model.h"
#include "model_backend.h"
#include "nearest_neighbor_index.h"

#include "flat_embed.h"
#include "csv-parser/csv.hpp"
#include "spdlog/logger.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "zxorm/orm/connection.hpp"
#include "zxorm/orm/table.hpp"

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

auto create_logger() -> std::shared_ptr<spdlog::logger> {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("[%^%l%$] %v");

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/skembed.txt", true);
    file_sink->set_level(spdlog::level::trace);

    auto logger = std::make_shared<spdlog::logger>("global",
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

auto main(int argc, char **argv) -> int {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif


    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    auto logger = create_logger();

    csv::CSVReader reader("R:\\Temp\\Packages.csv");

    for (csv::CSVRow& row: reader) { // Input iterator
        for (csv::CSVField& field: row) {
            // By default, get<>() produces a std::string.
            // A more efficient get<string_view>() is also available, where the resulting
            // string_view is valid as long as the parent CSVRow is alive
            logger->info("csv field : {}", field.get<std::string>());
        }
    }

    auto injector = di::make_injector(
            di::bind<abstract_model_backend>().to<model_backend>().in(di::singleton),
            di::bind<std::function<flat_embed(size_t, size_t)>>().to(
                    [](const auto & /*injector*/) {
                        return [](size_t row_size, size_t rows) {
                            return flat_embed{rows, row_size};
                        };
                    }
            ),
            di::bind<abstract_model>().to<model>().in(di::singleton),
            di::bind<abstract_index<faiss::idx_t, flat_embed, faiss::idx_t>>().to<nearest_neighbor_index>().in(
                    di::singleton),
            di::bind<gpt_params>().to(params),
            di::bind<spdlog::logger>().to(logger)
    );
    try {
        logger->info("Process start");
        logger->info("Creating index and db connection");
        auto index = injector.create<std::unique_ptr<index_of_embeddings>>();
        auto connection = zxorm::Connection<SkillsTable>("C:/Users/Sergei/Downloads/skills.sqlite");
        logger->info("Querying skills");
        auto skills = connection.select_query<SkillsTable>()
                              .order_by < SkillsTable::field_t < "id" >> (zxorm::order_t::ASC)
                .where_many(
                        SkillsTable::field_t<"path">().like("%.NET%") || SkillsTable::field_t<"path">().like("%C#%"))
                .exec().to_vector();
        if (skills.empty()) {
            logger->error("No skills found");
            return 1;
        }
        logger->info("Skills count = {}", skills.size());

        const auto batch_num = 10;

        const auto batch_size = skills.size() / batch_num;

        for (auto i = 0; i < batch_num; ++i) {
            logger->info("Processing batch {}", i);
            logger->info("Skills in batch {}", i);
            std::vector<std::string> skill_path_vector;
            std::vector<faiss::idx_t> skill_id_vector;
            for (const auto &skill: skills | std::views::drop(i * batch_size) | std::views::take(batch_size)) {
                logger->info("id = {}, path = {}", skill.id, skill.path);
                skill_path_vector.emplace_back(skill.path);
                skill_id_vector.emplace_back(skill.id);
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