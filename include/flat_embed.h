//
// Created by Sergei on 4/10/2024.
//

#pragma once

#include <vector>
#include <cmath>
#include <memory>

struct flat_embed {
public:
    class iterator {
    public:
        using value_type = std::vector<float_t>;

        iterator(std::shared_ptr<std::vector<float_t>> data, ptrdiff_t row_size, ptrdiff_t last_row_offset,
                 ptrdiff_t offset);

        iterator(const iterator &other) = default;

        auto operator++() -> iterator &;

        [[nodiscard]] auto operator==(const iterator &other) const -> bool;

        [[nodiscard]] auto operator!=(const iterator &other) const -> bool;

        [[nodiscard]] auto operator*() -> value_type;

    private:
        const ptrdiff_t row_size_, last_row_offset_;
        ptrdiff_t offset_;
        std::shared_ptr<std::vector<float_t>> data_;
    };

    [[nodiscard]] auto begin() const -> iterator;

    [[nodiscard]] auto end() const -> iterator;

    [[nodiscard]] auto data() const -> float_t *;

    [[nodiscard]] auto rows() const -> size_t;

    [[nodiscard]] auto row_size() const -> size_t;

    flat_embed(size_t rows, size_t row_size);

private:
    [[nodiscard]] inline auto last_row_offset() const -> size_t;

    size_t rows_, row_size_;
    std::shared_ptr<std::vector<float_t>> data_;
};
