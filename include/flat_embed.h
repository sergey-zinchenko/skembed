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

        iterator &operator++();

        [[nodiscard]] bool operator==(const iterator &other) const;

        [[nodiscard]] bool operator!=(const iterator &other) const;

        [[nodiscard]] value_type operator*();

    private:
        const ptrdiff_t row_size_, last_row_offset_;
        ptrdiff_t offset_;
        std::shared_ptr<std::vector<float_t>> data_;
    };

    [[nodiscard]] iterator begin();

    [[nodiscard]] iterator end();

    [[nodiscard]] float_t *data() const;

    [[nodiscard]] size_t rows() const;

    [[nodiscard]] size_t row_size() const;

    flat_embed(size_t rows, size_t row_size);

private:
    [[nodiscard]] inline size_t last_row_offset() const;

    size_t rows_, row_size_;
    std::shared_ptr<std::vector<float_t>> data_;
};
