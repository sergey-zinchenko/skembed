//
// Created by Sergei on 4/10/2024.
//

#pragma once

#include <iterator>
#include <memory>
#include <vector>

class abstract_flat_embed {
public:
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::shared_ptr<std::vector<float_t>>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type *;
        using reference = value_type &;

        iterator(std::shared_ptr<float_t[]> data, ptrdiff_t row_size, ptrdiff_t last_row_offset,
                 ptrdiff_t offset);

        iterator(const iterator &other) = default;

        iterator &operator++();

        [[nodiscard]] bool operator==(const iterator &other) const;

        [[nodiscard]] bool operator!=(const iterator &other) const;

        [[nodiscard]] value_type operator*();

    private:
        const ptrdiff_t row_size_, last_row_offset_;
        ptrdiff_t offset_;
        const std::shared_ptr<float_t[]> data_;
    };

    [[nodiscard]] virtual iterator begin() const = 0;

    [[nodiscard]] virtual iterator end() const = 0;

    [[nodiscard]] virtual float_t *data() const = 0;

    [[nodiscard]] virtual size_t rows() const = 0;

    [[nodiscard]] virtual size_t row_size() const = 0;
};