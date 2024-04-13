//
// Created by Sergei on 4/10/2024.
//

#pragma once

#include "abstract/abstract_flat_embed.h"

class flat_embed : public abstract_flat_embed {
public:
    [[nodiscard]] iterator begin() const override;

    [[nodiscard]] iterator end() const override;

    [[nodiscard]] float_t *data() const override;

    [[nodiscard]] size_t rows() const override;

    [[nodiscard]] size_t row_size() const override;

    flat_embed(size_t rows, size_t row_size);

private:
    [[nodiscard]] inline size_t last_row_offset() const;

    size_t rows_, row_size_;
    std::shared_ptr<float_t[]> data_ = std::shared_ptr<float_t[]>(new float_t[row_size_ * rows_]);
};
