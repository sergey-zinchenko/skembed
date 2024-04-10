//
// Created by Sergei on 4/10/2024.
//

#pragma once

#include "abstract/abstract_flat_embed.h"

template<size_t row_size_, size_t rows_>
class flat_embed : public abstract_flat_embed {
public:
    [[nodiscard]] iterator begin() const override;

    [[nodiscard]] iterator end() const override;

    [[nodiscard]] float_t *data() const override;

    [[nodiscard]] size_t rows() const override;

    [[nodiscard]] size_t row_size() const override;

private:
    static constexpr size_t last_row_offset();

    std::shared_ptr<float_t[]> data_ = std::make_shared<float_t[]>(row_size() * rows_);
};
