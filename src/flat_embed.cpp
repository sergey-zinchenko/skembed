//
// Created by Sergei on 4/10/2024.
//


#include "flat_embed.h"


abstract_flat_embed::iterator flat_embed::begin() const {
    return {data_, static_cast<ptrdiff_t>(row_size()),
            static_cast<ptrdiff_t>(last_row_offset()), 0};
}


size_t flat_embed::last_row_offset() const { return (rows_ - 1) * row_size_; }


size_t flat_embed::row_size() const { return row_size_; }


float_t *flat_embed::data() const { return data_.get(); }


abstract_flat_embed::iterator flat_embed::end() const {
    return {data_,
            static_cast<ptrdiff_t>(row_size()),
            static_cast<ptrdiff_t>(last_row_offset()),
            static_cast<ptrdiff_t>(last_row_offset())};
}


size_t flat_embed::rows() const { return rows_; }

flat_embed::flat_embed(size_t rows, size_t row_size) :
        rows_(rows), row_size_(row_size) {
}
