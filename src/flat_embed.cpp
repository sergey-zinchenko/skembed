//
// Created by Sergei on 4/10/2024.
//


#include "flat_embed.h"

template<size_t row_size_, size_t rows_>
abstract_flat_embed::iterator flat_embed<row_size_, rows_>::begin() const {
    return iterator(data_, row_size(), last_row_offset(), 0);
}

template<size_t row_size_, size_t rows_>
constexpr size_t flat_embed<row_size_, rows_>::last_row_offset() { return (rows_ - 1) * row_size_; }

template<size_t row_size_, size_t rows_>
size_t flat_embed<row_size_, rows_>::row_size() const { return row_size_; }

template<size_t row_size_, size_t rows_>
float_t *flat_embed<row_size_, rows_>::data() const { return data_.get(); }

template<size_t row_size_, size_t rows_>
abstract_flat_embed::iterator flat_embed<row_size_, rows_>::end() const {
    return iterator(data_, row_size(), last_row_offset(), last_row_offset());
}