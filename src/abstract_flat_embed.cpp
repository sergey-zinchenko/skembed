//
// Created by Sergei on 4/10/2024.
//

#include "abstract/abstract_flat_embed.h"


abstract_flat_embed::iterator::iterator(std::shared_ptr<float_t[]> data, ptrdiff_t row_size, ptrdiff_t last_row_offset,
                                        ptrdiff_t offset) :
        data_(std::move(data)),
        row_size_(row_size),
        last_row_offset_(last_row_offset),
        offset_(offset) {}

abstract_flat_embed::iterator &abstract_flat_embed::iterator::operator++() {
    auto next_row_offset = offset_ + row_size_;
    if (next_row_offset <= last_row_offset_) {
        offset_ = next_row_offset;
    }
    return *this;
}

bool abstract_flat_embed::iterator::operator==(const abstract_flat_embed::iterator &other) const {
    return data_ == other.data_ && offset_ == other.offset_;
}

bool abstract_flat_embed::iterator::operator!=(const abstract_flat_embed::iterator &other) const {
    return !(*this == other);
}

abstract_flat_embed::iterator::value_type abstract_flat_embed::iterator::operator*() {
    return std::make_shared<std::vector<float_t>>(row_size_, data_[offset_ + row_size_]);
}

