//
// Created by Sergei on 4/10/2024.
//


#include "flat_embed.h"


flat_embed::iterator::iterator(std::shared_ptr<std::vector<float_t>> data, const ptrdiff_t row_size,
                               const ptrdiff_t last_row_offset,
                               const ptrdiff_t offset) :
        data_(std::move(data)),
        row_size_(row_size),
        last_row_offset_(last_row_offset),
        offset_(offset) {}

flat_embed::iterator &flat_embed::iterator::operator++() {
    auto next_row_offset = offset_ + row_size_;
    if (next_row_offset <= last_row_offset_) {
        offset_ = next_row_offset;
    }
    return *this;
}

bool flat_embed::iterator::operator==(const flat_embed::iterator &other) const {
    return data_ == other.data_ && offset_ == other.offset_;
}

bool flat_embed::iterator::operator!=(const flat_embed::iterator &other) const {
    return !(*this == other);
}

flat_embed::iterator::value_type flat_embed::iterator::operator*() {
    return std::vector<float_t>{data_->begin() + offset_, data_->begin() + offset_ + row_size_};
}

flat_embed::iterator flat_embed::begin() {
    return {data_, static_cast<ptrdiff_t>(row_size()),
            static_cast<ptrdiff_t>(last_row_offset()), 0};
}

size_t flat_embed::last_row_offset() const { return (rows_ - 1) * row_size_; }


size_t flat_embed::row_size() const { return row_size_; }


float_t *flat_embed::data() const { return data_->data(); }


flat_embed::iterator flat_embed::end() {
    return {data_,
            static_cast<ptrdiff_t>(row_size()),
            static_cast<ptrdiff_t>(last_row_offset()),
            static_cast<ptrdiff_t>(last_row_offset())};
}


size_t flat_embed::rows() const { return rows_; }

flat_embed::flat_embed(size_t rows, size_t row_size) :
        rows_(rows),
        row_size_(row_size),
        data_(std::make_shared<std::vector<float_t>>(rows * row_size)) {
}
