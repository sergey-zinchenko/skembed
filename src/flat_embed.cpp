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

auto flat_embed::iterator::operator++() -> flat_embed::iterator & {
  auto next_row_offset = offset_ + row_size_;
  if (next_row_offset <= last_row_offset_) {
	  offset_ = next_row_offset;
	} else {
	  offset_ = last_row_offset_;
	}
  return *this;
}

auto flat_embed::iterator::operator==(const flat_embed::iterator &other) const -> bool {
    return data_ == other.data_ && offset_ == other.offset_;
}

auto flat_embed::iterator::operator!=(const flat_embed::iterator &other) const -> bool {
    return !(*this == other);
}

auto flat_embed::iterator::operator*() -> flat_embed::iterator::value_type {
    return std::vector<float_t>{data_->begin() + offset_, data_->begin() + offset_ + row_size_};
}

auto flat_embed::begin() const -> flat_embed::iterator {
    return {data_, static_cast<ptrdiff_t>(row_size()),
            static_cast<ptrdiff_t>(last_row_offset()), 0};
}

auto flat_embed::last_row_offset() const -> size_t { return (rows_ - 1) * row_size_; }


auto flat_embed::row_size() const -> size_t { return row_size_; }


auto flat_embed::data() const -> float_t * { return data_->data(); }


auto flat_embed::end() const -> flat_embed::iterator {
    return {data_,
            static_cast<ptrdiff_t>(row_size()),
            static_cast<ptrdiff_t>(last_row_offset()),
            static_cast<ptrdiff_t>(last_row_offset())};
}


auto flat_embed::rows() const -> size_t { return rows_; }

flat_embed::flat_embed(size_t rows, size_t row_size) :
        rows_(rows),
        row_size_(row_size),
        data_(std::make_shared<std::vector<float_t>>(rows * row_size)) {
}
