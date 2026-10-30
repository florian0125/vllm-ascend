/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
*/

#ifndef ACC_OFFLOAD_SPARSE_COPY_H
#define ACC_OFFLOAD_SPARSE_COPY_H

#include "kernel_operator.h"

constexpr uint32_t COPY_BYTES = 32 * 1024;
constexpr uint32_t SPARSE_COPY_WAVES = 8;
constexpr uint32_t NATIVE_TILE_ROWS = 16;
constexpr uint32_t NATIVE_FIELDS = 6;
constexpr uint32_t RECIPE_WIDTH = 10;

// Keep two AIV cores and 64 KiB tiles; release them between short copy waves.
class OffloadSparseCopyKernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR lens, GM_ADDR size, uint32_t wave) {
        inputs_ = reinterpret_cast<__gm__ uint64_t*>(inputs);
        outputs_ = reinterpret_cast<__gm__ uint64_t*>(outputs);
        lens_ = reinterpret_cast<__gm__ uint32_t*>(lens);
        count_ = *reinterpret_cast<__gm__ uint32_t*>(size);
        wave_ = wave;
        pipe_.InitBuffer(queue_, 2, SPARSE_TILE_BYTES);
    }

    __aicore__ inline void Process() {
        for (uint32_t field = 0; field < count_; ++field) {
            const uint64_t length = lens_[field];
            const uint64_t tile_count = (length + SPARSE_TILE_BYTES - 1) / SPARSE_TILE_BYTES;
            const uint64_t begin_tile = tile_count * wave_ / SPARSE_COPY_WAVES;
            const uint64_t end_tile = tile_count * (wave_ + 1) / SPARSE_COPY_WAVES;
            for (uint64_t tile = begin_tile + AscendC::GetBlockIdx();
                 tile < end_tile; tile += AscendC::GetBlockNum()) {
                const uint64_t offset = tile * static_cast<uint64_t>(SPARSE_TILE_BYTES);
                uint32_t bytes = static_cast<uint32_t>(length - offset);
                bytes = bytes > SPARSE_TILE_BYTES ? SPARSE_TILE_BYTES : bytes;
                AscendC::GlobalTensor<uint8_t> source, destination;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(inputs_[field]) + offset, bytes);
                destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(outputs_[field]) + offset, bytes);
                auto local = queue_.AllocTensor<uint8_t>();
                AscendC::DataCopyExtParams params(1, bytes, 0, 0, 0);
                AscendC::DataCopyPadExtParams<uint8_t> pad(false, 0, 0, 0);
                AscendC::DataCopyPad(local, source, params, pad);
                queue_.EnQue(local);
                local = queue_.DeQue<uint8_t>();
                AscendC::DataCopyPad(destination, local, params);
                queue_.FreeTensor(local);
            }
        }
    }
private:
    static constexpr uint32_t SPARSE_TILE_BYTES = 64 * 1024;
    AscendC::TPipe pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2> queue_;
    __gm__ uint64_t *inputs_, *outputs_;
    __gm__ uint32_t *lens_;
    uint32_t count_;
    uint32_t wave_;
};

// Recipe: source, destination, source expert stride, destination slot stride,
// source rows, source row bytes, destination rows, destination row offset,
// column chunk width and precomputed UB gather-offset pointer.
// Whole row tiles are fetched contiguously from host DRAM before UB reordering.
class NativeExpertCopyKernel {
public:
    __aicore__ inline void Init(GM_ADDR recipes, GM_ADDR expert, GM_ADDR slot, GM_ADDR active)
    {
        recipes_ = reinterpret_cast<__gm__ uint64_t *>(recipes);
        expert_ = *reinterpret_cast<__gm__ int64_t *>(expert);
        slot_ = *reinterpret_cast<__gm__ int64_t *>(slot);
        active_ = *reinterpret_cast<__gm__ uint8_t *>(active);
    }
    __aicore__ inline void Process()
    {
        if (!active_) return;
        pipe_.InitBuffer(input_, 1, COPY_BYTES);
        pipe_.InitBuffer(output_, 1, COPY_BYTES);
        pipe_.InitBuffer(indices_, 1, 2 * COPY_BYTES);
        pipe_.InitBuffer(plain_, 1, COPY_BYTES);
        uint64_t firstJob = 0;
        for (uint32_t field = 0; field < NATIVE_FIELDS; ++field) {
            auto recipe = recipes_ + field * RECIPE_WIDTH;
            uint32_t rowBytes = recipe[5], width = recipe[8];
            uint32_t jobs = recipe[4] / NATIVE_TILE_ROWS;
            AscendC::LocalTensor<uint32_t> offsets;
            if (width != rowBytes) {
                uint32_t itemBytes = width == 2 ? 2 : 4;
                uint32_t count = NATIVE_TILE_ROWS * rowBytes / itemBytes;
                offsets = indices_.AllocTensor<uint32_t>();
                AscendC::GlobalTensor<uint32_t> indexGM;
                indexGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(recipe[9]), count);
                AscendC::DataCopy(offsets, indexGM, count);
                indices_.EnQue(offsets);
                offsets = indices_.DeQue<uint32_t>();
            }
            uint32_t first = (AscendC::GetBlockIdx() + AscendC::GetBlockNum()
                - firstJob % AscendC::GetBlockNum()) % AscendC::GetBlockNum();
            for (uint32_t job = first; job < jobs; job += AscendC::GetBlockNum()) {
                uint32_t row = job * NATIVE_TILE_ROWS;
                AscendC::GlobalTensor<uint8_t> source, destination;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(recipe[0])
                    + expert_ * recipe[2] + row * rowBytes);
                destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(recipe[1])
                    + slot_ * recipe[3] + (row + recipe[7]) * width);
                if (width == rowBytes) {
                    auto local = plain_.AllocTensor<uint8_t>();
                    AscendC::DataCopyExtParams params(1, NATIVE_TILE_ROWS * rowBytes, 0, 0, 0);
                    AscendC::DataCopyPadExtParams<uint8_t> padding(false, 0, 0, 0);
                    AscendC::DataCopyPad(local, source, params, padding);
                    plain_.EnQue(local);
                    local = plain_.DeQue<uint8_t>();
                    AscendC::DataCopyPad(destination, local, params);
                    plain_.FreeTensor(local);
                    continue;
                }
                auto input = input_.AllocTensor<uint8_t>();
                AscendC::DataCopyExtParams load(1, NATIVE_TILE_ROWS * rowBytes, 0, 0, 0);
                AscendC::DataCopyPadExtParams<uint8_t> pad(false, 0, 0, 0);
                AscendC::DataCopyPad(input, source, load, pad);
                input_.EnQue(input);
                input = input_.DeQue<uint8_t>();
                auto output = output_.AllocTensor<uint8_t>();
                if (width == 2) {
                    AscendC::Gather(output.template ReinterpretCast<uint16_t>(),
                        input.template ReinterpretCast<uint16_t>(), offsets, 0,
                        static_cast<uint32_t>(NATIVE_TILE_ROWS * rowBytes / 2));
                } else {
                    AscendC::Gather(output.template ReinterpretCast<uint32_t>(),
                        input.template ReinterpretCast<uint32_t>(), offsets, 0,
                        static_cast<uint32_t>(NATIVE_TILE_ROWS * rowBytes / 4));
                }
                output_.EnQue(output);
                output = output_.DeQue<uint8_t>();
                AscendC::DataCopyExtParams store(rowBytes / width, NATIVE_TILE_ROWS * width,
                    0, (recipe[6] - NATIVE_TILE_ROWS) * width, 0);
                AscendC::DataCopyPad(destination, output, store);
                output_.FreeTensor(output);
                input_.FreeTensor(input);
            }
            if (width != rowBytes) indices_.FreeTensor(offsets);
            firstJob += jobs;
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> input_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> output_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> indices_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> plain_;
    __gm__ uint64_t *recipes_;
    int64_t expert_, slot_;
    uint8_t active_;
};

#endif
