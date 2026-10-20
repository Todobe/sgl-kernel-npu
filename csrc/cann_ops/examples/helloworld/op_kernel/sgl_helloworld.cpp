#include "kernel_operator.h"
#include "sgl_helloworld_tiling.h"

// Fixed-size UB buffers keep memory usage independent of the input shape.
class SglHelloworldKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t size) {
        length_ = size / AscendC::GetBlockNum();
        const uint32_t offset = length_ * AscendC::GetBlockIdx();
        x_.SetGlobalBuffer((__gm__ half*)x + offset, length_);
        y_.SetGlobalBuffer((__gm__ half*)y + offset, length_);
        z_.SetGlobalBuffer((__gm__ half*)z + offset, length_);
        pipe_.InitBuffer(qx_, 1, kTile * sizeof(half));
        pipe_.InitBuffer(qy_, 1, kTile * sizeof(half));
        pipe_.InitBuffer(qz_, 1, kTile * sizeof(half));
    }
    __aicore__ inline void Process() {
        for (uint32_t offset = 0; offset < length_; offset += kTile) {
            const uint32_t count = length_ - offset < kTile ? length_ - offset : kTile;
            auto x = qx_.AllocTensor<half>();
            auto y = qy_.AllocTensor<half>();
            AscendC::DataCopy(x, x_[offset], count);
            AscendC::DataCopy(y, y_[offset], count);
            qx_.EnQue(x);
            qy_.EnQue(y);
            x = qx_.DeQue<half>();
            y = qy_.DeQue<half>();
            auto z = qz_.AllocTensor<half>();
            AscendC::Add(z, x, y, count);
            qz_.EnQue(z);
            qx_.FreeTensor(x);
            qy_.FreeTensor(y);
            z = qz_.DeQue<half>();
            AscendC::DataCopy(z_[offset], z, count);
            qz_.FreeTensor(z);
        }
    }
private:
    static constexpr uint32_t kTile = 256;
    uint32_t length_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qx_, qy_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qz_;
    AscendC::GlobalTensor<half> x_, y_, z_;
};

extern "C" __global__ __aicore__ void sgl_helloworld(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SglHelloworldTilingData);
    GET_TILING_DATA(data, tiling);
    SglHelloworldKernel kernel;
    kernel.Init(x, y, z, data.size);
    kernel.Process();
}
