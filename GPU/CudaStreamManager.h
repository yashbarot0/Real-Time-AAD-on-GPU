// ===== CudaStreamManager.h =====
#pragma once
#include <cuda_runtime.h>
#include <vector>

class CudaStreamManager {
public:
    CudaStreamManager() = default;
    ~CudaStreamManager() { destroyAll(); }

    bool create(size_t count = 1, unsigned int flags = cudaStreamDefault) {
        destroyAll();
        streams_.resize(count, nullptr);
        for (size_t i = 0; i < count; ++i) {
            if (cudaStreamCreateWithFlags(&streams_[i], flags) != cudaSuccess) {
                destroyAll();
                return false;
            }
        }
        return true;
    }

    cudaStream_t get(size_t index = 0) const {
        return (index < streams_.size()) ? streams_[index] : nullptr;
    }

    size_t size() const { return streams_.size(); }

    void synchronize(size_t index = 0) const {
        if (index < streams_.size() && streams_[index]) cudaStreamSynchronize(streams_[index]);
    }

    void destroyAll() {
        for (auto &s : streams_) if (s) cudaStreamDestroy(s);
        streams_.clear();
    }

private:
    std::vector<cudaStream_t> streams_{};
};
