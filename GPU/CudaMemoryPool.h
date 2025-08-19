// ===== CudaMemoryPool.h =====
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

class CudaMemoryPool {
public:
    CudaMemoryPool() = default;
    ~CudaMemoryPool() { releaseAll(); }

    // Simple bump allocator for device memory blocks
    bool reserve(size_t bytes) {
        releaseAll();
        capacity_ = bytes;
        used_ = 0;
        if (cudaMalloc(&buffer_, capacity_) != cudaSuccess) {
            buffer_ = nullptr;
            capacity_ = used_ = 0;
            return false;
        }
        return true;
    }

    void* allocate(size_t bytes, size_t alignment = 256) {
        size_t current = reinterpret_cast<size_t>(buffer_) + used_;
        size_t aligned = (current + alignment - 1) & ~(alignment - 1);
        size_t offset = aligned - reinterpret_cast<size_t>(buffer_);
        if (offset + bytes > capacity_) return nullptr;
        used_ = offset + bytes;
        return static_cast<char*>(buffer_) + offset;
    }

    void reset() { used_ = 0; }

    void releaseAll() {
        if (buffer_) cudaFree(buffer_);
        buffer_ = nullptr;
        used_ = 0;
        capacity_ = 0;
    }

    size_t capacity() const { return capacity_; }
    size_t used() const { return used_; }

private:
    void* buffer_ = nullptr;
    size_t capacity_ = 0;
    size_t used_ = 0;
};
