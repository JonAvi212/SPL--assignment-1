#include "MemoryBlock.h"

// ================= MemoryBlock =================

MemoryBlock::MemoryBlock(size_t size)
    // כאן תבוא רשימת האתחול (Initializer List) לפי סדר השדות
    : size(size), refCount(0), data(new int[size]) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = 0;
        }
}


MemoryBlock::~MemoryBlock() {
    delete[] data;
}

MemoryBlock::MemoryBlock(const MemoryBlock& other)
    // כאן תבוא רשימת האתחול
    : size(other.size), refCount(0), data(new int[other.size]) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = other.data[i];
    }

}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this == &other) {
            return *this;
    }
    size = other.size;
    refCount = 0;
    delete[] data;
    data = new int[other.size];
    for (size_t i = 0; i < size; ++i) {
            data[i] = other.data[i];
    }
    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other)
    // כאן תבוא רשימת האתחול
    : size(other.size), refCount(other.refCount), data(other.data) {

    other.refCount = 0;
    other.size = 0;
    other.data = nullptr;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) {
    if (this == &other) {
        return *this;
    }
    delete[] data;
    size = other.size;
    refCount = other.refCount;
    data = other.data;
    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;
    return *this;
}

int MemoryBlock::readFromBlock(std::size_t offset) const {
    return data[offset];
}

void MemoryBlock::writeToBlock(std::size_t offset, int value) {
    if (size > offset) {
        data[offset] = value;
    }
}

void MemoryBlock::acquire() {
    refCount = refCount + 1;
}

IdleAction MemoryBlock::release() {
    refCount = refCount - 1;
    if (refCount == 0) {
        return onLastRelease ();
    }
    return IdleAction::KeepCached;
}

int MemoryBlock::getRefCount() const {
    return refCount;
}

// ================= KVBlock =================

KVBlock::KVBlock(size_t size)
    // כאן קוראים לבנאי של מחלקת הבסיס MemoryBlock ברשימת האתחול
    : MemoryBlock(size) {
}

IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}

MemoryBlock* KVBlock::clone() const {
    return new KVBlock(*this);
}

// ================= EmbeddingWorkspace =================

EmbeddingWorkspace::EmbeddingWorkspace(size_t size)
    // כאן קוראים לבנאי של מחלקת הבסיס MemoryBlock ברשימת האתחול
    : MemoryBlock(size) {
}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction:: Destroy;
}

MemoryBlock* EmbeddingWorkspace::clone() const {
    return new EmbeddingWorkspace(*this);
}