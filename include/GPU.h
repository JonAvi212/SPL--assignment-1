#ifndef GPU_H_
#define GPU_H_

#include <string>
#include <cstddef>
#include <vector>

struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
};

class MemoryBlock;

class GPU{
public:
    GPU(std::size_t blocksNum, std::size_t blockSize); // Constructor to initialize the GPU with a specified number of blocks and block size.
    std::size_t availableSlots() const; // Returns the number of available memory slots in the GPU.
    std::size_t getBlocksForMemory(std::size_t memory) const; // Returns the number of blocks required for a given amount of memory.
    void reserve(std::size_t n, const Request& request); // Reserves a specified number of memory blocks for a request.
    void unreserve(std::size_t n, const Request& request); // Unreserves a specified number of memory blocks for a request.
    std::size_t allocateBlock(const Request& request); // Allocates a memory block for a request.
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot); // Adds a memory block for a generative request.
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot); // Adds a memory block for an embedding request.
    void acquire(std::size_t slot); // Acquires a memory slot.
    void release(std::size_t slot); // Releases a memory slot.
    void copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks); // Copies memory blocks from another GPU.
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId); // Assigns a prefix to a memory block for caching purposes.
    int readFromBlock(std::size_t blockId, std::size_t offset) const; // Reads a value from a specified memory block at a given offset.
    void writeToBlock(std::size_t blockId, std::size_t offset, int value); // Writes a value to a specified memory block at a given offset.
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize, // Finds the longest cached prefix for a given model and embedding size.
                                        const std::vector<std::string>& promptTokens) const;
private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock*> memory;
};

#endif