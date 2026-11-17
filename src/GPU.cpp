#include "GPU.h"
#include <string>
#include <cstddef>
#include <vector>

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr){

    }
std::size_t GPU::availableSlots() const{
    return 0;
} 
std::size_t GPU::getBlocksForMemory(std::size_t memory) const{
    return 0;
} 
void GPU::reserve(std::size_t n, const Request& request){
    return;
}
void GPU::unreserve(std::size_t n, const Request& request){
    return;
}
std::size_t GPU::allocateBlock(const Request& request){
    return 0;
}
void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot){
    return;
}
void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot){
    return;
}
void GPU::acquire(std::size_t slot){
    return;
}
void GPU::release(std::size_t slot){
    return;
}
void GPU::copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks){
    return;
}
void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId){
    return;
}
int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const{
    return 0;
}
void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value){
    return;
}
PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                    const std::vector<std::string>& promptTokens) const{
    return PrefixMatch();
}
