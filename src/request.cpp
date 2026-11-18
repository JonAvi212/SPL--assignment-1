#include "Request.h"
#include "GPU.h"
#include "Model.h"
#include <string>   
#include <vector>
#include <algorithm>

// =================================================================
//                       Base Class: Request
// =================================================================

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt):
    id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)),memoryBlocks(), reservedBlocks(0) {
}
Request::~Request() = default;

int Request:: getId() const{
    return id;
}
const Model& Request:: getModel() const{
    return model;
}
int Request:: getAdmissionCycle() const{
    return admissionCycle;
}
void Request::deallocateMemory(GPU& gpu){
    gpu.unreserve(getNeedToUnreserve(), *this);
    for (std::size_t i = 0; i < memoryBlocks.size(); ++i) {
        gpu.release(memoryBlocks[i]);
    }
    memoryBlocks.clear();
}




// =================================================================
//                 Derived Class: GenerativeRequest
// =================================================================

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
std::size_t maxOutputTokens):
    Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput),scriptedOutputTokens(model.tokenizer(scriptedOutput)),generatedTokens(), maxOutputTokens(maxOutputTokens), processedTokens(0),allocatedBlocks(0) {
    }
std::size_t GenerativeRequest::getNeedToUnreserve() const{
    return reservedBlocks - allocatedBlocks;
}  
std::string GenerativeRequest::getOutput() const{
    std::string result = "";
    for (std::size_t i = 0; i < generatedTokens.size(); ++i) {
        if (i > 0) {
            result += " ";
        }
        result += generatedTokens[i];
    }
    return result;
}
std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const{ // the real amount of cells memory the request need (considering the cached prefix)
    return model.getEmbeddingSize() * (tokens.size() + maxOutputTokens - gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens).matchedTokens);
}
std::size_t GenerativeRequest::getMaxNeededMemory() const{  // the maximum memory cells GPU has to allocate to this request (without the cached prefix)
    return (model.getEmbeddingSize())*(tokens.size() + maxOutputTokens); 
}
std::vector<std::string> GenerativeRequest::getCacheableTokens() const{
    return tokens;
}
void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{
    gpu.addMemoryBlock(*this, slot);
}
Request* GenerativeRequest::clone(const Model& model) const {
    GenerativeRequest* copy = new GenerativeRequest(id, model, admissionCycle, "", scriptedOutput, maxOutputTokens);
    copy->tokens = tokens;
    copy->reservedBlocks = reservedBlocks;
    copy->memoryBlocks = memoryBlocks;
    copy->generatedTokens = generatedTokens;
    copy->processedTokens = processedTokens;
    copy->allocatedBlocks = allocatedBlocks;
    return copy;
}

void GenerativeRequest::act(GPU& gpu){
    if (finished()) {
        return;
    }

    std::size_t embeddingSize = model.getEmbeddingSize();
    std::size_t tokensPerBlock = gpu.getBlockSize() / embeddingSize;
    std::size_t tokenIndexInBlock = processedTokens % tokensPerBlock;

    // Allocate a new KV block only when starting a new block
    if (tokenIndexInBlock == 0) {
        memoryBlocks.push_back(gpu.allocateBlock(*this));
        allocatedBlocks++;
    }

    // Build the full prefix up to and including the current token
    std::vector<std::string> prefix;
    if (processedTokens < tokens.size()) {
        // Prefill phase: processing a prompt token
        prefix.assign(tokens.begin(), tokens.begin() + processedTokens + 1);
    } else {
        // Decoding phase: generating the next scripted output token
        std::size_t outputIdx = processedTokens - tokens.size();
        generatedTokens.push_back(scriptedOutputTokens[outputIdx]);

        prefix = tokens;
        prefix.insert(prefix.end(), generatedTokens.begin(), generatedTokens.end());
    }

    // Write the KV vector of the current token into the current block at its offset
    std::size_t currentSlot = memoryBlocks.back();
    std::size_t offset = tokenIndexInBlock * embeddingSize;
    model.generateTokenEmbedding(prefix, gpu, currentSlot, offset);

    processedTokens++;

    // Only full blocks get a prefix key for caching
    if (processedTokens % tokensPerBlock == 0) {
        gpu.assignPrefixToBlock(currentSlot, prefix, model.getId());
    }
}
bool GenerativeRequest::finished() const{
    std::size_t totalTokens = tokens.size() + std::min(maxOutputTokens, scriptedOutputTokens.size());
    return processedTokens == totalTokens;
}

void GenerativeRequest::reserveMemory(GPU& gpu){
    PrefixMatch cachedBlocks = gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), tokens);

    // Acquire the cached prefix blocks and mark their tokens as already processed
    for (std::size_t i = 0; i < cachedBlocks.blockIds.size(); ++i) {
        gpu.acquire(cachedBlocks.blockIds[i]);
        memoryBlocks.push_back(cachedBlocks.blockIds[i]);
    }
    processedTokens = cachedBlocks.matchedTokens;

    // Reserve the blocks still needed for remaining prompt tokens + maxOutputTokens
    std::size_t numBlocks = gpu.getBlocksForMemory(model.getEmbeddingSize() * (tokens.size() - cachedBlocks.matchedTokens + maxOutputTokens));
    gpu.reserve(numBlocks, *this);
    reservedBlocks = numBlocks;
}

 

// =================================================================
//                 Derived Class: EmbeddingRequest
// =================================================================

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt):
    Request(id, model, admissionCycle, prompt), finalOutput(""),processedTokensCount(0)  { 
}
std::size_t EmbeddingRequest ::getNeedToUnreserve() const{
    return reservedBlocks - memoryBlocks.size();
}
std::string EmbeddingRequest::getOutput() const{
    //the desired output: The output of an embedding request is the mean of its token vectors which computed in the act func.
    return finalOutput;
}
std::size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const{
    return 2*model.getEmbeddingSize();
}
std::size_t EmbeddingRequest::getMaxNeededMemory() const{ //one vector to the current prompt token in embedding size, another one vector to the running sum
    return 2*model.getEmbeddingSize();
}
std::vector<std::string> EmbeddingRequest::getCacheableTokens() const{
    return std::vector<std::string>();
}
void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{
    gpu.addMemoryBlock(*this, slot);
}
Request* EmbeddingRequest::clone(const Model& model) const{
    EmbeddingRequest* copy = new EmbeddingRequest(id, model, admissionCycle, "");
    copy->tokens = tokens;
    copy->reservedBlocks = reservedBlocks;
    copy->memoryBlocks = memoryBlocks;
    copy->processedTokensCount = processedTokensCount;
    copy->finalOutput = finalOutput;
    return copy;
}
void EmbeddingRequest::act(GPU& gpu){
    if (finished()) {
        return;
    }
    std::size_t embeddingSize = model.getEmbeddingSize();
    std::size_t firstBlockSlot = memoryBlocks[0];
    // If 2 vectors fit in 1 block: current token starts at offset embeddingSize in memoryBlocks[0].
    // If 1 vector per block (2 blocks): current token starts at offset 0 in memoryBlocks[1].
    std::size_t currTokenSlot = (memoryBlocks.size() == 1) ? memoryBlocks[0] : memoryBlocks[1];
    std::size_t currTokenOffset = (memoryBlocks.size() == 1) ? embeddingSize : 0;

    // Generate embedding for the current token only
    std::vector<std::string> currentToken(1, tokens[processedTokensCount]);
    model.generateTokenEmbedding(currentToken, gpu, currTokenSlot, currTokenOffset);

    // Add current token vector element-wise to the running sum at offset 0 of memoryBlocks[0]
    for (std::size_t i = 0; i < embeddingSize; ++i) {
        int runningSum = gpu.readFromBlock(firstBlockSlot, i);
        int tokenVal = gpu.readFromBlock(currTokenSlot, currTokenOffset + i);
        gpu.writeToBlock(firstBlockSlot, i, runningSum + tokenVal);
    }

    processedTokensCount++;

    // If this was the last token, compute the mean and build finalOutput
    if (processedTokensCount == tokens.size()) {
        int numTokens = static_cast<int>(tokens.size());
        if (numTokens == 0) {
            numTokens = 1;
        }

        finalOutput = "[";
        for (std::size_t i = 0; i < embeddingSize; ++i) {
            int currentSum = gpu.readFromBlock(firstBlockSlot, i);
            int averagedValue = currentSum / numTokens;

            if (i > 0) {
                finalOutput += ", ";
            }
            finalOutput += std::to_string(averagedValue);
        }
        finalOutput += "]";
    }
}

bool EmbeddingRequest::finished() const {
    return processedTokensCount == tokens.size();
}
void EmbeddingRequest::reserveMemory(GPU& gpu){
    std::size_t numBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    gpu.reserve(numBlocks, *this);
    reservedBlocks = numBlocks;
    for (std::size_t i = 0; i < numBlocks; ++i) {
        memoryBlocks.push_back(gpu.allocateBlock(*this));
    }
}

