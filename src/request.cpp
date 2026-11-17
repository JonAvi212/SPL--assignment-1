#include "Request.h"
#include "GPU.h"
#include "Model.h"
#include <string>   
#include <vector>

// =================================================================
//                       Base Class: Request
// =================================================================

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt):
    id(id), model(model), admissionCycle(admissionCycle), tokens(tokenize(prompt)), reservedBlocks(0) {
    }
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
    Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens) {
    }
std::size_t GenerativeRequest::getNeedToUnreserve() const{
    return reservedBlocks - memoryBlocks.size();
}  
std::string GenerativeRequest::getOutput() const{
    //the desired output: The output of a generative request is its generated tokens joined by single spaces (the scriptedoutput, cut to maxOutputTokens tokens).
    std::vector<std::string> tokens = model.tokenizer(scriptedOutput);
    std::size_t limit = std::min(tokens.size(), maxOutputTokens);
    std::string result = "";
for (std::size_t i = 0; i < limit; ++i) {
    result += tokens[i];
    if (i + 1 < limit) {
        result += " ";
    }
}
return result;
}
std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const{
    return 0;
}
std::size_t GenerativeRequest::getMaxNeededMemory() const{  // the maximum memory cells GPU has to allocate to this request.
    return model.getEmbeddingSize() * (tokens.size() + maxOutputTokens); 
}
std::vector<std::string> GenerativeRequest::getCacheableTokens() const{
    return tokens;
}
void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{
    return;
}
Request* GenerativeRequest::clone(const Model& model) const{
    return new GenerativeRequest(id, model, admissionCycle, joinTokens(tokens), scriptedOutput, maxOutputTokens);
}
void GenerativeRequest::act(GPU& gpu){
    return;
}
bool GenerativeRequest::finished() const{
    return true;
}
void GenerativeRequest::reserveMemory(GPU& gpu){
    return;
}

 

// =================================================================
//                 Derived Class: EmbeddingRequest
// =================================================================

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt):
    Request(id, model, admissionCycle, prompt), finalOutput("")  { 
}
std::size_t EmbeddingRequest ::getNeedToUnreserve() const{
    return reservedBlocks - memoryBlocks.size();
}
std::string EmbeddingRequest::getOutput() const override{
    //the desired output: The output of an embedding request is the mean of its token vectors which computed in the act func.
    return finalOutput;
}
std::size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const{
    return 0;
}
std::size_t EmbeddingRequest::getMaxNeededMemory() const{ //one vector to the current prompt token in embedding size, one vector to the running sum
    return 2*model.getEmbeddingSize();
}
std::vector<std::string> EmbeddingRequest::getCacheableTokens() const{
    return std::vector<std::string>();
}
void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{
    return;
}
Request* EmbeddingRequest::clone(const Model& model) const{
    return new EmbeddingRequest(id, model, admissionCycle, joinTokens(tokens));
}
void EmbeddingRequest::act(GPU& gpu){
    // ... 
    // [Your existing logic here: generating the current token's embedding 
    // and adding it to the running sum stored at offset 0 of memoryBlocks[0]]
    // ...

    // Increment your internal counter (assuming you have one)
    processedTokensCount++;

    // Check if this was the last token. If so, build and save the final output string.
    if (processedTokensCount == tokens.size()) {
        size_t numTokens = tokens.size();
        if (numTokens == 0) numTokens = 1; // Edge case protection

        finalOutput = "[";
        bool isFirstElement = true;

        // The slot holding our running sum
        size_t firstBlockSlot = memoryBlocks[0]; 

        for (size_t i = 0; i < embeddingSize; ++i) {
            // Read the accumulated sum for coordinate 'i' from the GPU.
            // We assume the running sum is logically placed starting at offset 0.
            int currentSum = gpu.readFromBlock(firstBlockSlot, i);
            
            // Calculate the average using integer division as required
            int averagedValue = currentSum / numTokens;

            if (!isFirstElement) {
                finalOutput += ", ";
            }
            finalOutput += std::to_string(averagedValue);
            isFirstElement = false;
        }

        finalOutput += "]";
        
        // Note: The request has now generated its output. 
        // In the next server cycle, finished() should return true.
    }
}

bool EmbeddingRequest::finished() const{
    return true;
}
void EmbeddingRequest::reserveMemory(GPU& gpu){
    return;
}

