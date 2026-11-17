#include "request.h"
#include <string>   
#include <vector>

// =================================================================
//                       Base Class: Request
// =================================================================

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt):
    id(id), model(model), admissionCycle(admissionCycle), tokens(tokenize(prompt)), reservedBlocks(0) {

    }
void Request::deallocateMemory(GPU& gpu){
    return;
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

// =================================================================
//                 Derived Class: GenerativeRequest
// =================================================================

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
std::size_t maxOutputTokens):
    Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens) {
    }
std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const{
    return 0;
}
std::size_t GenerativeRequest::getMaxNeededMemory() const{
    return 0;
}
std::vector<std::string> GenerativeRequest::getCacheableTokens() const{
    return std::vector<std::string>();
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
std::string GenerativeRequest::getOutput() const{
    return scriptedOutput;
}
std::size_t GenerativeRequest::getNeedToUnreserve() const{
    return reservedBlocks;
}   

// =================================================================
//                 Derived Class: EmbeddingRequest
// =================================================================

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt):
    Request(id, model, admissionCycle, prompt) {
    
}
std::size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const{
    return 0;
}
std::size_t EmbeddingRequest::getMaxNeededMemory() const{
    return 0;
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
    return;
}
bool EmbeddingRequest::finished() const{
    return true;
}
void EmbeddingRequest::reserveMemory(GPU& gpu){
    return;
}
std::string EmbeddingRequest::getOutput() const override{
    return "";
}
std::size_t EmbeddingRequest ::getNeedToUnreserve() const{
    return reservedBlocks;
}   