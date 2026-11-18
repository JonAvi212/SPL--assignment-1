#ifndef REQUEST_H_
#define REQUEST_H_

#include <string>
#include <vector>

class Model;
class GPU;

class Request{
protected:
    int id;
    const Model& model;
    int admissionCycle;
    std::vector<std::string> tokens;
    std::vector<std::size_t> memoryBlocks;
    std::size_t reservedBlocks;
    
public:
    Request(int id, const Model& model, int admissionCycle, std::string prompt); //splits the prompt into tokens
    virtual std::size_t getNeededMemory(const GPU& gpu) const=0; //
    virtual std::size_t getMaxNeededMemory() const=0; 
    virtual std::vector<std::string> getCacheableTokens() const=0; 
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const = 0; 
    virtual Request* clone(const Model& model) const = 0; 
    void deallocateMemory(GPU& gpu); 
    virtual void act(GPU& gpu)=0; 
    virtual bool finished() const=0; // Returns true if and only if the request has finished.
    virtual void reserveMemory(GPU& gpu) = 0; 
    virtual std::string getOutput() const = 0; 
    virtual std::size_t getNeedToUnreserve() const = 0; //the number of blocks the request reserved and has not used.
    int getId() const; 
    const Model& getModel() const;
    int getAdmissionCycle() const; 
};

class GenerativeRequest: public Request{
private:
    std::string scriptedOutput;
    std::vector<std::string> scriptedOutputTokens;
    std::size_t maxOutputTokens;
    std::vector<std::string> generatedTokens;
public:
    GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens); // the constructor of GenerativeRequest, which also receives the scripted output and the maximal number of tokens to generate.

    virtual std::size_t getNeededMemory(const GPU& gpu) const override; 
    virtual std::size_t getMaxNeededMemory() const override;
    virtual std::vector<std::string> getCacheableTokens() const override;
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const override;
    virtual Request* clone(const Model& model) const override;
    virtual void act(GPU& gpu) override;
    virtual bool finished() const override;
    virtual void reserveMemory(GPU& gpu) override;
    virtual std::string getOutput() const override;
    virtual std::size_t getNeedToUnreserve() const override;
};

class EmbeddingRequest: public Request{
private:
    std::string finalOutput;
public:
    EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt); //the constructor of EmbeddingRequest.
    virtual std::size_t getNeededMemory(const GPU& gpu) const override; 
    virtual std::size_t getMaxNeededMemory() const override;
    virtual std::vector<std::string> getCacheableTokens() const override;
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const override;
    virtual Request* clone(const Model& model) const override;
    virtual void act(GPU& gpu) override;
    virtual bool finished() const override;
    virtual void reserveMemory(GPU& gpu) override;
    virtual std::string getOutput() const override;
    virtual std::size_t getNeedToUnreserve() const override;
};

#endif