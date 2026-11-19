# Builds the entire project
all: bin/server

# Links all object files to create the executable
bin/server: bin/Main.o
	g++ -o bin/server bin/Main.o

# Compiles the source file
bin/Main.o: src/Main.cpp
	g++ -g -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o

# Cleans up compiled files
clean:
	rm -f bin/*.o bin/server