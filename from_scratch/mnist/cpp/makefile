CXX = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -I./include
DEBUGFLAGS = -g -O0 -DDEBUG
RELEASEFLAGS = -O3 -march=native -DNDEBUG

SRCDIR = src
OBJDIR = build
BINDIR = .

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/a.out

RUN_CMD = ./$(TARGET)

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $(RELEASEFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(RELEASEFLAGS) -c $< -o $@

run: $(TARGET)
	$(RUN_CMD)

debug: RELEASEFLAGS = $(DEBUGFLAGS)
debug: clean $(TARGET)
	lldb ./$(TARGET)

clean:
	@rm -rf $(OBJDIR)/*.o $(TARGET)

.PHONY: all run debug clean help
