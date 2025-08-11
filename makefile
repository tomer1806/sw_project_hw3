# Makefile for symnmf C executable
# This Makefile compiles the symnmf C program and links it with the math library

# Compiler and flags
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
# Target executable
TARGET = symnmf
# Source files
SRCS = symnmf.c
# Object files
OBJS = $(SRCS:.c=.o)
# Default rule
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm
%.o: %.c symnmf.h
	$(CC) $(CFLAGS) -c $< -o $@
# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)
.PHONY: all clean