SRC = Convert.cpp Convolve.cpp Image.cpp ImageIO.cpp ImageIOpng.cpp RefCntMem.cpp

CC = g++
WARN = -W -Wall
OPT ?= -O3
CPPFLAGS = $(OPT) $(WARN)

OBJ = $(SRC:.cpp=.o)

all: libImg.a

libImg.a: $(OBJ)
	rm -f libImg.a
	ar ruc libImg.a $(OBJ)
	ranlib libImg.a

clean: 
	rm -f $(OBJ) core *.stackdump *.bak

allclean: clean
	rm -f libImg.a

depend:
	@makedepend -Y -- $(CPPFLAGS) -- $(SRC) 2>> /dev/null

# DO NOT DELETE THIS LINE -- make depend depends on it.

Convert.o: Image.h RefCntMem.h Error.h Convert.h
Convolve.o: Image.h RefCntMem.h Error.h Convert.h Convolve.h
Image.o: Image.h RefCntMem.h Error.h
ImageIO.o: Image.h RefCntMem.h Error.h ImageIO.h
ImageIOpng.o: Image.h RefCntMem.h Error.h
RefCntMem.o: RefCntMem.h
