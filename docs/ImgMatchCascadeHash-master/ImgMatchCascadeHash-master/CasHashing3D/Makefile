CC=g++
CFLAGS=-mpopcnt
LIBS=-lz
OPTFLAGS=-Wall -O3

all: KeyMatchCasHash

KeyMatchCasHash: Main.o BucketBuilder.o DataPreProcessor.o HashConvertor.o KeyFileReader.o CasHashMatcher.o MatchPairLoader.o Timer.o
	$(CC) Main.o BucketBuilder.o DataPreProcessor.o HashConvertor.o KeyFileReader.o CasHashMatcher.o MatchPairLoader.o Timer.o -o KeyMatchCasHash $(OPTFLAGS) $(LIBS)

Main.o: Main.cc
	$(CC) $(OPTFLAGS) $(CFLAGS) -c Main.cc

BucketBuilder.o: BucketBuilder.cc BucketBuilder.h
	$(CC) $(OPTFLAGS) $(CFLAGS) -c BucketBuilder.cc

DataPreProcessor.o: DataPreProcessor.cc DataPreProcessor.h
	$(CC) $(OPTFLAGS) $(CFLAGS) -c DataPreProcessor.cc

HashConvertor.o: HashConvertor.cc HashConvertor.h
	$(CC) $(OPTFLAGS) $(CFLAGS) -c HashConvertor.cc

KeyFileReader.o: KeyFileReader.cc KeyFileReader.h
	$(CC) $(OPTFLAGS) $(CFLAGS) -c KeyFileReader.cc

CasHashMatcher.o: CasHashMatcher.cc CasHashMatcher.h
	$(CC) $(OPTFLAGS) $(CFLAGS) -c CasHashMatcher.cc

MatchPairLoader.o: MatchPairLoader.cc MatchPairLoader.h
	$(CC) $(OPTFLAGS) $(CFLAGS) -c MatchPairLoader.cc

Timer.o: Timer.cc Timer.h
	$(CC) $(OPTFLAGS) $(CFLAGS) -c Timer.cc

clean:
	rm -rf *~ *.o KeyMatchCasHash8B
