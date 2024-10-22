ifeq ($(wildcard ../../target),)
    TARGET = ../parser/target/release
else
    TARGET = ../../target/release
endif

all:
	cd ../parser && cargo build --release
	c++ -W -Wall -std=c++20 -o $(TARGET)/c_sample c_sample.cpp -I../parser -L$(TARGET) -lllguidance_parser
	$(TARGET)/c_sample ../sample_parser/data/blog.schema.ll.json ../sample_parser/data/blog.sample.json

