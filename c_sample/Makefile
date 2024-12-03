TARGET = ../target/release

all:
	cd ../parser && cargo build --release
	# force static linking
	rm -f $(TARGET)/libllguidance.so $(TARGET)/libllguidance.dylib $(TARGET)/libllguidance.dll
	c++ -W -Wall -std=c++20 -o $(TARGET)/c_sample c_sample.cpp -I../parser -L$(TARGET) -lllguidance
	$(TARGET)/c_sample ../sample_parser/data/blog.schema.ll.json ../sample_parser/data/blog.sample.json

