#include <string>

#ifndef PCAP_READER_H
#define PCAP_READER_H

namespace PCAPReader {

class PCAPReader {

private:

public:
    PCAPReader();
    std::string get_array() const;
};

}

#endif
