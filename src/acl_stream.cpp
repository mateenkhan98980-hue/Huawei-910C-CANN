// ============================================================================
// Stream Operations Implementation - CANN 8.0
// ============================================================================

#include "ascendcl_wrapper.h"
#include <iostream>

namespace ascendcl {

class StreamPool {
public:
    static StreamPool& getInstance() {
        static StreamPool instance;
        return instance;
    }
    
    std::shared_ptr<Stream> createStream() {
        auto stream = std::make_shared<Stream>();
        std::cout << "[StreamPool] Created stream" << std::endl;
        return stream;
    }
    
private:
    StreamPool() = default;
};

std::shared_ptr<Stream> createStream() {
    return StreamPool::getInstance().createStream();
}

} // namespace ascendcl