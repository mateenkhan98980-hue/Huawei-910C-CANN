// ============================================================================
// Device Management Implementation - CANN 8.0
// ============================================================================

#include "ascendcl_wrapper.h"
#include <iostream>
#include <unordered_map>

namespace ascendcl {

class DeviceManager {
public:
    static DeviceManager& getInstance() {
        static DeviceManager instance;
        return instance;
    }
    
    Device& getDevice(int device_id) {
        if (devices_.find(device_id) == devices_.end()) {
            devices_[device_id] = std::make_unique<Device>(device_id);
        }
        return *devices_[device_id];
    }
    
private:
    DeviceManager() = default;
    std::unordered_map<int, std::unique_ptr<Device>> devices_;
};

Device& getDeviceOrCreate(int device_id) {
    return DeviceManager::getInstance().getDevice(device_id);
}

} // namespace ascendcl