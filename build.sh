#!/bin/bash

# ============================================================================
# Ascend CANN 8.0 Wrapper Build Script
# ============================================================================

set -e

echo "╔════════════════════════════════════════════╗"
echo "║  Ascend CANN 8.0 Wrapper Build            ║"
echo "╚════════════════════════════════════════════╝"

# Set ASCEND paths (modify as needed)
export ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend}"
export ASCEND_DRIVER="${ASCEND_DRIVER:-/usr/local/Ascend/driver}"

echo "[INFO] ASCEND_HOME: $ASCEND_HOME"
echo "[INFO] ASCEND_DRIVER: $ASCEND_DRIVER"

# Validate paths
if [ ! -d "$ASCEND_HOME" ]; then
    echo "[ERROR] ASCEND_HOME does not exist: $ASCEND_HOME"
    exit 1
fi

if [ ! -d "$ASCEND_DRIVER" ]; then
    echo "[ERROR] ASCEND_DRIVER does not exist: $ASCEND_DRIVER"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "[BUILD] Running CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DASCEND_HOME=$ASCEND_HOME \
    -DASCEND_DRIVER=$ASCEND_DRIVER

if [ $? -ne 0 ]; then
    echo "[ERROR] CMake configuration failed"
    exit 1
fi

# Build
echo "[BUILD] Compiling..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed"
    exit 1
fi

# Install
echo "[BUILD] Installing..."
make install

if [ $? -ne 0 ]; then
    echo "[ERROR] Installation failed"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║  Build Completed Successfully!             ║"
echo "╚════════════════════════════════════════════╝"

echo ""
echo "Libraries installed in: $(pwd)/install/lib64"
ls -lah ./install/lib64/

echo ""
echo "Headers installed in: $(pwd)/install/include"
ls -lah ./install/include/

echo ""
echo "[TEST] Running ctest..."
if ctest --output-on-failure; then
    echo "[SUCCESS] All tests passed"
else
    echo "[ERROR] One or more tests failed"
    exit 1
fi

echo ""
echo "[SUCCESS] Build and tests completed"
exit 0