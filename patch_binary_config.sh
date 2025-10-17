#!/bin/bash
# Script to patch binary with coordinator configuration
# This allows the server to inject its own URL into distributed binaries

set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <binary_path> <coordinator_host> <coordinator_port>"
    echo "Example: $0 gogrid-tray coordinator.example.com 8443"
    exit 1
fi

BINARY_PATH="$1"
COORD_HOST="$2"
COORD_PORT="$3"

if [ ! -f "$BINARY_PATH" ]; then
    echo "Error: Binary not found: $BINARY_PATH"
    exit 1
fi

# Validate inputs
if [ -z "$COORD_HOST" ]; then
    echo "Error: Coordinator host cannot be empty"
    exit 1
fi

if ! [[ "$COORD_PORT" =~ ^[0-9]+$ ]] || [ "$COORD_PORT" -lt 1 ] || [ "$COORD_PORT" -gt 65535 ]; then
    echo "Error: Invalid port number: $COORD_PORT"
    exit 1
fi

echo "Patching $BINARY_PATH with coordinator: $COORD_HOST:$COORD_PORT"

# Check if patterns exist in binary
if ! grep -q "__GOGRID_DEFAULT_HOST" "$BINARY_PATH" 2>/dev/null; then
    echo "Warning: __GOGRID_DEFAULT_HOST marker not found in binary"
    echo "Binary may already be patched or was not built with markers"
fi

if ! grep -q "__GOGRID_DEFAULT_PORT" "$BINARY_PATH" 2>/dev/null; then
    echo "Warning: __GOGRID_DEFAULT_PORT marker not found in binary"
    echo "Binary may already be patched or was not built with markers"
fi

# Create backup
cp "$BINARY_PATH" "$BINARY_PATH.backup"

# Prepare the replacement strings with proper padding
# Host marker is 40 characters: "__GOGRID_DEFAULT_HOST________________"
HOST_MARKER="__GOGRID_DEFAULT_HOST________________"
HOST_LENGTH=${#HOST_MARKER}

# Port marker is 25 characters: "__GOGRID_DEFAULT_PORT____"
PORT_MARKER="__GOGRID_DEFAULT_PORT____"
PORT_LENGTH=${#PORT_MARKER}

# Pad the host to match marker length (right-padded with underscores)
PADDED_HOST=$(printf "%-${HOST_LENGTH}s" "$COORD_HOST" | sed 's/ /_/g')
# Ensure it doesn't exceed the marker length
PADDED_HOST=${PADDED_HOST:0:$HOST_LENGTH}

# Pad the port to match marker length (right-padded with underscores)
PADDED_PORT=$(printf "%-${PORT_LENGTH}s" "$COORD_PORT" | sed 's/ /_/g')
# Ensure it doesn't exceed the marker length
PADDED_PORT=${PADDED_PORT:0:$PORT_LENGTH}

echo "Padded host: $PADDED_HOST (length: ${#PADDED_HOST})"
echo "Padded port: $PADDED_PORT (length: ${#PADDED_PORT})"

# Use perl for in-place binary substitution (works on both Linux and macOS)
if command -v perl &> /dev/null; then
    # Replace host marker
    perl -pi -e "s/\Q$HOST_MARKER\E/$PADDED_HOST/g" "$BINARY_PATH"

    # Replace port marker
    perl -pi -e "s/\Q$PORT_MARKER\E/$PADDED_PORT/g" "$BINARY_PATH"

    echo "✓ Binary patched successfully"
    echo "  Coordinator: $COORD_HOST:$COORD_PORT"

    # Verify the patch
    if grep -q "$COORD_HOST" "$BINARY_PATH" 2>/dev/null; then
        echo "✓ Verification: Host found in patched binary"
    else
        echo "⚠ Warning: Could not verify host in patched binary"
    fi

    if grep -q "$COORD_PORT" "$BINARY_PATH" 2>/dev/null; then
        echo "✓ Verification: Port found in patched binary"
    else
        echo "⚠ Warning: Could not verify port in patched binary"
    fi

    # Remove backup on success
    rm "$BINARY_PATH.backup"
else
    echo "Error: perl is required for binary patching"
    rm "$BINARY_PATH.backup"
    exit 1
fi

echo "✓ Done"
