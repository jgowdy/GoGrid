# GoGrid Worker Auto-Update System

## Overview

The GoGrid Worker tray application includes automatic update functionality that checks for new versions on startup and every 24 hours, downloading and installing updates seamlessly.

## Architecture

### Update Flow

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Tray App   │────────>│   bx.ee:8443 │────────>│   Update    │
│  (Client)   │  Check  │  Coordinator │ Manifest│   Package   │
│             │<────────│    Server    │<────────│  (.tar.gz)  │
└─────────────┘ Download└──────────────┘ Install └─────────────┘
```

### Components

1. **Client Side** (Tray App):
   - Tauri updater plugin
   - Checks `/updates/{target}/{version}` endpoint
   - Downloads and verifies update package
   - Installs and restarts automatically

2. **Server Side** (bx.ee:8443):
   - HTTP server on port 8443
   - Serves update manifests (JSON)
   - Serves update packages (.tar.gz, .zip)
   - Static file serving from `/opt/gogrid/updates/`

## Implementation Details

### Client Configuration

`tauri.conf.json` (Tauri 2.0):
```json
{
  "plugins": {
    "updater": {
      "endpoints": [
        "https://bx.ee:8443/updates/{{target}}/{{current_version}}"
      ],
      "pubkey": ""
    }
  }
}
```

And in `Cargo.toml`:
```toml
[dependencies]
tauri = { version = "2.0", features = ["tray-icon"] }
tauri-plugin-updater = "2.0"
```

In `main.rs`:
```rust
tauri::Builder::default()
    .plugin(tauri_plugin_updater::Builder::new().build())
    // ... other config
```

### Update Check Logic

The client automatically checks for updates:
- **On startup**: After 5 seconds delay
- **Periodically**: Every 24 hours
- **Process**:
  1. Checks coordinator endpoint
  2. Compares version numbers
  3. Downloads if newer version available
  4. Verifies signature (if configured)
  5. Installs and prompts restart

### Server Endpoints

#### GET `/updates/{target}/{version}`

Returns update manifest if newer version available:

**Request:**
```
GET https://bx.ee:8443/updates/darwin-aarch64/0.1.0
```

**Response (200 OK):**
```json
{
  "version": "0.1.1",
  "notes": "Update to latest version with improvements",
  "pub_date": "2025-10-17T19:15:00Z",
  "platforms": {
    "darwin-aarch64": {
      "signature": "",
      "url": "https://bx.ee:8443/downloads/GoGrid_Worker_0.1.1_aarch64.app.tar.gz"
    }
  }
}
```

**Response (204 No Content):** When up to date

#### GET `/downloads/{file}`

Serves the actual update package.

### Platform Targets

| Platform | Target String | Package Format |
|----------|--------------|----------------|
| macOS ARM | `darwin-aarch64` | `.app.tar.gz` |
| macOS Intel | `darwin-x86_64` | `.app.tar.gz` |
| Windows | `windows-x86_64` | `-setup.nsis.zip` |
| Linux | `linux-x86_64` | `.AppImage.tar.gz` |

## Deployment

### Building Updates

1. **Build the application:**
   ```bash
   cd /Users/jgowdy/GoGrid
   cargo build --release --bin corpgrid-scheduler
   cp target/release/corpgrid-scheduler target/release/corpgrid-scheduler-aarch64-apple-darwin
   cd crates/tray-app/src-tauri
   cargo tauri build
   ```

2. **Package for updates:**
   ```bash
   cd /Users/jgowdy/GoGrid/target/release/bundle/macos
   tar -czf "GoGrid_Worker_0.1.0_aarch64.app.tar.gz" "GoGrid Worker.app"
   ```

3. **Upload to server:**
   ```bash
   ./deploy_updates.sh
   ```

### Manual Deployment

```bash
# Create updates directory
ssh bx.ee "doas mkdir -p /opt/gogrid/updates && doas chown jgowdy:jgowdy /opt/gogrid/updates"

# Upload update package
scp target/release/bundle/macos/GoGrid_Worker_0.1.0_aarch64.app.tar.gz bx.ee:/opt/gogrid/updates/

# Rebuild coordinator
cargo build --release --bin gogrid-coordinator
scp target/release/gogrid-coordinator bx.ee:/opt/gogrid/bin/

# Restart coordinator
ssh bx.ee "doas systemctl restart gogrid-coordinator"
```

### Update Manifest

The server automatically generates manifests based on:
- Client's current version
- Client's target platform
- Available updates in `/opt/gogrid/updates/`

## Security

### Signature Verification (TODO)

For production, updates should be signed:

1. **Generate signing key:**
   ```bash
   tauri signer generate -w ~/.tauri/gogrid.key
   ```

2. **Add public key to `tauri.conf.json`:**
   ```json
   {
     "updater": {
       "pubkey": "dW50cnVzdGVkIGNvbW1lbnQ6..."
     }
   }
   ```

3. **Sign packages:**
   ```bash
   tauri signer sign /path/to/package.tar.gz
   ```

### HTTPS (TODO)

Currently serving over HTTP. For production:

1. Install Let's Encrypt certificate on bx.ee
2. Configure HTTPS in coordinator
3. Update endpoint URLs to use `https://`

## Shortest Path Strategy

To minimize risk of breaking the updater:

### What We DO:

1. ✅ **Check version on server** - Server compares versions, simple logic
2. ✅ **Download complete package** - Full app bundle, not patches
3. ✅ **Let Tauri handle installation** - Built-in, well-tested updater
4. ✅ **Static file serving** - No complex backend logic
5. ✅ **Manual rollback** - Old DMG still available for download

### What We DON'T Do:

1. ❌ **Delta updates** - Too complex, risk of corruption
2. ❌ **Multi-stage updates** - More moving parts
3. ❌ **Database for versions** - Static files are simpler
4. ❌ **Custom installers** - Use platform-native methods
5. ❌ **Automatic rollback** - Keep it simple, manual only

### Failure Recovery

If an update fails:

1. **Old version still works** - Update happens in background
2. **Retry on next check** - 24 hour cycle
3. **Manual download available** - DMG on server
4. **Logs for debugging** - Check client logs

## Testing

### Local Testing

1. **Build v0.1.1:**
   ```bash
   # Update version in tauri.conf.json
   cargo tauri build
   ```

2. **Setup local server:**
   ```bash
   python3 -m http.server 8443 -d /opt/gogrid/updates
   ```

3. **Test update check:**
   - Run app with v0.1.0
   - Check logs for update detection
   - Verify download and install

### Production Testing

1. **Deploy to staging server**
2. **Test with single client**
3. **Verify update process**
4. **Roll out to all clients**

## Monitoring

### Server Logs

```bash
# View coordinator logs
ssh bx.ee "doas journalctl -u gogrid-coordinator -f"

# Check update requests
ssh bx.ee "doas tail -f /var/log/nginx/access.log | grep /updates"
```

### Client Logs

macOS:
```bash
tail -f ~/Library/Logs/GoGrid\ Worker/gogrid-tray.log | grep -i update
```

## Version Bumping

To release a new version:

1. **Update version:**
   ```bash
   # In tauri.conf.json
   "version": "0.1.1"

   # In Cargo.toml
   version = "0.1.1"
   ```

2. **Build and test:**
   ```bash
   cargo tauri build
   ```

3. **Package for updates:**
   ```bash
   cd target/release/bundle/macos
   tar -czf "GoGrid_Worker_0.1.1_aarch64.app.tar.gz" "GoGrid Worker.app"
   ```

4. **Deploy:**
   ```bash
   ./deploy_updates.sh
   ```

5. **Update coordinator manifest:**
   Edit `/opt/gogrid/updates/latest.json` if using static manifest

## File Structure

```
/opt/gogrid/
├── bin/
│   └── gogrid-coordinator          # Coordinator binary
├── updates/
│   ├── GoGrid_Worker_0.1.0_aarch64.app.tar.gz
│   ├── GoGrid Worker_0.1.0_aarch64.dmg  # Manual download
│   └── latest.json                 # Optional static manifest
└── logs/
    └── coordinator.log
```

## Troubleshooting

### Updates not detected

1. Check server is running:
   ```bash
   curl -I https://bx.ee:8443/updates/darwin-aarch64/0.1.0
   ```

2. Check client logs for errors

3. Verify network connectivity

### Update download fails

1. Check file permissions on server
2. Verify file exists in `/opt/gogrid/updates/`
3. Check disk space

### Update install fails

1. Check Tauri updater logs
2. Verify package format is correct
3. Test manual install from DMG

## Future Improvements

1. **Signature verification** - Add public key signing
2. **HTTPS** - Use proper TLS certificates
3. **CDN** - Use CloudFront/CloudFlare for distribution
4. **Incremental updates** - Only download changed files
5. **Automatic rollback** - Detect crashes and revert
6. **A/B testing** - Gradual rollout to subset of users
7. **Update channels** - Stable, beta, alpha
8. **Metrics** - Track update success rates

## References

- [Tauri Updater Docs](https://tauri.app/v1/guides/distribution/updater/)
- [Update Manifest Format](https://tauri.app/v1/api/config/#updaterconfig)
- [Code Signing](https://tauri.app/v1/guides/building/code-signing/)
