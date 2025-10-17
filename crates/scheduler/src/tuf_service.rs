use anyhow::Result;
use corpgrid_common::crypto::{BundleSignature, Ed25519Signer, TufKeyring};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// TUF (The Update Framework) metadata service
/// Manages trusted keys and provides signature verification
pub struct TufService {
    keyring: Arc<RwLock<TufKeyring>>,
    signers: Arc<RwLock<HashMap<String, Ed25519Signer>>>,
}

impl TufService {
    pub fn new() -> Self {
        Self {
            keyring: Arc::new(RwLock::new(TufKeyring::new())),
            signers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a trusted public key to the keyring
    pub async fn add_trusted_key(&self, key_id: String, public_key: Vec<u8>) -> Result<()> {
        let mut keyring = self.keyring.write().await;
        keyring.add_key(public_key.clone());

        info!(
            key_id = %key_id,
            key_len = public_key.len(),
            "Added trusted key to TUF keyring"
        );

        Ok(())
    }

    /// Register a signer for job bundle signing
    pub async fn add_signer(&self, key_id: String, signer: Ed25519Signer) -> Result<()> {
        let mut signers = self.signers.write().await;
        let public_key = signer.public_key();

        // Also add to keyring as trusted
        let mut keyring = self.keyring.write().await;
        keyring.add_key(public_key.clone());

        signers.insert(key_id.clone(), signer);

        info!(
            key_id = %key_id,
            "Registered TUF signer"
        );

        Ok(())
    }

    /// Sign a job bundle
    pub async fn sign_bundle(&self, key_id: &str, bundle_data: &[u8]) -> Result<BundleSignature> {
        let signers = self.signers.read().await;

        let signer = signers
            .get(key_id)
            .ok_or_else(|| anyhow::anyhow!("Signer not found: {}", key_id))?;

        let signature = BundleSignature::sign(bundle_data, signer);

        info!(
            key_id = %key_id,
            bundle_size = bundle_data.len(),
            "Signed job bundle"
        );

        Ok(signature)
    }

    /// Verify a job bundle signature
    pub async fn verify_bundle(
        &self,
        bundle_data: &[u8],
        signature: &BundleSignature,
    ) -> Result<()> {
        let keyring = self.keyring.read().await;

        keyring
            .verify_bundle(bundle_data, signature)
            .map_err(|e| {
                warn!(error = %e, "Bundle signature verification failed");
                e
            })?;

        info!(
            bundle_size = bundle_data.len(),
            "Bundle signature verified successfully"
        );

        Ok(())
    }

    /// Check if a signature is expired
    pub fn is_signature_expired(&self, signature: &BundleSignature, max_age_seconds: i64) -> bool {
        signature.is_expired(max_age_seconds)
    }

    /// Get number of trusted keys
    pub async fn trusted_key_count(&self) -> usize {
        let keyring = self.keyring.read().await;
        keyring.trusted_keys.len()
    }

    /// Rotate keys - remove old key and add new one
    pub async fn rotate_key(
        &self,
        old_key_id: String,
        new_key_id: String,
        new_signer: Ed25519Signer,
    ) -> Result<()> {
        info!(
            old_key_id = %old_key_id,
            new_key_id = %new_key_id,
            "Rotating TUF key"
        );

        // Add new signer
        self.add_signer(new_key_id.clone(), new_signer).await?;

        // Remove old signer
        let mut signers = self.signers.write().await;
        signers.remove(&old_key_id);

        // Note: We keep the old public key in the keyring to verify
        // bundles signed with the old key during the transition period

        info!(
            old_key_id = %old_key_id,
            new_key_id = %new_key_id,
            "Key rotation completed"
        );

        Ok(())
    }
}

impl Default for TufService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tuf_service() {
        let service = TufService::new();

        // Create a signer
        let signer = Ed25519Signer::new();
        service.add_signer("key1".to_string(), signer.clone()).await.unwrap();

        // Sign a bundle
        let bundle_data = b"test bundle data";
        let signature = service.sign_bundle("key1", bundle_data).await.unwrap();

        // Verify the signature
        assert!(service.verify_bundle(bundle_data, &signature).await.is_ok());

        // Verify with wrong data should fail
        assert!(service.verify_bundle(b"wrong data", &signature).await.is_err());
    }

    #[tokio::test]
    async fn test_key_rotation() {
        let service = TufService::new();

        let old_signer = Ed25519Signer::new();
        let new_signer = Ed25519Signer::new();

        service.add_signer("key1".to_string(), old_signer.clone()).await.unwrap();

        // Sign with old key
        let bundle_data = b"test bundle";
        let old_signature = service.sign_bundle("key1", bundle_data).await.unwrap();

        // Rotate to new key
        service.rotate_key("key1".to_string(), "key2".to_string(), new_signer.clone()).await.unwrap();

        // Old signature should still verify (key still in keyring)
        assert!(service.verify_bundle(bundle_data, &old_signature).await.is_ok());

        // New signature should also verify
        let new_signature = service.sign_bundle("key2", bundle_data).await.unwrap();
        assert!(service.verify_bundle(bundle_data, &new_signature).await.is_ok());
    }
}
