use anyhow::{Context, Result};
use blake3::Hasher;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

/// Ed25519 signature for job bundles
#[derive(Debug, Clone)]
pub struct Ed25519Signer {
    signing_key: SigningKey,
}

impl Ed25519Signer {
    /// Create a new signer with a random key
    pub fn new() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    /// Create from existing key bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let signing_key = SigningKey::from_bytes(
            bytes
                .try_into()
                .context("Invalid Ed25519 key length")?,
        );
        Ok(Self { signing_key })
    }

    /// Sign data
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        self.signing_key.sign(data).to_bytes().to_vec()
    }

    /// Get public key
    pub fn public_key(&self) -> Vec<u8> {
        self.signing_key.verifying_key().to_bytes().to_vec()
    }
}

impl Default for Ed25519Signer {
    fn default() -> Self {
        Self::new()
    }
}

/// Ed25519 signature verifier
#[derive(Debug, Clone)]
pub struct Ed25519Verifier {
    verifying_key: VerifyingKey,
}

impl Ed25519Verifier {
    /// Create from public key bytes
    pub fn from_public_key(public_key: &[u8]) -> Result<Self> {
        let verifying_key = VerifyingKey::from_bytes(
            public_key
                .try_into()
                .context("Invalid Ed25519 public key length")?,
        )
        .context("Invalid Ed25519 public key")?;
        Ok(Self { verifying_key })
    }

    /// Verify signature
    pub fn verify(&self, data: &[u8], signature: &[u8]) -> Result<()> {
        let sig = Signature::from_bytes(
            signature
                .try_into()
                .context("Invalid signature length")?,
        );
        self.verifying_key
            .verify(data, &sig)
            .context("Signature verification failed")?;
        Ok(())
    }
}

/// BLAKE3 Merkle tree hasher for job bundles
pub struct MerkleHasher;

impl MerkleHasher {
    /// Compute BLAKE3 hash of data
    pub fn hash(data: &[u8]) -> Vec<u8> {
        blake3::hash(data).as_bytes().to_vec()
    }

    /// Compute Merkle root of file chunks
    pub fn hash_chunks(chunks: &[&[u8]]) -> Vec<u8> {
        let mut hasher = Hasher::new();
        for chunk in chunks {
            hasher.update(chunk);
        }
        hasher.finalize().as_bytes().to_vec()
    }

    /// Verify hash matches expected
    pub fn verify(data: &[u8], expected_hash: &[u8]) -> Result<()> {
        let computed = Self::hash(data);
        if computed.as_slice() != expected_hash {
            anyhow::bail!(
                "Hash mismatch: expected {:?}, got {:?}",
                hex::encode(expected_hash),
                hex::encode(computed)
            );
        }
        Ok(())
    }
}

/// Bundle signature and verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleSignature {
    pub bundle_hash: Vec<u8>,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub timestamp: i64,
}

impl BundleSignature {
    /// Create new bundle signature
    pub fn sign(bundle_data: &[u8], signer: &Ed25519Signer) -> Self {
        let bundle_hash = MerkleHasher::hash(bundle_data);
        let signature = signer.sign(&bundle_hash);
        let public_key = signer.public_key();

        Self {
            bundle_hash,
            signature,
            public_key,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Verify bundle signature
    pub fn verify(&self, bundle_data: &[u8]) -> Result<()> {
        // Verify hash matches
        MerkleHasher::verify(bundle_data, &self.bundle_hash)?;

        // Verify signature
        let verifier = Ed25519Verifier::from_public_key(&self.public_key)?;
        verifier.verify(&self.bundle_hash, &self.signature)?;

        Ok(())
    }

    /// Check if signature is expired (optional timestamp validation)
    pub fn is_expired(&self, max_age_seconds: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        now - self.timestamp > max_age_seconds
    }
}

/// TUF (The Update Framework) keyring for bundle verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TufKeyring {
    pub trusted_keys: Vec<Vec<u8>>, // List of trusted public keys
}

impl TufKeyring {
    pub fn new() -> Self {
        Self {
            trusted_keys: Vec::new(),
        }
    }

    /// Add a trusted public key
    pub fn add_key(&mut self, public_key: Vec<u8>) {
        self.trusted_keys.push(public_key);
    }

    /// Verify bundle signature against any trusted key
    pub fn verify_bundle(&self, bundle_data: &[u8], signature: &BundleSignature) -> Result<()> {
        // First verify the signature itself
        signature.verify(bundle_data)?;

        // Then check if the public key is trusted
        if !self.trusted_keys.contains(&signature.public_key) {
            anyhow::bail!("Bundle signed by untrusted key");
        }

        Ok(())
    }
}

impl Default for TufKeyring {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ed25519_sign_verify() {
        let signer = Ed25519Signer::new();
        let data = b"test data";
        let signature = signer.sign(data);

        let verifier = Ed25519Verifier::from_public_key(&signer.public_key()).unwrap();
        assert!(verifier.verify(data, &signature).is_ok());

        // Wrong data should fail
        assert!(verifier.verify(b"wrong data", &signature).is_err());
    }

    #[test]
    fn test_merkle_hash() {
        let data = b"test data";
        let hash1 = MerkleHasher::hash(data);
        let hash2 = MerkleHasher::hash(data);
        assert_eq!(hash1, hash2);

        assert!(MerkleHasher::verify(data, &hash1).is_ok());
        assert!(MerkleHasher::verify(b"wrong", &hash1).is_err());
    }

    #[test]
    fn test_bundle_signature() {
        let signer = Ed25519Signer::new();
        let bundle_data = b"bundle contents";

        let sig = BundleSignature::sign(bundle_data, &signer);
        assert!(sig.verify(bundle_data).is_ok());
        assert!(sig.verify(b"wrong data").is_err());
    }

    #[test]
    fn test_tuf_keyring() {
        let signer = Ed25519Signer::new();
        let mut keyring = TufKeyring::new();
        keyring.add_key(signer.public_key());

        let bundle_data = b"bundle contents";
        let sig = BundleSignature::sign(bundle_data, &signer);

        assert!(keyring.verify_bundle(bundle_data, &sig).is_ok());

        // Untrusted signer should fail
        let untrusted_signer = Ed25519Signer::new();
        let untrusted_sig = BundleSignature::sign(bundle_data, &untrusted_signer);
        assert!(keyring.verify_bundle(bundle_data, &untrusted_sig).is_err());
    }
}
