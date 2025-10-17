use anyhow::{Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_s3::primitives::ByteStream;
use bytes::Bytes;
use tracing::{debug, info};

/// S3-compatible content-addressable storage
pub struct CasStorage {
    client: S3Client,
    bucket: String,
}

impl CasStorage {
    pub async fn new(bucket: String) -> Result<Self> {
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let client = S3Client::new(&config);

        Ok(Self { client, bucket })
    }

    /// Upload data to S3 with content-addressed key
    pub async fn put(&self, key: &str, data: Bytes) -> Result<()> {
        debug!(
            key = %key,
            size = data.len(),
            "Uploading to S3"
        );

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(data))
            .send()
            .await
            .context("Failed to upload to S3")?;

        info!(
            key = %key,
            "Upload complete"
        );

        Ok(())
    }

    /// Download data from S3
    pub async fn get(&self, key: &str) -> Result<Bytes> {
        debug!(
            key = %key,
            "Downloading from S3"
        );

        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .context("Failed to download from S3")?;

        let data = response
            .body
            .collect()
            .await
            .context("Failed to read S3 response body")?
            .into_bytes();

        info!(
            key = %key,
            size = data.len(),
            "Download complete"
        );

        Ok(data)
    }

    /// Check if object exists
    pub async fn exists(&self, key: &str) -> Result<bool> {
        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                // Check if it's a 404
                if e.to_string().contains("NotFound") {
                    Ok(false)
                } else {
                    Err(e).context("Failed to check S3 object existence")
                }
            }
        }
    }

    /// Delete object
    pub async fn delete(&self, key: &str) -> Result<()> {
        debug!(
            key = %key,
            "Deleting from S3"
        );

        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .context("Failed to delete from S3")?;

        info!(
            key = %key,
            "Delete complete"
        );

        Ok(())
    }

    /// Generate content-addressed key from hash
    pub fn hash_to_key(hash: &[u8], prefix: &str) -> String {
        let hex = hex::encode(hash);
        format!("{}/{}/{}/{}", prefix, &hex[0..2], &hex[2..4], hex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_to_key() {
        let hash = b"test_hash_1234567890";
        let key = CasStorage::hash_to_key(hash, "bundles");

        assert!(key.starts_with("bundles/"));
        assert!(key.contains("/"));
    }
}
