/// Everything that can go wrong while loading or running a model.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file is not valid GGUF, or ends early.
    #[error("invalid GGUF file: {0}")]
    Gguf(String),
    /// A tensor uses a storage type this crate does not implement.
    #[error("unsupported tensor type {0}")]
    UnsupportedType(u32),
    /// A metadata key the model needs is absent or has the wrong type.
    #[error("missing or invalid metadata key `{0}`")]
    Metadata(String),
    /// A tensor the model needs is absent.
    #[error("missing tensor `{0}`")]
    MissingTensor(String),
    /// The weights or the request do not fit the model.
    #[error("{0}")]
    Model(String),
}

/// Shorthand for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;
