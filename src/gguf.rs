//! Reader for the GGUF model file format (versions 2 and 3).
//!
//! The reader takes ownership of the whole file as one `Vec<u8>` and never
//! copies it. Tensors are described by offsets into that buffer, so a model
//! built from a [`Gguf`] reads its weights in place.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::quant::GgmlType;

const MAGIC: u32 = 0x4655_4747; // "GGUF", little endian
const DEFAULT_ALIGNMENT: usize = 32;

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// The value as an unsigned integer, if it is a non-negative integer.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// The value as a float, if it is a number.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }

    /// The value as a string slice, if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// The value as a slice of values, if it is an array.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// Where one tensor lives in the file and how it is stored.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions, innermost first, as stored in the file. A matrix with
    /// `rows` rows of `cols` values has `dims == [cols, rows]`.
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    /// Byte offset from the start of the file.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
}

impl TensorInfo {
    /// Number of values in the tensor.
    pub fn elements(&self) -> usize {
        self.dims.iter().product()
    }
}

/// A parsed GGUF file.
#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    bytes: Arc<Vec<u8>>,
}

impl Gguf {
    /// Parses a complete GGUF file held in memory.
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let (version, metadata, tensors) = {
            let mut r = Reader {
                buf: &bytes,
                pos: 0,
            };
            let magic = r.u32()?;
            if magic != MAGIC {
                return Err(Error::Gguf("bad magic number".into()));
            }
            let version = r.u32()?;
            if !(2..=3).contains(&version) {
                return Err(Error::Gguf(format!("unsupported version {version}")));
            }
            let tensor_count = r.len()?;
            let kv_count = r.len()?;

            let mut metadata = HashMap::with_capacity(kv_count.min(1024));
            for _ in 0..kv_count {
                let key = r.string()?;
                let ty = r.u32()?;
                let value = r.value(ty)?;
                metadata.insert(key, value);
            }

            let alignment = match metadata.get("general.alignment") {
                Some(v) => v
                    .as_u64()
                    .and_then(|a| usize::try_from(a).ok())
                    .filter(|a| *a > 0)
                    .ok_or_else(|| Error::Metadata("general.alignment".into()))?,
                None => DEFAULT_ALIGNMENT,
            };

            let mut raw = Vec::with_capacity(tensor_count.min(1 << 16));
            for _ in 0..tensor_count {
                let name = r.string()?;
                let n_dims = r.u32()? as usize;
                let mut dims = Vec::with_capacity(n_dims);
                for _ in 0..n_dims {
                    dims.push(r.len()?);
                }
                let ty = GgmlType::from_id(r.u32()?)?;
                let offset = r.len()?;
                raw.push((name, dims, ty, offset));
            }

            let data_start = r.pos.div_ceil(alignment) * alignment;
            let mut tensors = Vec::with_capacity(raw.len());
            for (name, dims, ty, offset) in raw {
                let elements: usize = dims.iter().product();
                let inner = dims.first().copied().unwrap_or(1);
                if inner % ty.block_size() != 0 {
                    return Err(Error::Gguf(format!(
                        "tensor `{name}` has rows of {inner} values, not a multiple of the block size"
                    )));
                }
                let size = elements / ty.block_size() * ty.type_size();
                let offset = data_start
                    .checked_add(offset)
                    .filter(|start| {
                        start
                            .checked_add(size)
                            .is_some_and(|end| end <= bytes.len())
                    })
                    .ok_or_else(|| Error::Gguf(format!("tensor `{name}` runs past the end")))?;
                tensors.push(TensorInfo {
                    name,
                    dims,
                    ty,
                    offset,
                    size,
                });
            }
            (version, metadata, tensors)
        };

        Ok(Self {
            version,
            metadata,
            tensors,
            bytes: Arc::new(bytes),
        })
    }

    /// The metadata value for `key`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// The tensor called `name`.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The raw bytes of a tensor.
    pub fn data(&self, info: &TensorInfo) -> &[u8] {
        &self.bytes[info.offset..info.offset + info.size]
    }

    /// The whole file. Tensor offsets index into this buffer.
    pub fn bytes(&self) -> &Arc<Vec<u8>> {
        &self.bytes
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|end| *end <= self.buf.len())
            .ok_or_else(|| Error::Gguf("unexpected end of file".into()))?;
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// A u64 count or size that must fit in memory.
    fn len(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?)
            .map_err(|_| Error::Gguf("length does not fit in memory".into()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.len()?;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(u8::from_le_bytes(self.array()?)),
            1 => Value::I8(i8::from_le_bytes(self.array()?)),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => Value::Bool(self.array::<1>()?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem = self.u32()?;
                let n = self.len()?;
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(elem)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => return Err(Error::Gguf(format!("unknown metadata type {other}"))),
        })
    }
}
