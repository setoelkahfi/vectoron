//! Prints a GGUF file's metadata and tensors.
//!
//! ```sh
//! cargo run --release --example inspect -- model.gguf
//! ```

use std::collections::BTreeMap;

use vectoron::Gguf;
use vectoron::gguf::Value;

fn main() -> vectoron::Result<()> {
    let path = std::env::args().nth(1).expect("usage: inspect <file.gguf>");
    let gguf = Gguf::parse(std::fs::read(&path).expect("read the file"))?;

    println!("GGUF v{}, {} tensors", gguf.version, gguf.tensors.len());
    let mut keys: Vec<_> = gguf.metadata.iter().collect();
    keys.sort_by_key(|(k, _)| k.as_str());
    for (key, value) in keys {
        let shown = match value {
            Value::Array(items) => format!("[{} items]", items.len()),
            Value::String(s) if s.len() > 80 => {
                format!("{:?}...", s.chars().take(80).collect::<String>())
            }
            other => format!("{other:?}"),
        };
        println!("  {key} = {shown}");
    }

    let mut by_type: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for t in &gguf.tensors {
        let entry = by_type.entry(format!("{:?}", t.ty)).or_default();
        entry.0 += 1;
        entry.1 += t.size;
    }
    println!("tensor types:");
    for (ty, (count, bytes)) in by_type {
        println!("  {ty:6} {count:4} tensors, {:.1} MB", bytes as f64 / 1e6);
    }
    if std::env::args().any(|a| a == "--all") {
        for t in &gguf.tensors {
            println!("  {:40} {:?} {:?}", t.name, t.ty, t.dims);
        }
    }
    Ok(())
}
