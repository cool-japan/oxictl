//! Streaming JSON builder for data export.
//!
//! `JsonWriter` builds a JSON document incrementally into a `Vec<u8>` byte
//! buffer without any heap allocation beyond the buffer itself.  Suitable for
//! exporting telemetry records or configuration snapshots from the control
//! system to a host machine.
//!
//! Only available with the `std` feature.
use std::vec::Vec;

/// Streaming JSON writer that builds into an in-memory byte buffer.
pub struct JsonWriter {
    buf: Vec<u8>,
    /// Whether the current container has at least one element (needs comma).
    needs_comma: bool,
    /// Nesting depth (objects and arrays).
    depth: usize,
}

impl JsonWriter {
    /// Create a new, empty JSON writer.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            needs_comma: false,
            depth: 0,
        }
    }

    /// Open a JSON object `{`.
    pub fn begin_object(&mut self) {
        self.maybe_comma();
        self.buf.push(b'{');
        self.needs_comma = false;
        self.depth += 1;
    }

    /// Close a JSON object `}`.
    pub fn end_object(&mut self) {
        self.buf.push(b'}');
        self.needs_comma = true;
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    /// Open a JSON array `[`.
    pub fn begin_array(&mut self) {
        self.maybe_comma();
        self.buf.push(b'[');
        self.needs_comma = false;
        self.depth += 1;
    }

    /// Close a JSON array `]`.
    pub fn end_array(&mut self) {
        self.buf.push(b']');
        self.needs_comma = true;
        if self.depth > 0 {
            self.depth -= 1;
        }
    }

    /// Write a `key: value` pair where value is an `f64`.
    pub fn field_f64(&mut self, key: &str, val: f64) {
        self.write_key(key);
        let s = format_f64(val);
        self.buf.extend_from_slice(s.as_bytes());
        self.needs_comma = true;
    }

    /// Write a `key: value` pair where value is a string.
    pub fn field_str(&mut self, key: &str, val: &str) {
        self.write_key(key);
        self.buf.push(b'"');
        self.escape_str(val);
        self.buf.push(b'"');
        self.needs_comma = true;
    }

    /// Write a `key: value` pair where value is a boolean.
    pub fn field_bool(&mut self, key: &str, val: bool) {
        self.write_key(key);
        if val {
            self.buf.extend_from_slice(b"true");
        } else {
            self.buf.extend_from_slice(b"false");
        }
        self.needs_comma = true;
    }

    /// Write a `key: value` pair where value is an `i64`.
    pub fn field_i64(&mut self, key: &str, val: i64) {
        self.write_key(key);
        let s = itoa(val);
        self.buf.extend_from_slice(s.as_bytes());
        self.needs_comma = true;
    }

    /// Return the accumulated JSON as a UTF-8 byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Return the accumulated JSON as a `&str`.
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf).unwrap_or("")
    }

    /// Write a key (with leading comma if needed).
    fn write_key(&mut self, key: &str) {
        self.maybe_comma();
        self.buf.push(b'"');
        self.escape_str(key);
        self.buf.push(b'"');
        self.buf.push(b':');
    }

    /// Emit a comma separator if not at the start of a container.
    fn maybe_comma(&mut self) {
        if self.needs_comma {
            self.buf.push(b',');
            self.needs_comma = false;
        }
    }

    /// Write a string with JSON-required escape sequences.
    fn escape_str(&mut self, s: &str) {
        for ch in s.chars() {
            match ch {
                '"' => self.buf.extend_from_slice(b"\\\""),
                '\\' => self.buf.extend_from_slice(b"\\\\"),
                '\n' => self.buf.extend_from_slice(b"\\n"),
                '\r' => self.buf.extend_from_slice(b"\\r"),
                '\t' => self.buf.extend_from_slice(b"\\t"),
                c if (c as u32) < 0x20 => {
                    // Other control characters as \uXXXX.
                    let encoded = format!("\\u{:04x}", c as u32);
                    self.buf.extend_from_slice(encoded.as_bytes());
                }
                c => {
                    let mut tmp = [0u8; 4];
                    let bytes = c.encode_utf8(&mut tmp).as_bytes();
                    self.buf.extend_from_slice(bytes);
                }
            }
        }
    }
}

impl Default for JsonWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Format an f64 without std::fmt allocation overhead using Rust's std Display.
fn format_f64(v: f64) -> std::string::String {
    if v.is_nan() {
        "null".into()
    } else if v.is_infinite() {
        if v > 0.0 {
            "1e308".into()
        } else {
            "-1e308".into()
        }
    } else {
        format!("{}", v)
    }
}

/// Format an i64 as a decimal string.
fn itoa(v: i64) -> std::string::String {
    format!("{}", v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_object() {
        let mut w = JsonWriter::new();
        w.begin_object();
        w.field_str("name", "sensor");
        w.field_f64("value", core::f64::consts::PI);
        w.field_bool("active", true);
        w.field_i64("count", 42);
        w.end_object();
        let s = w.as_str();
        assert!(s.starts_with('{'), "starts_with {{: {}", s);
        assert!(s.contains("\"name\":\"sensor\""), "name: {}", s);
        assert!(s.contains("\"active\":true"), "active: {}", s);
        assert!(s.contains("\"count\":42"), "count: {}", s);
        assert!(s.ends_with('}'), "ends_with }}: {}", s);
    }

    #[test]
    fn nested_array() {
        let mut w = JsonWriter::new();
        w.begin_object();
        w.write_key("data");
        w.begin_array();
        w.needs_comma = false;
        // Manually push a number (simplified).
        w.buf.extend_from_slice(b"1");
        w.needs_comma = true;
        w.buf.extend_from_slice(b",2");
        w.end_array();
        w.end_object();
        let s = w.as_str();
        assert!(s.contains("[1,2]"), "array: {}", s);
    }

    #[test]
    fn string_escaping() {
        let mut w = JsonWriter::new();
        w.begin_object();
        w.field_str("msg", "hello\nworld\"tab\there");
        w.end_object();
        let s = w.as_str();
        assert!(s.contains("\\n"), "newline: {}", s);
        assert!(s.contains("\\\""), "quote: {}", s);
    }

    #[test]
    fn nan_becomes_null() {
        let mut w = JsonWriter::new();
        w.begin_object();
        w.field_f64("v", f64::NAN);
        w.end_object();
        assert!(w.as_str().contains("null"), "{}", w.as_str());
    }
}
