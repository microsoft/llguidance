use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    static ref FORMAT_PATTERNS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("email", r"^[^\s@]+@[^\s@]+\.[^\s@]+$");
        m.insert("ipv4", r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$");
        m.insert("date", r"^\d{4}-\d{2}-\d{2}$");
        // Add more patterns as needed
        m
    };
}

pub fn lookup_format(name: &str) -> Option<&str> {
    FORMAT_PATTERNS.get(name).copied()
}
