use std::{any::Any, panic::UnwindSafe};

use anyhow::Result;

pub fn mk_panic_error(info: &Box<dyn Any + Send>) -> String {
    let msg = match info.downcast_ref::<&'static str>() {
        Some(s) => *s,
        None => match info.downcast_ref::<String>() {
            Some(s) => &s[..],
            None => "non-string panic!()",
        },
    };

    format!("panic: {msg}")
}

pub fn catch_unwind<F: FnOnce() -> Result<R> + UnwindSafe, R>(f: F) -> Result<R> {
    match std::panic::catch_unwind(f) {
        Ok(r) => r,
        Err(e) => Err(anyhow::anyhow!(mk_panic_error(&e))),
    }
}
