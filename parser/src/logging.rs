use std::fmt::Write;

pub struct Logger {
    effective_level: u32,
    buffer_level: u32,
    stderr_level: u32,
    buffer: String,
}

impl Clone for Logger {
    fn clone(&self) -> Self {
        Self {
            effective_level: self.effective_level,
            buffer_level: self.buffer_level,
            stderr_level: self.stderr_level,
            buffer: String::new(), // clean logs on clone
        }
    }
}

impl Logger {
    pub fn new(buffer_level: u32, stderr_level: u32) -> Self {
        Self {
            buffer_level,
            stderr_level,
            effective_level: std::cmp::max(buffer_level, stderr_level),
            buffer: String::new(),
        }
    }

    pub fn warn(&mut self, s: &str) {
        if self.level_enabled(1) {
            self.write_str("Warning: ").unwrap();
            self.write_str(s).unwrap();
            self.write_str("\n").unwrap();
        }
    }

    pub fn info(&mut self, s: &str) {
        if self.level_enabled(2) {
            self.write_str(s).unwrap();
            self.write_str("\n").unwrap();
        }
    }

    #[inline(always)]
    pub fn level_enabled(&self, level: u32) -> bool {
        level <= self.effective_level
    }

    #[inline(always)]
    pub fn effective_level(&self) -> u32 {
        self.effective_level
    }

    #[inline(always)]
    pub fn buffer_level(&self) -> u32 {
        self.buffer_level
    }

    #[inline(always)]
    pub fn stderr_level(&self) -> u32 {
        self.stderr_level
    }

    pub fn set_buffer_level(&mut self, buffer_level: u32) {
        self.buffer_level = buffer_level;
        self.effective_level = std::cmp::max(self.effective_level, self.buffer_level);
    }

    pub fn set_stderr_level(&mut self, stderr_level: u32) {
        self.stderr_level = stderr_level;
        self.effective_level = std::cmp::max(self.effective_level, self.stderr_level);
    }

    pub fn get_buffer(&self) -> &str {
        &self.buffer
    }

    pub fn get_and_clear_logs(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }
}

impl Write for Logger {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        if self.effective_level >= self.buffer_level {
            self.buffer.push_str(s);
        }
        if self.effective_level >= self.stderr_level {
            eprint!("{}", s);
        }
        Ok(())
    }
}
