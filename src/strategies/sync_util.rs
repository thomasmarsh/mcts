use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{sleep, spawn};
use std::time::{Duration, Instant};

pub struct Timer {
    start_time: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(self.start_time)
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

pub(super) fn timeout_signal(dur: Duration) -> Arc<AtomicBool> {
    // Theoretically we could include an async runtime to do this and use
    // fewer threads, but the stdlib implementation is only a few lines...
    let signal = Arc::new(AtomicBool::new(false));
    let signal2 = signal.clone();
    spawn(move || {
        sleep(dur);
        signal2.store(true, Ordering::Relaxed);
    });
    signal
}

// 64-bytes is a common cache line size.
#[repr(align(64))]
pub(super) struct CachePadded<T> {
    value: T,
}

impl<T: Default> Default for CachePadded<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
        }
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
