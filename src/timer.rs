use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::sleep;
use std::thread::spawn;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone)]
pub struct Timer {
    start_time: Instant,
    timeout: Arc<AtomicBool>,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            timeout: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn start(&mut self, duration: Duration) {
        self.timeout = if duration == Duration::default() {
            Arc::new(AtomicBool::new(false))
        } else {
            timeout_signal(duration)
        };
        self.start_time = Instant::now();
    }

    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(self.start_time)
    }

    pub fn done(&self) -> bool {
        self.timeout.load(std::sync::atomic::Ordering::Relaxed)
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
        signal2.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    signal
}
