//! # fallibles-core
//! This module contains the core logic for failure injection and configuration.

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Trait for error types that can be generated during simulated failures.
///
/// Implement this trait for your custom error types to use them with `#[fallible]`.
/// The trait provides a default error value to return when a failure is triggered.
///
/// # Example
/// ```
/// use fallibles::fallibles_core::FallibleError;
///
/// #[derive(Debug)]
/// struct MyError { message: String }
///
/// impl FallibleError for MyError {
///     fn simulated_failure() -> Self {
///         MyError { message: "test failure".to_string() }
///     }
/// }
/// ```
pub trait FallibleError {
    fn simulated_failure() -> Self;
}

impl FallibleError for &'static str {
    fn simulated_failure() -> Self {
        "simulated failure"
    }
}

impl FallibleError for alloc::string::String {
    fn simulated_failure() -> Self {
        alloc::string::String::from("simulated failure")
    }
}

impl<T: FallibleError> FallibleError for alloc::boxed::Box<T> {
    fn simulated_failure() -> Self {
        alloc::boxed::Box::new(T::simulated_failure())
    }
}

#[cfg(feature = "std")]
impl FallibleError for std::io::Error {
    fn simulated_failure() -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, "simulated failure")
    }
}

#[cfg(feature = "anyhow")]
impl FallibleError for anyhow::Error {
    fn simulated_failure() -> Self {
        anyhow::anyhow!("simulated failure")
    }
}

#[cfg(feature = "eyre")]
impl FallibleError for eyre::Report {
    fn simulated_failure() -> Self {
        eyre::eyre!("simulated failure")
    }
}

impl FallibleError for () {
    fn simulated_failure() -> Self {}
}

impl FallibleError for bool {
    fn simulated_failure() -> Self {
        false
    }
}

impl<T> FallibleError for Option<T> {
    fn simulated_failure() -> Self {
        None
    }
}

/// Handler trait for custom failure behavior.
///
/// This should only be used if you need complete control over
/// what happens during failures.
pub trait FailureHandler {
    fn handle(&self, fp: FailurePoint) -> !;
}

/// Unique identifier for a failure point.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FailurePointId(pub u32);

/// Information about a specific failure point.
///
/// Contains location metadata (file, line, column) and a unique identifier.
/// This is passed to callbacks for observability and debugging.
#[derive(Copy, Clone, Debug)]
pub struct FailurePoint {
    pub id: FailurePointId,
    pub function: &'static str,
    pub file: &'static str,
    pub line: u32,
    pub column: u32,
}

pub struct PanicHandler;

impl FailureHandler for PanicHandler {
    fn handle(&self, fp: FailurePoint) -> ! {
        panic!(
            "fallible simulated failure {:?} at {}:{}:{} ({})",
            fp.id, fp.file, fp.line, fp.column, fp.function,
        );
    }
}

static GLOBAL_HANDLER_DATA: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_HANDLER_VTABLE: AtomicUsize = AtomicUsize::new(0);
static CONFIG_PTR: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "std")]
use std::cell::RefCell;

#[cfg(feature = "std")]
std::thread_local! {
    static THREAD_CONFIG_PTR: RefCell<usize> = const { RefCell::new(0) };
}

/// Callback function type for observability hooks.
///
/// Used with `on_check()` and `on_failure()` to monitor failures.
pub type FailureCallback = Box<dyn Fn(FailurePoint) + Send + Sync>;

/// Predicate function type for conditional failure injection.
///
/// Used with `when()` to dynamically control if a failure can occur.
pub type FailurePredicate = Box<dyn Fn() -> bool + Send + Sync>;

/// Statistics about failure behavior.
///
/// Tracks how many times failure points were checked and how many failures were triggered.
#[derive(Clone, Debug)]
pub struct FailureStats {
    /// Total number of times failure points were evaluated
    pub total_checks: u64,
    /// Total number of failures that were actually triggered
    pub total_failures: u64,
    /// Number of checks that would have failed but were blocked by limits
    pub limited_failures: u64,
    /// Total latency injected in nanoseconds
    #[cfg(feature = "std")]
    pub total_latency_ns: u64,
}

impl FailureStats {
    /// Get the failure rate as a percentage (0.0 to 100.0).
    pub fn failure_rate(&self) -> f64 {
        if self.total_checks == 0 {
            0.0
        } else {
            (self.total_failures as f64 / self.total_checks as f64) * 100.0
        }
    }

    /// Get the success rate as a percentage (0.0 to 100.0).
    pub fn success_rate(&self) -> f64 {
        100.0 - self.failure_rate()
    }

    /// Print statistics report to stdout.
    ///
    /// Uses ANSI colors: green for successes, red for failures, yellow for limited.
    #[cfg(feature = "std")]
    pub fn report(&self) {
        use std::println;

        const RESET: &str = "\x1b[0m";
        const GREEN: &str = "\x1b[32m";
        const RED: &str = "\x1b[31m";
        const YELLOW: &str = "\x1b[33m";
        const CYAN: &str = "\x1b[36m";
        const BOLD: &str = "\x1b[1m";

        println!("{}{}failure injection statistics:{}", BOLD, CYAN, RESET);
        println!("  total checks:     {}{}{}", CYAN, self.total_checks, RESET);
        println!(
            "  failures:         {}{}{} ({:.1}%)",
            RED, self.total_failures, RESET, self.failure_rate()
        );
        println!(
            "  successes:        {}{}{} ({:.1}%)",
            GREEN,
            self.total_checks.saturating_sub(self.total_failures),
            RESET,
            self.success_rate()
        );
        if self.limited_failures > 0 {
            println!(
                "  limited:          {}{}{} (blocked by max_failures)",
                YELLOW, self.limited_failures, RESET
            );
        }
        if self.total_latency_ns > 0 {
            let latency_ms = self.total_latency_ns as f64 / 1_000_000.0;
            println!(
                "  total latency:    {}{:.2}ms{}",
                CYAN, latency_ms, RESET
            );
            if self.total_checks > 0 {
                let avg_latency_us = self.total_latency_ns as f64 / self.total_checks as f64 / 1000.0;
                println!(
                    "  avg per check:    {}{:.2}µs{}",
                    CYAN, avg_latency_us, RESET
                );
            }
        }
    }
}

/// Configuration for failure injection behavior.
///
/// Controls when and how failures are triggered. It supports probability-based,
/// deterministic (every nth call), seeded for reproducible randomness, and includes
/// preset policies for triggering failures.
///
/// # Example
/// ```
/// use fallibles::fallibles_core::FailureConfig;
///
/// // Fail 30% of the time
/// let config = FailureConfig::new().with_probability(0.3);
///
/// // Fail every 5th call
/// let config = FailureConfig::new().trigger_every(5);
///
/// // Fail only when condition is true
/// let config = FailureConfig::new()
///     .with_probability(1.0)
///     .when(|| std::env::var("CHAOS_MODE").is_ok());
/// ```
pub struct FailureConfig {
    enabled_points: Vec<FailurePointId>,
    probability: u32,
    counter: AtomicU64,
    trigger_every: u64,
    on_check: Option<FailureCallback>,
    on_failure: Option<FailureCallback>,
    failures_triggered: AtomicU64,
    seed: u64,
    predicate: Option<FailurePredicate>,
    #[cfg(feature = "std")]
    latency_min_ns: u64,
    #[cfg(feature = "std")]
    latency_max_ns: u64,
    max_failures: u64,
    limited_failures: AtomicU64,
    #[cfg(feature = "std")]
    total_latency_ns: AtomicU64,
}

impl FailureConfig {
    /// Create a new failure configuration with no failures enabled.
    ///
    /// Use builder methods like `with_probability()` or `trigger_every()` to configure behavior.
    pub fn new() -> Self {
        Self {
            enabled_points: Vec::new(),
            probability: 0,
            counter: AtomicU64::new(0),
            trigger_every: 0,
            on_check: None,
            on_failure: None,
            failures_triggered: AtomicU64::new(0),
            seed: 0,
            predicate: None,
            #[cfg(feature = "std")]
            latency_min_ns: 0,
            #[cfg(feature = "std")]
            latency_max_ns: 0,
            max_failures: 0,
            limited_failures: AtomicU64::new(0),
            #[cfg(feature = "std")]
            total_latency_ns: AtomicU64::new(0),
        }
    }

    /// Chaos Monkey policy: 10% random failure rate.
    ///
    /// Simulates unpredictable failures for resilience testing.
    ///
    /// # Example
    /// ```
    /// let config = FailureConfig::chaos_monkey();
    /// ```
    pub fn chaos_monkey() -> Self {
        Self::new().with_probability(0.1)
    }

    /// Degraded Service policy: custom failure rate.
    ///
    /// Simulates a degraded system with specified failure probability.
    ///
    /// # Example
    /// ```
    /// // 30% of requests fail
    /// let config = FailureConfig::degraded_service(0.3);
    /// ```
    pub fn degraded_service(degradation: f64) -> Self {
        Self::new().with_probability(degradation)
    }

    /// Circuit Breaker policy: fail every nth call.
    ///
    /// Simulates a circuit breaker that fails periodically.
    ///
    /// # Example
    /// ```
    /// // Fail every 5th call
    /// let config = FailureConfig::circuit_breaker(5);
    /// ```
    pub fn circuit_breaker(failure_threshold: u64) -> Self {
        Self::new().trigger_every(failure_threshold)
    }

    /// Enable all failure points with 100% failure rate.
    ///
    /// Useful for testing that all failure points are correctly handled.
    pub fn enable_all() -> Self {
        Self {
            enabled_points: Vec::new(),
            probability: u32::MAX,
            counter: AtomicU64::new(0),
            trigger_every: 0,
            on_check: None,
            on_failure: None,
            failures_triggered: AtomicU64::new(0),
            seed: 0,
            predicate: None,
            #[cfg(feature = "std")]
            latency_min_ns: 0,
            #[cfg(feature = "std")]
            latency_max_ns: 0,
            max_failures: 0,
            limited_failures: AtomicU64::new(0),
            #[cfg(feature = "std")]
            total_latency_ns: AtomicU64::new(0),
        }
    }

    /// Enable failures for a specific failure point ID.
    ///
    /// When using `enable_point()`, only the specified points will fail.
    pub fn enable_point(mut self, id: FailurePointId) -> Self {
        self.enabled_points.push(id);
        self
    }

    /// Set probability of failure (0.0 to 1.0).
    ///
    /// Each failure point check will fail with this probability.
    ///
    /// # Example
    /// ```
    /// // 25% failure rate
    /// let config = FailureConfig::new().with_probability(0.25);
    /// ```
    pub fn with_probability(mut self, prob: f64) -> Self {
        self.probability = (prob * u32::MAX as f64) as u32;
        self
    }

    /// Fail every nth call deterministically.
    ///
    /// Creates a predictable failure pattern for testing scenarios.
    ///
    /// # Example
    /// ```
    /// // Fail on calls 0, 3, 6, 9, ...
    /// let config = FailureConfig::new().trigger_every(3);
    /// ```
    pub fn trigger_every(mut self, n: u64) -> Self {
        self.trigger_every = n;
        self
    }

    /// Set a seed for reproducible randomness.
    ///
    /// # Example
    /// ```
    /// // Same seed always produces same failure pattern
    /// let config = FailureConfig::new()
    ///     .with_probability(0.3)
    ///     .with_seed(12345);
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set seed from `FALLIBLES_SEED` environment variable.
    ///
    /// If the environment variable is not set or invalid, uses default (0).
    ///
    /// # Example
    /// ```bash
    /// FALLIBLES_SEED=12345 cargo test
    /// ```
    #[cfg(feature = "std")]
    pub fn with_seed_from_env(mut self) -> Self {
        if let Ok(seed_str) = std::env::var("FALLIBLES_SEED") {
            if let Ok(seed) = seed_str.parse::<u64>() {
                self.seed = seed;
            }
        }
        self
    }

    /// Set a predicate that must return true for failures to occur.
    ///
    /// Allows control over when failures are enabled based on runtime conditions.
    ///
    /// # Example
    /// ```
    /// // Only fail when chaos mode is enabled
    /// let config = FailureConfig::new()
    ///     .with_probability(0.5)
    ///     .when(|| std::env::var("CHAOS_MODE").is_ok());
    /// ```
    pub fn when<F>(mut self, predicate: F) -> Self
    where
        F: Fn() -> bool + Send + Sync + 'static,
    {
        self.predicate = Some(Box::new(predicate));
        self
    }

    /// Add artificial latency/delay to failure points.
    ///
    /// Adds a delay (in the specified range) every time a failure point is checked,
    /// even if no failure is triggered. Useful for simulating slow networks or I/O.
    ///
    /// # Example
    /// ```
    /// use std::time::Duration;
    ///
    /// // Add 10-50ms latency to all checks
    /// let config = FailureConfig::new()
    ///     .with_probability(0.2)
    ///     .with_latency(Duration::from_millis(10), Duration::from_millis(50));
    ///
    /// // Fixed latency
    /// let config = FailureConfig::new()
    ///     .with_latency(Duration::from_millis(100), Duration::from_millis(100));
    /// ```
    #[cfg(feature = "std")]
    pub fn with_latency(mut self, min: Duration, max: Duration) -> Self {
        self.latency_min_ns = min.as_nanos() as u64;
        self.latency_max_ns = max.as_nanos() as u64;
        self
    }

    /// Limit the total number of failures that can be triggered.
    ///
    /// Once this limit is reached, no more failures will occur even if
    /// probability or other conditions would trigger them.
    ///
    /// # Example
    /// ```
    /// // Allow at most 5 failures, then stop
    /// let config = FailureConfig::new()
    ///     .with_probability(0.5)
    ///     .max_failures(5);
    /// ```
    pub fn max_failures(mut self, max: u64) -> Self {
        self.max_failures = max;
        self
    }

    /// Register a callback that's called every time a failure point is checked.
    /// The callback receives information about the failure point being checked.
    ///
    /// # Example
    /// ```
    /// let config = FailureConfig::new()
    ///     .with_probability(0.3)
    ///     .on_check(|fp| println!("Checking: {}:{}", fp.file, fp.line));
    /// ```
    pub fn on_check<F>(mut self, callback: F) -> Self
    where
        F: Fn(FailurePoint) + Send + Sync + 'static,
    {
        self.on_check = Some(Box::new(callback));
        self
    }

    /// Register a callback that's called when a failure is actually triggered.
    ///
    /// Useful for logging, metrics, or coordinating failures across multiple points.
    ///
    /// # Example
    /// ```
    /// let config = FailureConfig::new()
    ///     .with_probability(0.3)
    ///     .on_failure(|fp| eprintln!("FAILURE at {}:{}", fp.file, fp.line));
    /// ```
    pub fn on_failure<F>(mut self, callback: F) -> Self
    where
        F: Fn(FailurePoint) + Send + Sync + 'static,
    {
        self.on_failure = Some(Box::new(callback));
        self
    }

    /// Get statistics about failure injection behavior.
    ///
    /// Returns total checks and total failures triggered.
    ///
    /// # Example
    /// ```
    /// let stats = config.stats();
    /// println!("Failure rate: {}/{}", stats.total_failures, stats.total_checks);
    /// ```
    pub fn stats(&self) -> FailureStats {
        FailureStats {
            total_checks: self.counter.load(Ordering::Relaxed),
            total_failures: self.failures_triggered.load(Ordering::Relaxed),
            limited_failures: self.limited_failures.load(Ordering::Relaxed),
            #[cfg(feature = "std")]
            total_latency_ns: self.total_latency_ns.load(Ordering::Relaxed),
        }
    }

    fn should_trigger(&self, fp_id: FailurePointId) -> bool {
        if let Some(predicate) = &self.predicate
            && !predicate() {
                return false;
            }

        if !self.enabled_points.is_empty() && !self.enabled_points.contains(&fp_id) {
            return false;
        }

        if self.trigger_every > 0 {
            let count = self.counter.fetch_add(1, Ordering::Relaxed);
            return count.is_multiple_of(self.trigger_every);
        }

        if self.probability > 0 {
            let counter = self.counter.fetch_add(1, Ordering::Relaxed);
            let mut bytes = [0u8; 12];
            bytes[0..4].copy_from_slice(&fp_id.0.to_le_bytes());
            bytes[4..12].copy_from_slice(&counter.to_le_bytes());

            let hash1 = fxhash::hash32(&bytes);
            let hash2 = fxhash::hash64(&bytes);

            let mut combined = (hash1 as u64) ^ hash2;

            if self.seed != 0 {
                combined ^= self.seed.wrapping_mul(0x517cc1b727220a95);
            } else {
                #[cfg(feature = "std")]
                {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    let nanos = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0);
                    let thread_id = std::thread::current().id();
                    let thread_hash = fxhash::hash64(&std::format!("{:?}", thread_id).as_bytes());
                    let stack_addr = &nanos as *const _ as usize as u64;
                    combined ^= nanos.wrapping_add(stack_addr).wrapping_mul(thread_hash);
                }
            }

            combined ^= combined >> 33;
            combined = combined.wrapping_mul(0xff51afd7ed558ccd);
            combined ^= combined >> 33;
            combined = combined.wrapping_mul(0xc4ceb9fe1a85ec53);
            combined ^= combined >> 33;

            let threshold = ((self.probability as u64) << 32) | self.probability as u64;
            return combined < threshold;
        }

        false
    }
}

impl Default for FailureConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Set a custom global failure handler.
///
/// You should use `configure_failures()` instead unless you need complete control
/// over failure behavior.
pub fn set_global_handler<H: FailureHandler + 'static>(handler: H) {
    let handler: Box<dyn FailureHandler> = Box::new(handler);
    let ptr = Box::into_raw(handler);

    let parts: [usize; 2] = unsafe { core::mem::transmute(ptr) };

    GLOBAL_HANDLER_DATA.store(parts[0], Ordering::SeqCst);
    GLOBAL_HANDLER_VTABLE.store(parts[1], Ordering::SeqCst);
}

/// Set global configuration.
///
/// This affects all `#[fallible]` functions in your program.
///
/// # Example
/// ```
/// use fallibles::fallibles_core::{configure_failures, FailureConfig};
///
/// // Enable 30% failure rate globally
/// configure_failures(FailureConfig::new().with_probability(0.3));
/// ```
pub fn configure_failures(config: FailureConfig) {
    let old_ptr = CONFIG_PTR.swap(Box::into_raw(Box::new(config)) as usize, Ordering::SeqCst);
    if old_ptr != 0 {
        unsafe {
            drop(Box::from_raw(old_ptr as *mut FailureConfig));
        }
    }
}

/// Clear global configuration.
///
/// After calling this, no failures will be injected unless a new config is set.
pub fn clear_failure_config() {
    let old_ptr = CONFIG_PTR.swap(0, Ordering::SeqCst);
    if old_ptr != 0 {
        unsafe {
            drop(Box::from_raw(old_ptr as *mut FailureConfig));
        }
    }
}

/// Set thread-local configuration.
///
/// This affects only the current thread, allowing independent failure injection
/// per thread in concurrent tests.
///
/// # Example
/// ```
/// use fallibles::fallibles_core::{configure_thread_failures, FailureConfig};
/// use std::thread;
///
/// thread::spawn(|| {
///     // This config only affects this thread
///     configure_thread_failures(FailureConfig::new().with_probability(0.5));
/// });
/// ```
#[cfg(feature = "std")]
pub fn configure_thread_failures(config: FailureConfig) {
    THREAD_CONFIG_PTR.with(|cell| {
        let old_ptr = cell.replace(Box::into_raw(Box::new(config)) as usize);
        if old_ptr != 0 {
            unsafe {
                drop(Box::from_raw(old_ptr as *mut FailureConfig));
            }
        }
    });
}

/// Clear thread-local configuration.
///
/// After calling this, the thread will fall back to the global configuration.
#[cfg(feature = "std")]
pub fn clear_thread_failure_config() {
    THREAD_CONFIG_PTR.with(|cell| {
        let old_ptr = cell.replace(0);
        if old_ptr != 0 {
            unsafe {
                drop(Box::from_raw(old_ptr as *mut FailureConfig));
            }
        }
    });
}

/// Automatically clears configuration when dropped.
///
/// Created by `with_config()` or `with_thread_config()`. Ensures cleanup
/// even if your code panics.
#[cfg(feature = "std")]
pub struct FailureConfigGuard {
    was_global: bool,
}

#[cfg(feature = "std")]
impl Drop for FailureConfigGuard {
    fn drop(&mut self) {
        if self.was_global {
            clear_failure_config();
        } else {
            clear_thread_failure_config();
        }
    }
}

/// Set global failure configuration with automatic cleanup.
///
/// Returns a guard that clears the configuration when dropped.
///
/// # Example
/// ```
/// use fallibles::fallibles_core::{with_config, FailureConfig};
///
/// {
///     let _guard = with_config(FailureConfig::new().with_probability(0.3));
///     // failures enabled here
/// } // config automatically cleared
/// ```
#[cfg(feature = "std")]
pub fn with_config(config: FailureConfig) -> FailureConfigGuard {
    configure_failures(config);
    FailureConfigGuard { was_global: true }
}

/// Set thread-local failure configuration with automatic cleanup.
///
/// Returns a guard that clears the configuration when dropped.
///
/// # Example
/// ```
/// use fallibles::fallibles_core::{with_thread_config, FailureConfig};
/// use std::thread;
///
/// let handle = thread::spawn(|| {
///     let _guard = with_thread_config(FailureConfig::new().with_probability(0.5));
///     // only this thread has failures enabled
/// });
/// ```
#[cfg(feature = "std")]
pub fn with_thread_config(config: FailureConfig) -> FailureConfigGuard {
    configure_thread_failures(config);
    FailureConfigGuard { was_global: false }
}

/// Check if a failure should be simulated at this point.
///
/// This is called internally by the `#[fallible]` macro.
#[inline(always)]
pub fn should_simulate_failure(fp: FailurePoint) -> bool {
    #[cfg(feature = "std")]
    {
        let thread_ptr = THREAD_CONFIG_PTR.with(|cell| *cell.borrow());
        if thread_ptr != 0 {
            return unsafe {
                let config = &*(thread_ptr as *const FailureConfig);
                check_and_trigger(config, fp)
            };
        }
    }

    let config_ptr = CONFIG_PTR.load(Ordering::Acquire);
    if config_ptr == 0 {
        return false;
    }

    unsafe {
        let config = &*(config_ptr as *const FailureConfig);
        check_and_trigger(config, fp)
    }
}

fn check_and_trigger(config: &FailureConfig, fp: FailurePoint) -> bool {
    // Inject latency if configured
    #[cfg(feature = "std")]
    if config.latency_max_ns > 0 {
        let latency_ns = if config.latency_min_ns == config.latency_max_ns {
            config.latency_min_ns
        } else {
            // Generate random latency in range [min, max]
            let counter = config.counter.load(Ordering::Relaxed);
            let mut bytes = [0u8; 12];
            bytes[0..4].copy_from_slice(&fp.id.0.to_le_bytes());
            bytes[4..12].copy_from_slice(&counter.to_le_bytes());
            let hash = fxhash::hash64(&bytes);
            
            let range = config.latency_max_ns - config.latency_min_ns;
            config.latency_min_ns + (hash % range)
        };
        
        if latency_ns > 0 {
            std::thread::sleep(Duration::from_nanos(latency_ns));
            config.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
        }
    }

    if let Some(on_check) = &config.on_check {
        on_check(fp);
    }

    let should_fail = config.should_trigger(fp.id);

    if should_fail {
        // Check if we've hit the failure limit
        if config.max_failures > 0 {
            let current_failures = config.failures_triggered.load(Ordering::Relaxed);
            if current_failures >= config.max_failures {
                // Record that we would have failed but were limited
                config.limited_failures.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        }

        config.failures_triggered.fetch_add(1, Ordering::Relaxed);
        if let Some(on_failure) = &config.on_failure {
            on_failure(fp);
        }
        return true;
    }

    false
}

/// Get statistics about the current configuration.
///
/// Returns `None` if no configuration is active.
/// Checks thread-local config first, then falls back to global config.
///
/// # Example
/// ```
/// use fallibles::fallibles_core::{configure_failures, get_failure_stats, FailureConfig};
///
/// configure_failures(FailureConfig::new().with_probability(0.3));
/// // ... run some tests ...
/// if let Some(stats) = get_failure_stats() {
///     println!("Failures: {}/{}", stats.total_failures, stats.total_checks);
/// }
/// ```
pub fn get_failure_stats() -> Option<FailureStats> {
    #[cfg(feature = "std")]
    {
        let thread_ptr = THREAD_CONFIG_PTR.with(|cell| *cell.borrow());
        if thread_ptr != 0 {
            return unsafe {
                let config = &*(thread_ptr as *const FailureConfig);
                Some(config.stats())
            };
        }
    }

    let config_ptr = CONFIG_PTR.load(Ordering::Acquire);
    if config_ptr == 0 {
        return None;
    }

    unsafe {
        let config = &*(config_ptr as *const FailureConfig);
        Some(config.stats())
    }
}
