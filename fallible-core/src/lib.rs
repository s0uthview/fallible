#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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
    fn simulated_failure() -> Self {
        
    }
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

pub trait FailureHandler {
    fn handle(&self, fp: FailurePoint) -> !;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FailurePointId(pub u32);

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

pub type FailureCallback = Box<dyn Fn(FailurePoint) + Send + Sync>;

pub struct FailureStats {
    pub total_checks: u64,
    pub total_failures: u64,
}

pub struct FailureConfig {
    enabled_points: Vec<FailurePointId>,
    probability: u32,
    counter: AtomicU64,
    trigger_every: u64,
    on_check: Option<FailureCallback>,
    on_failure: Option<FailureCallback>,
    failures_triggered: AtomicU64,
}

impl FailureConfig {
    pub fn new() -> Self {
        Self {
            enabled_points: Vec::new(),
            probability: 0,
            counter: AtomicU64::new(0),
            trigger_every: 0,
            on_check: None,
            on_failure: None,
            failures_triggered: AtomicU64::new(0),
        }
    }

    pub fn chaos_monkey() -> Self {
        Self::new().with_probability(0.1)
    }

    pub fn degraded_service(degradation: f64) -> Self {
        Self::new().with_probability(degradation)
    }

    pub fn circuit_breaker(failure_threshold: u64) -> Self {
        Self::new().trigger_every(failure_threshold)
    }

    pub fn enable_all() -> Self {
        Self {
            enabled_points: Vec::new(),
            probability: u32::MAX,
            counter: AtomicU64::new(0),
            trigger_every: 0,
            on_check: None,
            on_failure: None,
            failures_triggered: AtomicU64::new(0),
        }
    }

    pub fn enable_point(mut self, id: FailurePointId) -> Self {
        self.enabled_points.push(id);
        self
    }

    pub fn with_probability(mut self, prob: f64) -> Self {
        self.probability = (prob * u32::MAX as f64) as u32;
        self
    }

    pub fn trigger_every(mut self, n: u64) -> Self {
        self.trigger_every = n;
        self
    }

    pub fn on_check<F>(mut self, callback: F) -> Self
    where
        F: Fn(FailurePoint) + Send + Sync + 'static,
    {
        self.on_check = Some(Box::new(callback));
        self
    }

    pub fn on_failure<F>(mut self, callback: F) -> Self
    where
        F: Fn(FailurePoint) + Send + Sync + 'static,
    {
        self.on_failure = Some(Box::new(callback));
        self
    }

    pub fn stats(&self) -> FailureStats {
        FailureStats {
            total_checks: self.counter.load(Ordering::Relaxed),
            total_failures: self.failures_triggered.load(Ordering::Relaxed),
        }
    }

    fn should_trigger(&self, fp_id: FailurePointId) -> bool {
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

            combined ^= combined >> 33;
            combined = combined.wrapping_mul(0xff51afd7ed558ccd);
            combined ^= combined >> 33;
            combined = combined.wrapping_mul(0xc4ceb9fe1a85ec53);
            combined ^= combined >> 33;

            let final_hash = (combined >> 32) as u32;
            return final_hash < self.probability;
        }

        false
    }
}

impl Default for FailureConfig {
    fn default() -> Self {
        Self::new()
    }
}

pub fn set_global_handler<H: FailureHandler + 'static>(handler: H) {
    let handler: Box<dyn FailureHandler> = Box::new(handler);
    let ptr = Box::into_raw(handler);

    let parts: [usize; 2] = unsafe { core::mem::transmute(ptr) };

    GLOBAL_HANDLER_DATA.store(parts[0], Ordering::SeqCst);
    GLOBAL_HANDLER_VTABLE.store(parts[1], Ordering::SeqCst);
}

pub fn configure_failures(config: FailureConfig) {
    let old_ptr = CONFIG_PTR.swap(Box::into_raw(Box::new(config)) as usize, Ordering::SeqCst);
    if old_ptr != 0 {
        unsafe {
            drop(Box::from_raw(old_ptr as *mut FailureConfig));
        }
    }
}

pub fn clear_failure_config() {
    let old_ptr = CONFIG_PTR.swap(0, Ordering::SeqCst);
    if old_ptr != 0 {
        unsafe {
            drop(Box::from_raw(old_ptr as *mut FailureConfig));
        }
    }
}

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

#[cfg(feature = "std")]
pub fn with_config(config: FailureConfig) -> FailureConfigGuard {
    configure_failures(config);
    FailureConfigGuard { was_global: true }
}

#[cfg(feature = "std")]
pub fn with_thread_config(config: FailureConfig) -> FailureConfigGuard {
    configure_thread_failures(config);
    FailureConfigGuard { was_global: false }
}

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
    if let Some(on_check) = &config.on_check {
        on_check(fp);
    }

    let should_fail = config.should_trigger(fp.id);

    if should_fail {
        config.failures_triggered.fetch_add(1, Ordering::Relaxed);
        if let Some(on_failure) = &config.on_failure {
            on_failure(fp);
        }
    }

    should_fail
}

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
