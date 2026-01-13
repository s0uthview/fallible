//! # fallible
//!
//! Fallible enables controlled failure injection for testing error handling in Rust applications.
//! Mark functions with `#[fallible]` and configure when they should fail for comprehensive
//! testing.
//!
//! # Quick Start
//!
//! ```rust
//! use fallible::*;
//!
//! #[fallible]
//! fn database_query() -> Result<String, &'static str> {
//!     Ok("data".to_string())
//! }
//!
//! // Enable 30% failure rate
//! fallible_core::configure_failures(
//!     fallible_core::FailureConfig::new().with_probability(0.3)
//! );
//! ```
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust
//! use fallible::*;
//!
//! #[fallible]
//! fn read_config() -> Result<i32, &'static str> {
//!     Ok(42)
//! }
//!
//! // Without config, function always succeeds
//! assert_eq!(read_config().unwrap(), 42);
//!
//! // Enable failures with RAII guard
//! {
//!     let _guard = fallible_core::with_config(
//!         fallible_core::FailureConfig::new().with_probability(1.0)
//!     );
//!     // Now it will fail
//!     assert!(read_config().is_err());
//! } // Config automatically cleared
//! ```
//!
//! ## Inline Configuration
//!
//! ```rust
//! use fallible::*;
//!
//! #[fallible(probability = 0.2)]  // 20% failure rate
//! fn flaky_api() -> Result<String, &'static str> {
//!     Ok("response".to_string())
//! }
//!
//! #[fallible(trigger_every = 5)]  // Fail every 5th call
//! fn periodic_task() -> Result<(), String> {
//!     Ok(())
//! }
//! ```
//!
//! ## Policy-Based Testing
//!
//! ```rust
//! use fallible::fallible_core::{FailureConfig, with_config};
//!
//! // Chaos Monkey: 10% random failures
//! let _guard = with_config(FailureConfig::chaos_monkey());
//!
//! // Degraded Service: 30% failures
//! let _guard = with_config(FailureConfig::degraded_service(0.3));
//!
//! // Circuit Breaker: fail every 5th call
//! let _guard = with_config(FailureConfig::circuit_breaker(5));
//! ```
//!
//! ## Conditional Failures
//!
//! ```rust
//! use fallible::fallible_core::{FailureConfig, with_config};
//!
//! // Only fail when environment variable is set
//! let _guard = with_config(
//!     FailureConfig::new()
//!         .with_probability(0.5)
//!         .when(|| std::env::var("CHAOS_MODE").is_ok())
//! );
//! ```
//!
//! ## Reproducible Testing
//!
//! ```rust
//! use fallible::fallible_core::{FailureConfig, with_config};
//!
//! // Same seed always produces same failure pattern
//! let _guard = with_config(
//!     FailureConfig::new()
//!         .with_probability(0.3)
//!         .with_seed(12345)
//! );
//! // Or from environment: FALLIBLE_SEED=12345 cargo test
//! ```
//!
//! ## Custom Error Types
//!
//! ```rust
//! use fallible::*;
//!
//! #[derive(Debug, FallibleError)]
//! #[fallible(message = "timeout occurred")]
//! struct TimeoutError {
//!     message: String,
//! }
//!
//! #[fallible]
//! fn network_call() -> Result<String, TimeoutError> {
//!     Ok("data".to_string())
//! }
//! ```
//!
//! # Features
//!
//! - `fallible-sim` - Enable failure injection (required)
//! - `std` - Standard library support (default)
//! - `anyhow` - Support for anyhow::Error
//! - `eyre` - Support for eyre::Report

pub use fallible_core::*;
pub use fallible_macro::*;

pub extern crate fallible_core as fallible_core;
pub extern crate fxhash;