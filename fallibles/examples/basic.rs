use fallibles::*;

/// simple function that could fail
#[fallible]
fn read_config() -> Result<i32, &'static str> {
    Ok(42)
}

/// api call with inline probability
#[fallible(probability = 0.3)]
fn fetch_data() -> Result<String, &'static str> {
    Ok("Hello, World!".to_string())
}

/// task that fails every 3rd call
#[fallible(trigger_every = 3)]
fn periodic_task() -> Result<(), String> {
    println!("Task executing...");
    Ok(())
}

fn main() {
    println!("fallible examples:\n");
    println!("1. without failure injection:");

    match read_config() {
        Ok(x) => println!("   read_config() = {}", x),
        Err(e) => println!("   read_config() failed: {}", e),
    }

    println!("\n2. 50% failure probability:");
    fallibles_core::configure_failures(
        fallibles_core::FailureConfig::new().with_probability(0.5),
    );

    for i in 0..10 {
        match read_config() {
            Ok(_) => print!("."),
            Err(_) => print!("X"),
        }
        if (i + 1) % 5 == 0 {
            print!(" ");
        }
    }

    println!();
    println!("\n3. using RAII guard with chaos monkey:");

    {
        let _guard = fallibles_core::with_config(fallibles_core::FailureConfig::chaos_monkey());
        for i in 0..20 {
            match read_config() {
                Ok(_) => print!("."),
                Err(_) => print!("X"),
            }
            if (i + 1) % 10 == 0 {
                print!(" ");
            }
        }
        println!();
    } // config gets cleared here

    println!("\n4. inline probability:");
    for i in 0..20 {
        match fetch_data() {
            Ok(_) => print!("."),
            Err(_) => print!("X"),
        }
        if (i + 1) % 10 == 0 {
            print!(" ");
        }
    }
    println!();

    println!("\n5. trigger every 3rd call:");
    for i in 0..10 {
        match periodic_task() {
            Ok(_) => println!("   Attempt {}: success", i),
            Err(_) => println!("   Attempt {}: FAILED", i),
        }
    }

    println!("\n6. seeded (seed = 99999):");
    {
        let _guard = fallibles_core::with_config(
            fallibles_core::FailureConfig::new()
                .with_probability(0.25)
                .with_seed(99999),
        );
        for i in 0..20 {
            match read_config() {
                Ok(_) => print!("."),
                Err(_) => print!("X"),
            }
            if (i + 1) % 10 == 0 {
                print!(" ");
            }
        }
        println!();
    }

    println!("\n7. conditional failures with predicate (counter > 5):");
    {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let _guard = fallibles_core::with_config(
            fallibles_core::FailureConfig::new()
                .with_probability(1.0)
                .when(move || counter_clone.load(Ordering::Relaxed) > 5),
        );

        for i in 0..10 {
            counter.fetch_add(1, Ordering::Relaxed);
            match read_config() {
                Ok(_) => println!("   Attempt {}: success", i),
                Err(_) => println!("   Attempt {}: FAILED", i),
            }
        }
    }

    println!("\n8. with callback for logging:");
    {
        let _guard = fallibles_core::with_config(
            fallibles_core::FailureConfig::new()
                .with_probability(0.5)
                .on_failure(|fp| {
                    eprintln!(
                        "   [FAILURE] {}:{} in {}",
                        fp.file, fp.line, fp.function
                    );
                }),
        );

        for i in 0..15 {
            print!("   Attempt {}: ", i);
            match read_config() {
                Ok(_) => println!("success"),
                Err(_) => println!("failed (callback triggered above)"),
            }
        }
    }

    println!("\n9. latency injection (10-50ms delay per check):");
    {
        use std::time::{Duration, Instant};

        let _guard = fallibles_core::with_config(
            fallibles_core::FailureConfig::new()
                .with_probability(0.3)
                .with_latency(Duration::from_millis(10), Duration::from_millis(50)),
        );

        let start = Instant::now();
        for i in 0..5 {
            match read_config() {
                Ok(_) => print!("."),
                Err(_) => print!("X"),
            }
        }
        let elapsed = start.elapsed();
        println!(" (took {:.0}ms for 5 checks)", elapsed.as_millis());
    }

    println!("\n10. failure limits (max 3 failures):");
    {
        let config = fallibles_core::FailureConfig::new()
            .with_probability(0.8)
            .max_failures(3);
        let _guard = fallibles_core::with_config(config);

        for i in 0..15 {
            match read_config() {
                Ok(_) => print!("."),
                Err(_) => print!("X"),
            }
        }
        println!();

        if let Some(stats) = fallibles_core::get_failure_stats() {
            println!("   {} failures triggered (max was 3)", stats.total_failures);
            if stats.limited_failures > 0 {
                println!("   {} additional failures were blocked by limit", stats.limited_failures);
            }
        }
    }

    println!("\n11. combined: latency + limits + metrics:");
    {
        use std::time::Duration;

        let config = fallibles_core::FailureConfig::new()
            .with_probability(0.4)
            .with_latency(Duration::from_millis(5), Duration::from_millis(15))
            .max_failures(5)
            .on_failure(|fp| {
                println!("      [FAIL] {} at line {}", fp.function, fp.line);
            });

        let _guard = fallibles_core::with_config(config);

        println!("   Running 20 checks:");
        for _ in 0..20 {
            let _ = read_config();
        }

        println!("\n   Statistics:");
        if let Some(stats) = fallibles_core::get_failure_stats() {
            stats.report();
        }
    }

    println!("\ncomplete!");
}
