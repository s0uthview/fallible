use fallible::*;

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
    fallible_core::configure_failures(
        fallible_core::FailureConfig::new().with_probability(0.5),
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
        let _guard = fallible_core::with_config(fallible_core::FailureConfig::chaos_monkey());
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
        let _guard = fallible_core::with_config(
            fallible_core::FailureConfig::new()
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

        let _guard = fallible_core::with_config(
            fallible_core::FailureConfig::new()
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
        let _guard = fallible_core::with_config(
            fallible_core::FailureConfig::new()
                .with_probability(0.3)
                .on_failure(|fp| {
                    eprintln!(
                        "   [FAILURE] {}:{} in {}",
                        fp.file, fp.line, fp.function
                    );
                }),
        );

        for _ in 0..5 {
            let _ = read_config();
        }
    }

    println!("\nfinished");
}
