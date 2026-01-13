use fallible::*;

#[fallible]
fn read_config() -> Result<i32, &'static str> {
    Ok(42)
}

#[fallible]
fn fetch_data() -> Result<&'static str, &'static str> {
    Ok("Hello, Fallible!")
}

fn main() {
    println!("Without failure injection:");
    match read_config() {
        Ok(x) => println!("read_config succeeded: {x}"),
        Err(_) => println!("read_config failed!"),
    }
    match fetch_data() {
        Ok(msg) => println!("fetch_data succeeded: {msg}"),
        Err(_) => println!("fetch_data failed!"),
    }

    println!("\nWith 50% probability:");
    fallible_core::configure_failures(
        fallible_core::FailureConfig::new()
            .with_probability(0.5)
    );

    for i in 0..10 {
        match read_config() {
            Ok(x) => println!("Attempt {}: read_config succeeded: {x}", i),
            Err(_) => println!("Attempt {}: read_config failed!", i),
        }
    }

    println!("\nWith trigger_every(3):");
    fallible_core::configure_failures(
        fallible_core::FailureConfig::new()
            .trigger_every(3)
    );

    for i in 0..10 {
        match fetch_data() {
            Ok(msg) => println!("Attempt {}: fetch_data succeeded: {msg}", i),
            Err(_) => println!("Attempt {}: fetch_data failed!", i),
        }
    }

    fallible_core::clear_failure_config();
}