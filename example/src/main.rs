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
    match read_config() {
        Ok(x) => println!("read_config succeeded: {x}"),
        Err(_) => println!("read_config failed!"),
    }

    match fetch_data() {
        Ok(msg) => println!("fetch_data succeeded: {msg}"),
        Err(_) => println!("fetch_data failed!"),
    }
}