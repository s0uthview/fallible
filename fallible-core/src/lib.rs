#![no_std]

extern crate alloc;

use core::sync::atomic::{AtomicUsize, Ordering};
use alloc::boxed::Box;

pub trait FailureHandler {
    fn handle(&self, fp: FailurePoint) -> !;
}

#[derive(Copy, Clone, Debug)]
pub struct FailurePointId(pub u32);

#[derive(Copy, Clone, Debug)]
pub struct FailurePoint {
    pub id: FailurePointId,
}

pub struct PanicHandler;

impl FailureHandler for PanicHandler {
    fn handle(&self, fp: FailurePoint) -> ! {
        panic!("fallible simulated failure {:?}", fp);
    }
}

static GLOBAL_HANDLER_DATA: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_HANDLER_VTABLE: AtomicUsize = AtomicUsize::new(0);

pub fn set_global_handler<H: FailureHandler + 'static>(handler: H) {
    let handler: Box<dyn FailureHandler> = Box::new(handler);
    let ptr = Box::into_raw(handler);

    let parts: [usize; 2] = unsafe { core::mem::transmute(ptr) };

    GLOBAL_HANDLER_DATA.store(parts[0], Ordering::SeqCst);
    GLOBAL_HANDLER_VTABLE.store(parts[1], Ordering::SeqCst);
}

pub fn simulated_failure(fp: FailurePoint) -> ! {
    unsafe {
        let data = GLOBAL_HANDLER_DATA.load(Ordering::SeqCst);
        if data != 0 {
            let vtable = GLOBAL_HANDLER_VTABLE.load(Ordering::SeqCst);
            let parts = [data, vtable];
            let ptr: *const dyn FailureHandler = core::mem::transmute(parts);
            let handler = &*ptr;

            handler.handle(fp);
        }
    }

    PanicHandler.handle(fp)
}