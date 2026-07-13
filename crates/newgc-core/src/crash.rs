//! Windows SEH crash handler for GC diagnostics.
//!
//! Modeled on NewOpenDylan's `nod-runtime/src/crash_dump.rs`. For a
//! **moving** collector an access violation (`0xC0000005`) almost always
//! means a dangling / stale / mis-forwarded pointer was dereferenced —
//! the single hardest class of GC bug to localize, because the default
//! crash is an opaque exit code with no location. This installs an
//! unhandled-exception filter that, on the faulting thread, reports:
//!
//!   - the exception kind + code,
//!   - the faulting **instruction** address (the code that dereferenced),
//!   - for an access violation, whether it was a read/write and the
//!     **data address** that was inaccessible (`ExceptionInformation[1]`)
//!     — for a GC, comparing this against the heap reservation tells you
//!     instantly whether it was a wild pointer, a freed page, or null,
//!   - a symbolized backtrace.
//!
//! # Not production-signal-safe — and that's fine here
//!
//! Capturing a symbolized backtrace allocates and locks dbghelp, which a
//! strictly signal-safe handler must not do. This is a *diagnostic* aid:
//! a data access violation leaves the heap allocator and dbghelp intact,
//! so the capture works in practice. (Dylan's production variant writes a
//! pre-rendered, no-alloc GC-state dump instead; the role is the same —
//! emit actionable state before the process dies.) Call [`install`] from
//! a test or tool that wants the diagnosis; do not rely on it as a
//! recovery mechanism.

use std::sync::OnceLock;

/// Install the crash handler (idempotent across the process). Registers a
/// panic hook that prints a backtrace and, on Windows, an
/// unhandled-exception filter that decodes structured exceptions. Safe to
/// call from multiple tests; only the first call takes effect.
pub fn install() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        install_panic_hook();
        #[cfg(windows)]
        unsafe {
            install_seh_filter();
        }
    });
}

fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("\n=== newgc-core panic ===\n{info}");
        eprintln!("backtrace:\n{}", std::backtrace::Backtrace::force_capture());
        prev(info);
    }));
}

// ── Windows SEH filter ────────────────────────────────────────────────

/// Minimal `EXCEPTION_RECORD` matching the Win32 x86-64 ABI. Unlike the
/// Dylan version, we keep `NumberParameters` + `ExceptionInformation` so
/// we can read the inaccessible data address of an access violation.
#[cfg(windows)]
#[repr(C)]
struct ExceptionRecord {
    exception_code: u32,
    exception_flags: u32,
    exception_record_chain: *mut ExceptionRecord,
    exception_address: *mut core::ffi::c_void,
    number_parameters: u32,
    _pad: u32, // align ExceptionInformation to 8 bytes on x64
    exception_information: [usize; 15],
}

#[cfg(windows)]
#[repr(C)]
struct ExceptionPointers {
    exception_record: *mut ExceptionRecord,
    context_record: *mut core::ffi::c_void,
}

#[cfg(windows)]
unsafe fn install_seh_filter() {
    // `-unwind` ABI: an exception/panic crossing either direction across
    // this boundary unwinds per the platform ABI instead of being turned
    // into an immediate `abort` at a plain `extern "system"` edge.
    unsafe extern "system-unwind" {
        fn SetUnhandledExceptionFilter(
            handler: *const core::ffi::c_void,
        ) -> *mut core::ffi::c_void;
    }
    unsafe {
        SetUnhandledExceptionFilter(
            seh_filter
                as unsafe extern "system-unwind" fn(*mut ExceptionPointers) -> i32
                as *const core::ffi::c_void,
        );
    }
}

#[cfg(windows)]
unsafe extern "system-unwind" fn seh_filter(info: *mut ExceptionPointers) -> i32 {
    use std::io::Write as _;

    if info.is_null() {
        return 0; // EXCEPTION_CONTINUE_SEARCH
    }
    let rec = unsafe { (*info).exception_record };
    if rec.is_null() {
        return 0;
    }
    let code = unsafe { (*rec).exception_code };
    let mut out = String::with_capacity(512);
    use std::fmt::Write as _;
    let _ = writeln!(out, "\n=== newgc-core SEH crash ===");
    let _ = writeln!(
        out,
        "  exception : {} (code {code:#010x})",
        exception_code_name(code)
    );
    let _ = writeln!(
        out,
        "  faulting  : instruction at {:p}",
        unsafe { (*rec).exception_address }
    );
    if code == 0xC000_0005 || code == 0xC000_0006 {
        let np = unsafe { (*rec).number_parameters } as usize;
        if np >= 2 {
            let info0 = unsafe { (*rec).exception_information[0] };
            let addr = unsafe { (*rec).exception_information[1] };
            let kind = match info0 {
                0 => "read",
                1 => "write",
                8 => "execute (DEP)",
                _ => "?",
            };
            let _ = writeln!(out, "  access    : {kind} of {addr:#018x}");
        }
    }
    eprint!("{out}");
    eprintln!(
        "backtrace (faulting thread):\n{}",
        std::backtrace::Backtrace::force_capture()
    );
    let _ = std::io::stderr().flush();

    // EXCEPTION_EXECUTE_HANDLER (1): we have reported; terminate the
    // process rather than returning EXCEPTION_CONTINUE_SEARCH (0), which
    // would pop a Windows Error Reporting dialog and hang an unattended
    // test run.
    1
}

#[cfg(windows)]
fn exception_code_name(code: u32) -> &'static str {
    match code {
        0xC000_0005 => "EXCEPTION_ACCESS_VIOLATION",
        0xC000_0006 => "EXCEPTION_IN_PAGE_ERROR",
        0x8000_0003 => "EXCEPTION_BREAKPOINT",
        0xC000_001D => "EXCEPTION_ILLEGAL_INSTRUCTION",
        0xC000_0025 => "EXCEPTION_NONCONTINUABLE_EXCEPTION",
        0xC000_0094 => "EXCEPTION_INT_DIVIDE_BY_ZERO",
        0xC000_0095 => "EXCEPTION_INT_OVERFLOW",
        0xC000_00FD => "EXCEPTION_STACK_OVERFLOW",
        _ => "EXCEPTION_UNKNOWN",
    }
}
