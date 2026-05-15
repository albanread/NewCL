# NewCormanLisp Windows FFI — Design

*Status: design. No implementation yet. Last updated 2026-05-15.*

## What this is

A plan for letting NCL Lisp code call arbitrary Windows API functions
directly — `CreateWindowExW`, `MessageBoxW`, `GetSystemMetrics`,
everything that lives in `USER32.dll`/`GDI32.dll`/`KERNEL32.dll`/etc.
— without going through hand-written Rust shims.

The current approach is "Rust does the Win32 call, exports a Lisp
shim, Lisp calls the shim." That works for iGui because iGui needs
a handful of carefully-managed calls, but it doesn't scale: there
are 18,000+ Win32 entry points. We need a generic mechanism.

## Pre-existing assets we lean on

### `E:\windows_api\`

A pre-built normalized SQLite database of the entire Win32 API
surface, generated from Microsoft's official `Windows.Win32.winmd`
metadata package. Built by a sister project (NewM2 — a Modula-2
compiler reusing the same metadata). Workflow:

```
NuGet → winmd → C# importer (winmd_inspect/Program.cs)
              → SQLite (windows_api.db, 29 MB)
```

What `windows_api.db` currently holds:

| Table | Rows | Notes |
|---|---:|---|
| `namespaces` | 329 | `Windows.Win32.UI.WindowsAndMessaging`, … |
| `dlls` | 0 | (not populated; `dll_name` lives on `functions`) |
| `types` | 37,830 | primitives + structs + enums + handles + interfaces |
| `functions` | 18,271 | one row per Win32 export |
| `function_params` | 60,780 | with direction, optional, buffer_len links |

Top DLLs by function count: KERNEL32 (1,403), USER32 (768),
gdiplus (629), ADVAPI32 (619), GDI32 (431), OLEAUT32 (405),
SHLWAPI (360), OPENGL32 (355), SETUPAPI (334), WININET (296),
OLE32 (273), …

Critical properties of the schema:

- **Primitives are pre-resolved.** Typedefs like `DWORD`/`HANDLE`/
  `LPCWSTR` are already collapsed to Rust-style canonical names:
  `i8 i16 i32 i64 u8 u16 u32 u64 isize usize f32 f64 bool char void`.
  We don't maintain our own typedef chain.

- **ANSI/Unicode pairs flagged.** `aw_family` is `A`, `W`, or NULL.
  Modern code only needs `W`. Generator skips `A` variants.

- **`set_last_error` recorded.** ~13K functions need a `GetLastError()`
  follow-up after the call. The FFI kernel does that automatically.

- **Calling convention recorded.** `winapi` (16,897 functions) or
  `cdecl` (1,374). On x64 these are the same ABI; on x86 they'd
  differ. We're x64-only for now.

- **Out-params flagged.** `direction='out'` lets the generator emit
  `(multiple-value-bind (result success?) …)` rather than forcing
  the user to allocate temps.

**M2-specific columns can be ignored.** The `m2_*` columns and the
`windows_m2_type_map` table are scaffolding for the sister Modula-2
compiler. We don't reuse the M2 generator — we add a sibling output
mode.

### iGui's UI-thread plumbing

iGui already solves the "marshal work to a UI thread" problem the
same way every Win32 app does: a UI thread runs `GetMessage` /
`DispatchMessage`; background threads `SendMessageW` private
`WM_USER+N` messages with stack-allocated request structs in
`lparam`. Today there are eight verbs (`WM_IGUI_OPEN_CHILD`,
`WM_IGUI_CLOSE_CHILD`, …) handled by arms in `frame_wnd_proc`.

We generalize that pattern: one private message that carries
**any** request, dispatched by type tag. Specifically:

- `WM_NCL_EXECUTE` — carries a Lisp closure to `funcall`
- `WM_NCL_FFI_CALL` — carries an `FfiCallRequest` for direct FFI

## The runtime invariant

When `--windows` is enabled:

```
Thread 0 (process main)  =  generic UI dispatcher, runs message pump
Worker thread            =  Lisp evaluation (REPL, script, GC mutators)
iGui UI thread           =  iGui's frame + pump (only if (igui-start) called)
```

Without `--windows`:

```
Thread 0                 =  Lisp evaluation (today's behavior)
iGui UI thread           =  iGui's frame + pump (only if (igui-start) called)
```

iGui works in both modes. It manages its own UI thread regardless
of `--windows`. The two UI surfaces are siblings, not in conflict:
iGui windows live on iGui's UI thread; generic Win32 windows live
on thread 0. No HWND ownership overlap.

## The `--windows` flag

Driver:

```rust
fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let want_windows = args.iter().any(|a| a == "--windows" || a == "-W");

    if want_windows {
        run_with_windows_surface(args)
    } else {
        run_without_windows_surface(args)   // exactly what main() does today
    }
}
```

What `--windows` flips:

| Without `--windows` | With `--windows` |
|---|---|
| Lisp runs on thread 0 | Lisp runs on a worker thread |
| No message pump | Thread 0 runs `GetMessage`/`DispatchMessage` |
| No hidden HWND | Hidden `HWND_MESSAGE` dispatcher registered before any Lisp runs |
| `%ffi-call` shim not installed | `%ffi-call` shim installed |
| `(windows-enabled-p)` → NIL | `(windows-enabled-p)` → T |
| `win32-threading.lisp` not auto-loaded | Auto-required as part of `init.lisp` |
| `(on-ui-thread BODY)` errors | Routes via `WM_NCL_EXECUTE` |
| Win32 binding modules not loadable | Available via per-namespace `(require …)` |

A `ncl script.lisp` invocation that does pure compute is byte-for-byte
identical to today. The Windows machinery is paid for **only** if asked.

## Architecture in three layers

### Layer 1: FFI kernel (Rust)

One new shim:

```
(%ffi-call DLL-NAME FN-NAME ARG-TYPES RETURN-TYPE ARGS... &key (route :auto))
```

Routing keyword:
- `:auto` — generator-chosen default based on DLL (USER32/GDI32/COMCTL32 → `:ui`, KERNEL32/ADVAPI32 → `:any`, OLE32 per-function)
- `:ui` — force UI-thread dispatch (errors without `--windows`)
- `:any` — call directly on the invoking thread

Implementation:

1. `LoadLibraryW(DLL-NAME)` with a global cache (one `HMODULE` per DLL,
   lifetime = process)
2. `GetProcAddress` with a per-(DLL, FN) cache
3. Walk `ARG-TYPES` (list of keywords: `:i32`, `:u32`, `:i64`, `:f64`,
   `:wstr`, `:handle`, `:ptr`, …), unbox each Lisp `Word` into the
   matching native register/stack slot
4. Call the function pointer via **libffi** (vendored via
   `libffi-sys` crate, MIT licensed). On x64 Windows, all 18K
   functions share one calling convention so a single dispatcher
   works.
5. Box the return value back to a `Word` per `RETURN-TYPE`
6. If `set_last_error`: call `GetLastError()` immediately, stash in a
   thread-local; expose as `(win32-last-error)`

**Why libffi rather than hand-rolled asm**: maintainability. About
200 lines of glue vs. 50 lines of inline asm × every calling
convention × every arity we want to support. Already battle-tested.

### Layer 2: UI-thread routing (Rust + Lisp)

#### Hidden dispatch HWND

Created in `main()` immediately when `--windows` is set, **before**
the worker is spawned (so no race):

```rust
// src/ncl-runtime/src/ui_dispatch.rs (new module)
pub(crate) const WM_NCL_EXECUTE: u32  = WM_USER + 99;
pub(crate) const WM_NCL_FFI_CALL: u32 = WM_USER + 100;

#[repr(C)]
pub struct ExecuteRequest {
    pub closure_word: u64,     // Lisp closure to call
    pub out_result:   u64,     // [out] return value as Word
    pub out_error:    u64,     // [out] condition Word if call panicked
}

#[repr(C)]
pub struct FfiCallRequest {
    pub dll_name:       *const u16,   // UTF-16, NUL-terminated
    pub fn_name:        *const u8,    // ASCII, NUL-terminated
    pub arg_type_tags:  *const u8,    // one byte per arg
    pub arg_words:      *const u64,   // packed Lisp Words
    pub n_args:         u32,
    pub return_tag:     u8,
    pub set_last_error: u8,
    pub out_result:     u64,
    pub out_last_error: u32,
    pub out_error_word: u64,
}

static UI_DISPATCH: OnceLock<UiDispatch> = OnceLock::new();
struct UiDispatch { hwnd: HwndPtr, thread_id: u32 }

unsafe extern "system" fn dispatch_wnd_proc(
    hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_NCL_EXECUTE => {
            let req = &mut *(lparam.0 as *mut ExecuteRequest);
            run_closure_on_ui_thread(req);
            LRESULT(0)
        }
        WM_NCL_FFI_CALL => {
            let req = &mut *(lparam.0 as *mut FfiCallRequest);
            perform_ffi_call(req);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}
```

`SendMessageW` is synchronous — the worker blocks while thread 0
runs the handler. That's the RPC mechanism. No condvars, no
channels, just the Win32 message pump doing its job.

#### Driver restructure

```rust
fn run_with_windows_surface(args: Vec<String>) -> ExitCode {
    // Thread 0:
    let dispatch_hwnd = create_message_only_window();
    UI_DISPATCH.set(UiDispatch {
        hwnd: dispatch_hwnd,
        thread_id: unsafe { GetCurrentThreadId() },
    }).expect("UI_DISPATCH already set");

    // Spawn the worker — it does what main() does today.
    let (tx, rx) = mpsc::sync_channel(1);
    let worker = thread::Builder::new()
        .name("ncl-lisp-worker".into())
        .spawn(move || {
            let code = lisp_worker_main(args);
            tx.send(code).unwrap();
            unsafe {
                PostThreadMessageW(
                    UI_DISPATCH.get().unwrap().thread_id,
                    WM_QUIT, WPARAM(0), LPARAM(0)
                );
            }
        })
        .expect("spawn worker");

    // Thread 0 takes over the pump.
    let mut msg = MSG::default();
    unsafe {
        while GetMessageW(&mut msg, None, 0, 0).0 > 0 {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    let _ = worker.join();
    rx.try_recv().unwrap_or(ExitCode::from(1))
}
```

#### Mutator registration

The UI thread needs to be able to `funcall` a Lisp closure for
`WM_NCL_EXECUTE`. That means it has to be registered with the
`GcCoordinator` as a mutator. NCL already supports N>1 mutators
(iGui's "language thread" exercises that today). One line in `main()`
after creating the coordinator:

```rust
let _ui_mutator = coord.register_mutator();  // stashed in OnceLock
```

#### Lisp surface

`Lisp/Library/win32-threading.lisp` (auto-required by `init.lisp`
when `(windows-enabled-p)`):

```lisp
(provide 'win32-threading)

;; Predicates — work even without --windows; they tell you the truth.
(defun ui-thread-p () (%ui-thread-p))     ; T iff this is thread 0
(defun ui-thread-id () (%ui-thread-id))   ; OS thread ID, or NIL

;; Synchronous dispatch. Errors if (not (windows-enabled-p)).
(defmacro on-ui-thread (&rest body)
  `(if (ui-thread-p)
       (progn ,@body)
       (%ui-execute (lambda () ,@body))))

;; Fire-and-forget (PostMessage; returns immediately).
(defmacro post-to-ui-thread (&rest body)
  `(%ui-post (lambda () ,@body)))
```

`%ui-execute` and `%ui-post` are Rust shims. `%ui-execute` builds an
`ExecuteRequest` on the stack, `SendMessageW`s the dispatch HWND with
`WM_NCL_EXECUTE`, reads back the result. `%ui-post` allocates the
request on the heap, `PostMessageW`s, returns immediately — the
handler frees the request after running.

#### Re-entrancy

Inside `on-ui-thread`, every nested call sees `(ui-thread-p) → T`
and skips the marshalling. Batching is free:

```lisp
(on-ui-thread
  (let ((hwnd (create-window-ex-w 0 "STATIC" "Hi" +ws-popup+
                                  100 100 200 100
                                  +null-hwnd+ +null-hwnd+
                                  +null-handle+ +null+)))
    (show-window hwnd +sw-show+)
    (update-window hwnd)
    hwnd))
```

One thread hop, three Win32 calls.

#### The deadlock to document

**Don't** make a blocking SendMessage from inside `on-ui-thread`
that targets a thread currently waiting on the UI thread. Standard
SendMessage circular-wait. Same constraint as any cross-thread RPC.

`SendMessage` to your own thread short-circuits — so a re-entrant
`(on-ui-thread …)` inside an `on-ui-thread` body is safe by
construction. For genuinely concurrent cross-thread choreography,
use `post-to-ui-thread` (fire-and-forget; no blocking).

### Layer 3: Generator + bindings (Python + Lisp)

#### Extend `bootstrap.py`

Add a `generate-ncl-bindings` subcommand:

```
python windows_api/bootstrap.py generate-ncl-bindings \
    --namespace Windows.Win32.UI.WindowsAndMessaging \
    --output Lisp/Library/win32/windowsandmessaging.lisp
```

Queries `windows_api.db` for the namespace's functions. Emits one
file per namespace with:
- `(provide 'win32-windowsandmessaging)` at the top
- One `(defun …)` per non-deprecated `W`-family function
- Constant definitions for enum values (`+mb-ok+`, `+mb-iconwarning+`, etc.)

Example output:

```lisp
;;;; Generated from windows_api.db — Windows.Win32.UI.WindowsAndMessaging
;;;; DO NOT EDIT BY HAND. Regenerate via
;;;;     python windows_api/bootstrap.py generate-ncl-bindings ...

(provide 'win32-windowsandmessaging)

;; MESSAGEBOX_STYLE constants
(defconstant +mb-ok+              #x00000000)
(defconstant +mb-okcancel+        #x00000001)
(defconstant +mb-iconwarning+     #x00000030)
;; ... 60 more ...

;; USER32.dll default route: :ui
(defun message-box-w (hwnd text caption u-type)
  (%ffi-call "USER32.dll" "MessageBoxW"
             '(:handle :wstr :wstr :u32) :i32
             hwnd text caption u-type
             :route :ui))

(defun get-system-metrics (n-index)
  (%ffi-call "USER32.dll" "GetSystemMetrics"
             '(:i32) :i32 n-index
             :route :ui))

;; ... ~770 more for USER32 alone ...
```

#### Route policy (per DLL default)

| DLL | Default route | Rationale |
|---|---|---|
| `USER32.dll` | `:ui` | Window/message functions touch HWND state |
| `GDI32.dll` | `:ui` | Drawing into a DC owned by the UI thread |
| `COMCTL32.dll` | `:ui` | Common controls — same constraint |
| `KERNEL32.dll` | `:any` | Process/threading/file/memory — thread-agnostic |
| `ADVAPI32.dll` | `:any` | Registry/crypto — thread-agnostic |
| `SHLWAPI.dll` | `:any` | String utilities — thread-agnostic |
| `OLE32.dll` | per function | STA-only → `:ui`; MTA-safe → `:any` |
| `D2D1.dll`, `DWrite.dll` | per function | Device-context affinity rules |

User can override at call site:

```lisp
;; Force this MessageBoxW to bypass UI-thread routing
;; (it will block the caller thread, which is fine for an early
;; bootstrap "no UI yet" diagnostic before the worker is up):
(message-box-w +null-hwnd+ "boot fail" "ncl" +mb-iconerror+ :route :any)
```

#### Type tag map

The byte tags packed into `arg_type_tags` and `return_tag`:

| Tag | Lisp keyword | C type | Word marshalling |
|---|---|---|---|
| 0 | `:void` | `void` | only valid as return; → NIL |
| 1 | `:i8` / `:u8` | `int8_t` / `uint8_t` | fixnum |
| 2 | `:i16` / `:u16` | `int16_t` / `uint16_t` | fixnum |
| 3 | `:i32` / `:u32` | `int32_t` / `uint32_t` | fixnum |
| 4 | `:i64` / `:u64` | `int64_t` / `uint64_t` | fixnum or bignum |
| 5 | `:isize` / `:usize` | pointer-sized int | fixnum (on x64) |
| 6 | `:f32` / `:f64` | `float` / `double` | float Word |
| 7 | `:bool` | C `BOOL` (32-bit int) | T/NIL ↔ 1/0 |
| 8 | `:handle` | opaque pointer-sized handle | fixnum |
| 9 | `:ptr` | raw `void*` | fixnum |
| 10 | `:wstr` | `LPCWSTR` (UTF-16) | NCL string → UTF-16 buffer |
| 11 | `:cstr` | `LPCSTR` (ANSI) | NCL string → CP_ACP buffer |

Signed-vs-unsigned at width N share a tag — the keyword tells Lisp
how to interpret; the byte layout is identical at the ABI level.

## File layout

```
src/
  ncl-driver/src/main.rs                  modified — --windows branch
  ncl-runtime/src/ui_dispatch.rs          new — hidden HWND + WndProc + shims
  ncl-runtime/src/ffi.rs                  new — %ffi-call kernel via libffi
  ncl-compiler/src/lib.rs                 modified — install_native_functions
                                                     for %ffi-call etc., gated

Lisp/Library/
  init.lisp                               modified — (when (windows-enabled-p)
                                                       (require 'win32-threading))
  win32-threading.lisp                    new — (on-ui-thread …), (ui-thread-p)
  win32-utils.lisp                        new — (with-wstring …), defstruct-win32
  win32/                                  new dir — generated bindings:
    foundation.lisp                       Windows.Win32.Foundation
    user32.lisp                           Windows.Win32.UI.WindowsAndMessaging
    gdi32.lisp                            Windows.Win32.Graphics.Gdi
    kernel32.lisp                         Windows.Win32.System.* (subset)
    ...

windows_api/                              EXTERNAL — already exists
  windows_api.db                          37,830 types, 18,271 functions
  bootstrap.py                            extended with generate-ncl-bindings
  schema.sql                              v5
  winmd_inspect/Program.cs                C# WinMD importer (read-only for us)

tests/ncl-tests/
  src/win_helper.rs                       new — with_windows_session() helper
  tests/ffi_smoke.rs                      new — Beep, MessageBoxW, etc.
  tests/ui_thread.rs                      new — (on-ui-thread …) round-trip
```

## Test strategy

Most tests stay unchanged — they don't need `--windows`. They build
a `Session` and run on the single test thread.

For tests that touch the UI surface, a helper:

```rust
// tests/ncl-tests/src/win_helper.rs
//
// Spin up a thread-0-style dispatcher + a worker, run `test` on the
// worker, return its result. Only used by the handful of tests that
// genuinely need (on-ui-thread …) or :route :ui FFI.
pub fn with_windows_session<F, R>(test: F) -> R
where F: FnOnce(&mut Session) -> R + Send + 'static,
      R: Send + 'static
{
    let dispatch_hwnd = create_message_only_window();
    register_ui_dispatch(dispatch_hwnd, current_thread_id());

    let (tx, rx) = mpsc::sync_channel(1);
    let _worker = thread::spawn(move || {
        let mut s = Session::with_stdlib_and_windows().unwrap();
        tx.send(test(&mut s)).unwrap();
        unsafe {
            PostThreadMessageW(ui_thread_id(), WM_QUIT, WPARAM(0), LPARAM(0));
        }
    });

    run_dispatch_pump();
    rx.recv().unwrap()
}
```

Tests opt in:

```rust
#[test]
fn message_box_routes_to_ui_thread() {
    with_windows_session(|s| {
        // Worker thread runs this.
        let r = s.eval(r#"
            (on-ui-thread
              (list (ui-thread-p) (ui-thread-id)))"#).unwrap();
        // Expect (T <some-tid>).
        assert!(r.starts_with("(T "));
    });
}
```

## Things deferred to later slices

- **Structs / records** — `defstruct-win32 RECT (left :i32) (top :i32) …`
  with pack/unpack marshalling. Phase 3.
- **Out-params** — `(multiple-value-bind (result success?) …)` idiom
  for Win32 BOOL+out-pointer pattern. Phase 3.
- **Callbacks** — Lisp-side function passed to Win32 (e.g. `WindowProc`
  for `RegisterClassExW`, `EnumProc` callbacks). Needs a JIT-emitted
  trampoline per signature. Phase 4.
- **COM** — IUnknown vtable dispatch is a different model from flat
  exports. Worth a dedicated design pass. Phase 6.
- **Eat dog food** — replace iGui's hand-coded `WM_IGUI_*` verbs with
  FFI calls into generated bindings. Phase 5.

## Open question parked

Snapshot strategy for the generated bindings:

**(a) Commit generated `.lisp` files into NCL's tree.** Regenerate on
demand. Users see them in `git diff`. Doesn't require Python/C# at
build time. ← *preferred*

**(b) Vendor `windows_api.db` directly.** Users can regenerate
without the upstream toolchain. Db file is 29 MB, not ideal for git.

Going with (a) unless someone has a reason to prefer (b).

## Implementation sequencing

| Phase | LOC est. | Deliverable |
|---|---:|---|
| 1 | ~200 Rust | `--windows` flag + driver split + `(windows-enabled-p)`. Worker runs Lisp; thread 0 runs an (empty) pump. Test: `ncl --windows -e '(format t "~A~%" (windows-enabled-p))'` → `T`. |
| 2 | ~250 Rust + 60 Lisp | Hidden dispatch HWND + `WM_NCL_EXECUTE` + `%ui-execute` + `(on-ui-thread …)` + UI-thread mutator. Test: `(on-ui-thread (ui-thread-p))` → `T` from a worker. |
| 3 | ~300 Rust | libffi vendor + `%ffi-call` kernel + `WM_NCL_FFI_CALL` arm + 5 hand-coded smoke tests (`Beep`, `Sleep`, `GetTickCount64`, `MessageBoxW`, `GetSystemMetrics`). |
| 4 | ~400 Python + ~2 MB generated Lisp | Bootstrap generator + bindings for Console + WindowsAndMessaging + Foundation. Test: generated `(message-box-w …)` actually pops up a dialog. |
| 5 | ~200 Lisp | `(with-wstring …)`, `defstruct-win32`, out-param idiom. Phase-3 ergonomics. |
| 6 | later | Callbacks via JIT trampolines. |
| 7 | later | Replace iGui's `WM_IGUI_*` verbs with FFI. |
| 8 | later | COM. |

Each phase is independently committable and observable.

## What this design does NOT do

- **No cross-platform abstraction.** Win32-specific. If we go cross-
  platform later, the `(on-ui-thread …)` macro generalises (just
  needs different dispatch under the hood), but `%ffi-call` would
  need a parallel macOS/Linux implementation (libffi works on those
  too, but the UI dispatch story differs).
- **No automatic struct layout from WinMD.** Phase 5 adds manual
  `defstruct-win32` declarations. The generator could emit those
  too, but the WinMD metadata's struct shape needs cross-checking
  against the actual C headers — out of scope for the initial cut.
- **No COM.** Vtable dispatch is a different model. Separate doc when
  we get there.
- **No 32-bit support.** x64 only. The metadata's calling-convention
  distinction (`winapi` vs `cdecl`) matters on x86 but collapses on
  x64; our libffi binding hard-codes the x64 ABI.

## Why this scales

The kernel is one Rust shim and a Python generator. Every Win32
function added to the database (regenerate when Microsoft ships a
new SDK metadata package) becomes a one-line Lisp `defun` for free.
The UI-thread routing is a single private message that everything
flows through — no per-function boilerplate.

iGui keeps doing what iGui does. The generic FFI doesn't replace it;
it sits alongside as a sibling surface for programs that want raw
Win32 reach without going through iGui's IDE abstractions.
