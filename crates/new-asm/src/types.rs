//! Core types for ASM procedure declarations.

/// ABI register class for a parameter or return value.
///
/// Maps to Windows x64 register families. Packed-SIMD types
/// (Pair, FPair, Quad, Oct) all lower to `i64` in BCPL's IR and
/// therefore travel in integer registers — they are `Word` here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmType {
    /// i64 — Word, Int, pointers, and all packed-SIMD types.
    Word,
    /// f64 — Float scalar.
    Float,
    /// <4 x f32> — FQuad (XMM).
    FQuad,
    /// <8 x f32> — FOct (YMM).
    FOct,
}

/// Return type of an ASM procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmRetType {
    /// i64, returned in rax.
    Word,
    /// f64, returned in xmm0.
    Float,
    /// <4 x f32>, returned in xmm0.
    FQuad,
    /// <8 x f32>, returned in ymm0.
    FOct,
    /// No return value (BE ASM / void).
    Void,
}

/// One parameter of an ASM procedure.
#[derive(Debug, Clone)]
pub struct AsmParam {
    pub name: String,
    pub ty: AsmType,
}

/// A whole-procedure ASM definition.
///
/// Emitted as a `module asm` blob that defines the symbol in Intel
/// syntax plus an LLVM `declare` so callers can type-check it.
#[derive(Debug, Clone)]
pub struct AsmProc {
    pub name: String,
    pub params: Vec<AsmParam>,
    pub return_type: AsmRetType,
    /// Raw Intel-syntax body text.  No `.intel_syntax` header — that
    /// is prepended by `build_module_asm_string`.  `#name` tokens are
    /// substituted with their Windows x64 ABI registers.
    pub body: String,
}
