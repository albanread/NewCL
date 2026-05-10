//! S-expression reader.
//!
//! Phase 1b: tokenizer (this and `lexer` module).
//! Phase 1c: parser, sharp-dispatch interpretation, package
//! qualification, readtable-case folding (still to come).

pub mod lexer;
pub mod token;

pub use lexer::{LexError, LexErrorKind, Lexer, LexerIter};
pub use token::{AtomText, Span, SpannedToken, Token};
