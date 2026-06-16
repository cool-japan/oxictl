/// Errors produced by the RTPS parser and serializer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtpsError {
    /// Input buffer ended before the field could be read.
    TruncatedHeader,
    /// RTPS magic bytes not found at start of message.
    InvalidMagic,
    /// RTPS protocol version not 2.x.
    UnsupportedVersion,
    /// Serializer output buffer too small for the message.
    BufferTooSmall,
    /// An unrecognized submessage kind was encountered and cannot be skipped
    /// because `octets_to_next_header` was zero (ambiguous length).
    InvalidSubmessageKind(u8),
    /// A Locator kind value was not one of the known RTPS kinds.
    InvalidLocatorKind(i32),
    /// A single Message contained more than 64 submessages.
    TooManySubmessages,
    /// A ParameterList contained more than 32 parameters.
    TooManyParameters,
    /// A Parameter length field would read outside the available slice.
    InvalidParameterLength,
    /// A multi-byte field was not at its required CDR alignment boundary.
    AlignmentError,
    /// A CDR string contained invalid UTF-8 or missing null terminator.
    InvalidStringEncoding,
}

impl core::fmt::Display for RtpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "RTPS: input truncated before header end"),
            Self::InvalidMagic => write!(f, "RTPS: magic bytes not found"),
            Self::UnsupportedVersion => write!(f, "RTPS: protocol version is not 2.x"),
            Self::BufferTooSmall => write!(f, "RTPS: serialize buffer too small"),
            Self::InvalidSubmessageKind(k) => {
                write!(
                    f,
                    "RTPS: unknown submessage kind 0x{k:02X} with zero-length body"
                )
            }
            Self::InvalidLocatorKind(k) => write!(f, "RTPS: unknown locator kind {k}"),
            Self::TooManySubmessages => {
                write!(f, "RTPS: message contains more than 64 submessages")
            }
            Self::TooManyParameters => {
                write!(f, "RTPS: parameter list has more than 32 entries")
            }
            Self::InvalidParameterLength => {
                write!(f, "RTPS: parameter length extends beyond available slice")
            }
            Self::AlignmentError => write!(f, "RTPS: field at incorrect CDR alignment"),
            Self::InvalidStringEncoding => {
                write!(f, "RTPS: CDR string has invalid encoding")
            }
        }
    }
}
