// Tests for u64 precision in JSON round-trip between Rust serde_json and JavaScript.
// The core finding: JavaScript parses JSON numbers as f64, which loses precision for
// values above 2^53. Even though `BigInt(Number(x))` can recover some values, the
// intermediate rounding means serde_json in Rust receives a DIFFERENT number than what
// was originally sent. This corrupts bitboard-based game state.

#[test]
fn test_breakthrough_starting_bitboard_roundtrip() {
    // The correct starting black bitboard
    let start_black: u64 = 0xFFFF_0000_0000_0000;

    // serde_json serializes this as a JSON number
    let serialized_val = serde_json::to_value(start_black).unwrap();
    let json_str = serde_json::to_string(&serialized_val).unwrap();
    println!("Rust serializes: {json_str}");

    // JavaScript receives this number and stores it as f64.
    // When JS sends it back (e.g., in a legal_moves request), JSON.stringify
    // outputs the shortest decimal that maps to the same f64.
    // For 18446462598732840960, that is "18446462598732840000".
    // (verified in Node: JSON.stringify(18446462598732840960) === "18446462598732840000")

    let js_sent_value: u64 = serde_json::from_str("18446462598732840000").unwrap();
    println!("JS sends back:     {js_sent_value} ({js_sent_value:#016x})");
    println!("Original value:    {start_black}    ({start_black:#016x})");
    println!("Difference:        {}", start_black as i64 - js_sent_value as i64);

    // The u64 parsed by Rust from the JS value is DIFFERENT from the original:
    assert_ne!(js_sent_value, start_black,
        "After JS round-trip, the bitboard should be corrupted");

    // The JS round-trip value is 18446462598732840000, which is NOT equal to
    // 0xFFFF_0000_0000_0000. When deser_state uses this to create a BitBoard,
    // the resulting state has a wrong bit pattern.
    assert_ne!(js_sent_value, 0xFFFF_0000_0000_0000,
        "The recovered value should NOT match the correct starting bitboard");

    println!("Server receives u64: {js_sent_value}");
    println!("Server receives hex: {js_sent_value:#016x}");

    // Note: the f64 representation still encodes the correct value.
    // `js_sent_value as f64 as u64` would recover the original.
    // But serde_json never does that — it parses the JSON string directly
    // as a u64, receiving the corrupted value.
    let recovered_via_f64 = (js_sent_value as f64) as u64;
    println!("Recovered via f64:   {recovered_via_f64} ({recovered_via_f64:#016x})");
    assert_eq!(recovered_via_f64, 0xFFFF_0000_0000_0000,
        "f64 truncation recovers the original");
}

#[test]
fn test_serde_json_u64_exactness() {
    // Test which values survive the round-trip through serde_json alone (no JS)
    let test_values = [
        0u64,
        1u64,
        0xFFFFu64,
        0xFFFFFFFFu64,
        0xFFFFFFFFFFFFFFFFu64,
        0xFFFF_0000_0000_0000u64,  // breakthrough black start
        0x0000_0000_0000_FFFFu64,  // breakthrough white start
        0x0000_0000_0000_FF00u64,  // white after losing one piece
        0xFFFF_0000_0000_0100u64,  // black after gaining bit 8
    ];

    for &val in &test_values {
        let serialized = serde_json::to_value(val).unwrap();
        let deserialized: u64 = serde_json::from_value(serialized).unwrap();
        assert_eq!(deserialized, val,
            "serde_json alone must round-trip u64 exactly for {val:#018x}");
    }
}

#[test]
fn test_which_values_lose_precision_via_js() {
    // These are values that, when converted through an f64 (simulating JS),
    // lose precision.
    // The key question: which bitboard values commonly appear in breakthrough
    // that would lose precision?

    let start_black = 0xFFFF_0000_0000_0000u64;
    let start_white = 0x0000_0000_0000_FFFFu64;

    let serialized = serde_json::to_string(
        &serde_json::to_value(start_black).unwrap()
    ).unwrap();

    // Simulate JS: parse the JSON string as f64
    let parsed_f64: f64 = serde_json::from_str(&serialized).unwrap();

    // When JavaScript re-serializes this value, JSON.stringify outputs the
    // shortest decimal that maps to the same f64. For 18446462598732840960,
    // that is "18446462598732840000" (verified in Node.js).
    let _js_string = format!("{parsed_f64:.1?}"); // Simulate JS number serialization

    // When the JavaScript value is sent back to Rust, serde_json parses
    // "18446462598732840000" as a u64. This is a DIFFERENT u64 value!
    let js_sent_value: u64 = serde_json::from_str("18446462598732840000").unwrap();
    println!("JS re-serialized and sent back: {js_sent_value}");
    println!("Original value:                  {start_black}");
    println!("Difference:                      {}", start_black - js_sent_value);
    assert_ne!(js_sent_value, start_black,
        "The value from JS is corrupted and differs from the original");

    // The f64 as u64 trick recovers the original because f64 stores the
    // nearest representable value, but the actual JSON number sent back
    // by JavaScript is a different decimal string, which Rust parses
    // as a different u64 directly.
    println!("parsed_f64 as u64: {}", parsed_f64 as u64);
    assert_eq!(parsed_f64 as u64, start_black,
        "f64 as u64 recovers the original (f64 stores nearest representable)");

    // White bitboard is only 16 bits, well within f64 exact range
    assert_eq!(
        serde_json::from_str::<f64>(
            &serde_json::to_string(&serde_json::to_value(start_white).unwrap()).unwrap()
        ).unwrap() as u64,
        start_white,
        "White start bitboard (small value) should survive f64 round-trip"
    );
}

#[test]
fn test_precision_on_smaller_values() {
    // Find the threshold above which u64 values lose precision
    let max_safe = 1u64 << 53;
    println!("Max safe integer (2^53): {max_safe}");

    // Values up to 2^53 are exact
    assert_eq!(
        serde_json::from_value::<u64>(serde_json::json!(max_safe)).unwrap(),
        max_safe
    );

    // Values above 2^53 are problematic when going through JS
    let above = max_safe + 1;
    let serialized = serde_json::to_string(&serde_json::json!(above)).unwrap();
    let parsed_f64: f64 = serde_json::from_str(&serialized).unwrap();
    let back = parsed_f64 as u64;
    println!("{above} → f64 → {back}, equal? {}", above == back);
    // This may or may not be exact depending on the specific value
}