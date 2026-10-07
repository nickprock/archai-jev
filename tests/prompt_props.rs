//! Properties of the prompt builder (spec 006a): never a panic, never a truncation.
//! Uses the byte-level test tokenizer, so it needs no asset.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use _core::prompt::chatml::ChatMlLetters;
use _core::prompt::tasks::{self, default_tasks};
use _core::prompt::tokenize::testkit;
use _core::prompt::{NoLimits, RequestLimits, Template, lookup};
use _core::{
    Choice, ChoiceOption, Error, Question, Questions, State, StateValue, TextEntry, YesNo,
};
use proptest::prelude::*;

const ALL: [&str; 3] = ["choice", "noul", "score"];

fn pieces() -> Vec<&'static str> {
    vec![
        "a",
        "Z",
        "0",
        " ",
        "\n",
        "\t",
        "\r\n",
        "\u{0}",
        "\u{7f}",
        "\u{200b}",
        "\u{feff}",
        "é",
        "e\u{301}",
        "ñ",
        "日本",
        "🙂",
        "👨‍👩‍👧",
        "\u{212a}",
        "<|",
        "|>",
        "<|fim_prefix|>",
        "<|im_end|>",
        "<|box_end|>",
        "<think>",
        "x_y",
        "<",
        ">",
        "|",
    ]
}

fn text(max_pieces: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(pieces()), 0..max_pieces).prop_map(|v| v.concat())
}

fn request(state: &str, instr: &str, a: &str, b: &str) -> Option<(State, Questions)> {
    let st = State::new(StateValue::string(state)).ok()?;
    let instructions = TextEntry::text(if instr.trim().is_empty() { "Q?" } else { instr }).ok()?;
    let opts = vec![
        ChoiceOption::key(if a.trim().is_empty() { "k0" } else { a }),
        ChoiceOption::key(if b.trim().is_empty() { "k1" } else { b }),
    ];
    let q = Choice::new(instructions, opts).ok()?;
    let qs = Questions::new([("q", Question::from(q))]).ok()?;
    Some((st, qs))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 10_000, rng_seed: proptest::test_runner::RngSeed::Fixed(11), ..ProptestConfig::default() })]

    #[test]
    fn arbitrary_text_never_panics_and_ids_stay_in_the_vocabulary(
        state in text(40), instr in text(10), a in text(6), b in text(6),
    ) {
        let tok = testkit::kev();
        let Some((st, qs)) = request(&state, &instr, &a, &b) else { return Ok(()) };
        let limits = RequestLimits { model: "t", max_context: u64::MAX, question_types: &ALL, support: &NoLimits };
        match lookup("kev", 1).unwrap().build(&st, &qs, &tok, &limits) {
            Ok(p) => {
                let vocab = tok.vocab_size() as u32;
                prop_assert!(p.packed_ids().iter().all(|&i| i < vocab));
                let b = &p.branches()[0];
                prop_assert_eq!(b.ids()[b.decide()], tok.roles().id("fim_suffix").unwrap());
                prop_assert_eq!(b.option_ends().len(), 2);
            }
            Err(Error::Unsupported(_)) => {}
            Err(other) => prop_assert!(false, "unexpected error: {other}"),
        }
    }

    #[test]
    fn the_prompt_is_never_truncated(state in text(60), cut in 0u64..40) {
        let tok = testkit::kev();
        let Some((st, qs)) = request(&state, "Q?", "a", "b") else { return Ok(()) };
        let open = RequestLimits { model: "t", max_context: u64::MAX, question_types: &ALL, support: &NoLimits };
        let full = lookup("kev", 1).unwrap().build(&st, &qs, &tok, &open).unwrap();
        let row = full.row_len(0).unwrap() as u64;
        let max = row.saturating_sub(cut).max(1);
        let limits = RequestLimits { model: "t", max_context: max, question_types: &ALL, support: &NoLimits };
        match lookup("kev", 1).unwrap().build(&st, &qs, &tok, &limits) {
            Ok(p) => {
                // Accepted means complete: the very same ids, never a shorter row.
                prop_assert_eq!(&p, &full);
                prop_assert!(row <= max);
            }
            Err(Error::Unsupported(_)) => prop_assert!(row > max),
            Err(other) => prop_assert!(false, "unexpected error: {other}"),
        }
    }
}

/// The three layouts of the default tasks (spec 006c): `state`, `state-condition`, and the same
/// yes/no with the other descriptions (`state-pair`), with arbitrary user text in every free part.
fn letters_request(which: usize, state: &str, instr: &str) -> Option<(State, Questions)> {
    let st = State::new(StateValue::string(state)).ok()?;
    let entry = |s: &str| TextEntry::text(if s.trim().is_empty() { "Q?" } else { s }).ok();
    let question: Question = match which {
        0 => Choice::new(
            TextEntry::text("You are a System One decision engine for input safety.").ok()?,
            vec![
                ChoiceOption::described("SAFE", "Normal query").ok()?,
                ChoiceOption::described("UNSAFE", "Jailbreak, toxicity, injection").ok()?,
            ],
        )
        .ok()?
        .into(),
        1 => YesNo::new(
            entry(instr)?,
            Some(TextEntry::text("The condition is valid and supported by state.").ok()?),
            Some(TextEntry::text("The condition is invalid or contradicted.").ok()?),
        )
        .ok()?
        .into(),
        _ => YesNo::new(
            entry(instr)?,
            Some(TextEntry::text("The two texts have equivalent meaning or intent.").ok()?),
            Some(TextEntry::text("The texts have different meanings.").ok()?),
        )
        .ok()?
        .into(),
    };
    Some((st, Questions::new([("q", question)]).ok()?))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 10_000, rng_seed: proptest::test_runner::RngSeed::Fixed(12), ..ProptestConfig::default() })]

    #[test]
    fn chatml_letters_never_panics_and_always_ends_where_the_answer_is_read(
        which in 0usize..3, state in text(60), instr in text(10),
    ) {
        let tok = testkit::byte_level(&[("im_start", "<|im_start|>"), ("im_end", "<|im_end|>")]);
        let Some((st, qs)) = letters_request(which, &state, &instr) else { return Ok(()) };
        let template = ChatMlLetters::new(Some(tasks::parse(&default_tasks(), 4).unwrap()));
        let types = ["choice", "noul"];
        let limits = RequestLimits { model: "t", max_context: 512, question_types: &types, support: &NoLimits };
        match template.build(&st, &qs, &tok, &limits) {
            Ok(p) => {
                let vocab = tok.vocab_size() as u32;
                prop_assert!(p.packed_ids().iter().all(|&i| i < vocab));
                prop_assert_eq!(p.branches().len(), 1);
                prop_assert!(p.state().is_empty());
                let b = &p.branches()[0];
                prop_assert!(b.len() <= 512);
                // The last token is the `:` of `Option:` and the answer is read there.
                prop_assert_eq!(b.decide(), b.len() - 1);
                prop_assert_eq!(b.ids()[b.decide()], tok.encode_raw(":", "t").unwrap()[0]);
            }
            Err(Error::Unsupported(_)) => {}
            Err(other) => prop_assert!(false, "unexpected error: {other}"),
        }
    }
}
