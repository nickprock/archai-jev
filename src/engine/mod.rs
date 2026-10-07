//! The inference engine seen from the core (spec 006b section 7).
//!
//! A [`Forward`] turns token **ids** into numbers: the hidden state of the last layer (after the
//! final norm) at some positions, or the model's logits at a position restricted to some ids.
//! It never sees text (D24) and it never reads an output by "row of output" index, which in
//! llama.cpp silently returns the wrong numbers (spike S4): the only way to ask for a number is
//! [`OutputSpec`], by position in the branch you sent.
//!
//! The pattern is the one proven in S4: the state is decoded **once** per request, then every
//! question runs on a private copy of it, so no question can see another and the state is never
//! changed. One [`Session`] is one request.

pub mod cpu;
#[cfg(any(test, feature = "testing"))]
pub mod fake;
pub mod llama;

use crate::error::{Error, Result};

/// What to read from a row of the model. More kinds can be added without touching the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputSpec {
    /// The hidden state (last layer, after the final norm, as f32) at `position` of the branch.
    Hidden {
        /// Index inside the branch.
        position: usize,
    },
    /// The logits of the model at `position` of the branch, only for `ids`.
    Logits {
        /// Index inside the branch.
        position: usize,
        /// Token ids whose logits are wanted, in this order.
        ids: Vec<u32>,
    },
}

/// A number read from the model: the answer to one [`OutputSpec`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Output {
    /// `n_embd` values.
    Hidden(Vec<f32>),
    /// One value per requested id.
    Logits(Vec<f32>),
}

/// One request being answered: the state is already decoded and kept.
pub trait Session {
    /// Run one question: decode `branch` after the state, on a copy that is discarded, and return
    /// one [`Output`] per entry of `outputs`, in order. Values that are not finite are an error
    /// ([`crate::Error::NonFiniteHidden`], with an empty question name: the caller names it).
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the engine fails; the session stays usable.
    fn row(&mut self, branch: &[u32], outputs: &[OutputSpec]) -> Result<Vec<Output>>;
}

/// A loaded model that can answer requests.
pub trait Forward: Send + Sync {
    /// Start a request: decode `state` once (positions `0..state.len()`) and keep it. The model
    /// is not thread-safe, so the returned session holds it: concurrent requests wait their turn.
    ///
    /// # Errors
    /// [`crate::Error::Inference`] if the state cannot be decoded.
    fn begin<'a>(&'a self, state: &[u32]) -> Result<Box<dyn Session + 'a>>;
}

/// The logits of `ids` out of a **whole** row of the model's logits (one value per token of the
/// vocabulary), read at `position`.
///
/// Every value of the row is checked, not only the wanted ones: a `nan` or an infinity anywhere
/// means a corrupt file or an overflow, and the wanted numbers cannot be trusted (spike S5, 8.1).
/// The wanted tokens usually have almost no mass in the full softmax (the letters head was trained
/// on the restricted one), so the mass on them is deliberately **not** checked.
///
/// # Errors
/// [`Error::NonFiniteHidden`] (with an empty question name: the caller names it) if the row has a
/// value that is not finite; [`Error::Inference`] if an id is outside the vocabulary.
pub fn pick_logits(row: &[f32], ids: &[u32], position: usize) -> Result<Vec<f32>> {
    if let Some(bad) = row.iter().copied().find(|v| !v.is_finite()) {
        return Err(Error::NonFiniteHidden {
            question: String::new(),
            position,
            value: f64::from(bad),
        });
    }
    ids.iter()
        .map(|&id| {
            row.get(id as usize)
                .copied()
                .ok_or_else(|| Error::Inference {
                    detail: format!("token id {id} is outside the vocabulary"),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wanted_logits_come_in_the_order_of_the_ids() {
        let row = [0.5f32, -1.0, 2.0, 3.5];
        assert_eq!(pick_logits(&row, &[3, 1], 7).unwrap(), vec![3.5, -1.0]);
        assert!(pick_logits(&row, &[], 7).unwrap().is_empty());
    }

    #[test]
    fn a_value_that_is_not_finite_anywhere_in_the_row_is_refused() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // The wanted ids (0 and 1) are fine: the bad value is elsewhere in the row.
            let mut row = vec![0.0f32; 16];
            row[11] = bad;
            let err = pick_logits(&row, &[0, 1], 4).unwrap_err();
            match err {
                Error::NonFiniteHidden {
                    position, value, ..
                } => {
                    assert_eq!(position, 4);
                    assert!(value.is_nan() == bad.is_nan() && !value.is_finite());
                }
                other => panic!("{other}"),
            }
        }
    }

    #[test]
    fn tokens_with_almost_no_mass_in_the_full_vocabulary_are_not_a_fault() {
        // The wanted tokens are 25 nats below the best one: ~1e-11 of the full softmax. The model
        // was trained on the softmax restricted to them, so this is the normal case.
        let mut row = vec![-10.0f32; 32];
        row[5] = 15.0;
        row[0] = -10.0;
        row[1] = -11.0;
        assert_eq!(pick_logits(&row, &[0, 1], 0).unwrap(), vec![-10.0, -11.0]);
    }

    #[test]
    fn an_id_outside_the_vocabulary_is_an_error_not_a_panic() {
        let err = pick_logits(&[1.0, 2.0], &[0, 2], 0).unwrap_err();
        assert!(matches!(err, Error::Inference { .. }), "{err}");
    }
}
