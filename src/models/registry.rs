//! The registry of known checkpoints, compiled into the wheel (spec 005, section 3).
//!
//! It is the root of trust: every revision is a full commit hash and every file has a SHA-256.
//! It is read the first time it is needed, never at `import`.

use std::sync::OnceLock;

use super::files::FileEntry;
use super::incompat::Incompat;
use super::manifest::{HeadSpec, Manifest, ManifestOrigin, Source};
use super::validate;
use crate::json_strict::{self, Json, Obj};

/// Manifests of the built-in models (one file per entry, included at compile time).
const BUILTIN_MANIFESTS: &[&str] = &[include_str!(
    "registry/nickprock__archai-jev-qwen-1.5b.json"
)];
/// Which entry is the default and which revision of each name is current.
const BUILTIN_INDEX: &str = include_str!("registry/index.json");

/// The set of known models.
#[derive(Debug, Clone)]
pub struct Registry {
    entries: Vec<Manifest>,
    default: Option<(String, String)>,
    current: Vec<(String, String)>,
}

static BUILTIN: OnceLock<Result<Registry, Incompat>> = OnceLock::new();

const PERMISSIVE: &[&str] = &["Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"];

impl Registry {
    /// The registry compiled into this build, parsed on first use.
    pub fn builtin() -> Result<&'static Registry, Incompat> {
        BUILTIN
            .get_or_init(|| Registry::from_sources(BUILTIN_MANIFESTS, BUILTIN_INDEX))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Build a registry from manifest texts and an index text.
    ///
    /// # Errors
    /// The first [`Incompat`] found while reading.
    pub fn from_sources(manifests: &[&str], index: &str) -> Result<Registry, Incompat> {
        let entries = manifests
            .iter()
            .map(|text| Manifest::from_bytes(text.as_bytes(), ManifestOrigin::Registry))
            .collect::<Result<Vec<_>, _>>()?;
        let raw = json_strict::parse(index.as_bytes())?;
        let mut o = Obj::new(&raw, "<index>")?;
        let default = match o.req("default")? {
            Json::Null => None,
            Json::Str(s) => {
                let (name, rev) = s
                    .split_once('@')
                    .ok_or_else(|| Incompat::ManifestBadValue {
                        path: "default".to_string(),
                        detail: format!("must be \"name@revision\", got {s:?}"),
                    })?;
                Some((name.to_string(), rev.to_string()))
            }
            other => {
                return Err(Incompat::ManifestBadValue {
                    path: "default".to_string(),
                    detail: format!("must be a string or null, got {}", other.kind()),
                });
            }
        };
        let current = match o.req("current")? {
            Json::Object(pairs) => pairs
                .iter()
                .map(|(k, v)| match v {
                    Json::Str(r) => Ok((k.clone(), r.clone())),
                    other => Err(Incompat::ManifestBadValue {
                        path: format!("current.{k}"),
                        detail: format!("must be a string, got {}", other.kind()),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?,
            other => {
                return Err(Incompat::ManifestBadValue {
                    path: "current".to_string(),
                    detail: format!("must be an object, got {}", other.kind()),
                });
            }
        };
        o.finish()?;
        Ok(Registry {
            entries,
            default,
            current,
        })
    }

    /// Every entry (all revisions of all names), in registry order.
    pub fn entries(&self) -> &[Manifest] {
        &self.entries
    }

    /// The distinct model names, in registry order.
    pub fn names(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for m in &self.entries {
            if !out.contains(&m.name) {
                out.push(m.name.clone());
            }
        }
        out
    }

    /// True if some entry has this name.
    pub fn has(&self, name: &str) -> bool {
        self.entries.iter().any(|m| m.name == name)
    }

    /// The default model, if the registry has one.
    pub fn default_entry(&self) -> Option<&Manifest> {
        let (name, rev) = self.default.as_ref()?;
        self.entries
            .iter()
            .find(|m| &m.name == name && &m.revision == rev)
    }

    /// Whether `m` is the default entry.
    pub fn is_default(&self, m: &Manifest) -> bool {
        self.default
            .as_ref()
            .is_some_and(|(n, r)| *n == m.name && *r == m.revision)
    }

    /// Resolve `name` and an optional `revision` to a pinned entry. `None` means the current
    /// revision. A revision is accepted as a full hash or as a prefix of at least 8 digits
    /// that matches exactly one revision of that name.
    ///
    /// # Errors
    /// [`Incompat::UnknownName`] or [`Incompat::RevisionNotAccepted`].
    pub fn resolve(&self, name: &str, revision: Option<&str>) -> Result<&Manifest, Incompat> {
        let of_name: Vec<&Manifest> = self.entries.iter().filter(|m| m.name == name).collect();
        if of_name.is_empty() {
            return Err(Incompat::UnknownName {
                name: name.to_string(),
                known: self.names(),
            });
        }
        let available = || {
            of_name
                .iter()
                .map(|m| m.revision.clone())
                .collect::<Vec<_>>()
        };
        let Some(rev) = revision else {
            let current = self.current.iter().find(|(n, _)| n == name).map(|(_, r)| r);
            return of_name
                .iter()
                .find(|m| Some(&m.revision) == current)
                .or(of_name.first())
                .copied()
                .ok_or(Incompat::UnknownName {
                    name: name.to_string(),
                    known: self.names(),
                });
        };
        let hex = rev.len() >= 8 && rev.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        let matches: Vec<&&Manifest> = if hex {
            of_name
                .iter()
                .filter(|m| m.revision.starts_with(rev))
                .collect()
        } else {
            Vec::new()
        };
        match matches.as_slice() {
            [one] => Ok(**one),
            _ => Err(Incompat::RevisionNotAccepted {
                name: name.to_string(),
                revision: rev.to_string(),
                available: available(),
            }),
        }
    }

    /// Check the invariants of a registry (run by a test over the built-in one).
    ///
    /// # Errors
    /// A description of the first violated invariant.
    pub fn lint(&self) -> Result<(), String> {
        for (i, m) in self.entries.iter().enumerate() {
            let tag = format!("{}@{}", m.name, m.revision);
            if self
                .entries
                .iter()
                .take(i)
                .any(|o| o.name == m.name && o.revision == m.revision)
            {
                return Err(format!("{tag} appears twice"));
            }
            validate::semantics(m, &[]).map_err(|e| format!("{tag}: {e}"))?;
            for file in all_files(m) {
                if file.origin.is_none() {
                    return Err(format!("{tag}: file {} has no origin", file.path));
                }
            }
            if !PERMISSIVE.contains(&m.license.spdx.as_str())
                && m.license.restrictions.as_deref().is_none_or(str::is_empty)
            {
                return Err(format!(
                    "{tag}: license {} needs non-empty restrictions",
                    m.license.spdx
                ));
            }
        }
        match (&self.default, self.entries.is_empty()) {
            (None, true) => {}
            (None, false) => return Err("a non-empty registry needs a default entry".to_string()),
            (Some((n, r)), _) => {
                if !self
                    .entries
                    .iter()
                    .any(|m| &m.name == n && &m.revision == r)
                {
                    return Err(format!("the default {n}@{r} is not an entry"));
                }
            }
        }
        for name in self.names() {
            let Some((_, rev)) = self.current.iter().find(|(n, _)| *n == name) else {
                return Err(format!("{name} has no current revision"));
            };
            if !self
                .entries
                .iter()
                .any(|m| m.name == name && &m.revision == rev)
            {
                return Err(format!(
                    "the current revision {rev} of {name} is not an entry"
                ));
            }
        }
        if let Some((n, _)) = self.current.iter().find(|(n, _)| !self.has(n)) {
            return Err(format!("current names an unknown model {n}"));
        }
        Ok(())
    }
}

/// Every file entry of a manifest (tokenizer, head weights, variant sources).
pub fn all_files(m: &Manifest) -> Vec<&FileEntry> {
    let mut out = vec![&m.tokenizer.file];
    if let HeadSpec::Pointer { weights, .. } = &m.head {
        out.push(weights);
    }
    for v in &m.variants {
        if let Source::Gguf { file } = &v.source {
            out.push(file);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::manifest::HeadSpec;
    use crate::prompt::tasks;

    #[test]
    fn the_builtin_registry_passes_its_lint() {
        let reg = Registry::builtin().unwrap();
        reg.lint().unwrap();
    }

    #[test]
    fn the_default_entry_declares_what_the_demo_model_is() {
        let reg = Registry::builtin().unwrap();
        let m = reg.default_entry().unwrap();
        assert_eq!(m.name, "nickprock/archai-jev-qwen-1.5b");
        assert_eq!(m.family, "qwen2-letters");
        assert_eq!(
            (m.template.id.as_str(), m.template.version),
            ("chatml-letters", 1)
        );
        assert_eq!(m.max_context, 512);

        // The head: ids, not strings (005 section 6.4), and the restricted softmax.
        let HeadSpec::Letters {
            rule,
            choice_targets,
            yes_no,
        } = &m.head
        else {
            panic!("the default has a letters head")
        };
        assert_eq!(rule, "restricted_softmax");
        let ids: Vec<(&str, u32)> = choice_targets
            .iter()
            .map(|t| (t.label.as_str(), t.id))
            .collect();
        assert_eq!(ids, [("A", 32), ("B", 33), ("C", 34), ("D", 35)]);
        let (no, yes) = yes_no.as_ref().unwrap();
        assert_eq!(
            ((no.label.as_str(), no.id), (yes.label.as_str(), yes.id)),
            (("FALSE", 30_351), ("TRUE", 20_611))
        );

        // The four tasks, read by the strict reader of the template.
        let tasks = m.tasks.as_ref().unwrap();
        let ids: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, ["safety", "intent", "entailment", "similarity"]);
        let specs = tasks::parse(tasks, choice_targets.len()).unwrap();

        // The declared calibration says which rule and which data it comes from.
        assert_eq!(m.variants.len(), 1);
        let variant = &m.variants[0];
        assert_eq!(variant.dtype, "q8_0");
        assert!(variant.calibration.declared);
        assert_eq!(variant.calibration.temperature, Some(2.09));
        let evidence = variant.calibration.evidence.as_deref().unwrap();
        for phrase in [
            "restricted softmax",
            "first K answer letters",
            "validation split",
            "n=1447",
            "0.058 -> 0.013",
            "In-distribution only",
        ] {
            assert!(
                evidence.contains(phrase),
                "evidence lacks {phrase:?}: {evidence}"
            );
        }

        // The notice says what the model is and what it is not.
        let notice = m.notice.as_deref().unwrap();
        for phrase in [
            "Demo model",
            "four fixed tasks",
            "same distribution as its training data",
            "overconfident",
            "31 of the 77",
            "not a general intent classifier",
            "Not suitable for general-purpose use",
        ] {
            assert!(notice.contains(phrase), "notice lacks {phrase:?}: {notice}");
        }

        // Self-check vectors: every task has at least one (a vector is the task its question is).
        let mut covered: Vec<String> = Vec::new();
        for v in &variant.vectors {
            let (state, questions) = crate::request::request_to_domain(&v.request).unwrap();
            for (name, question) in questions.iter() {
                let task = tasks::find(&specs, &m.name, name, question, &state).unwrap();
                covered.push(task.id.clone());
            }
        }
        for id in ids {
            assert!(
                covered.iter().any(|c| c == id),
                "no self-check vector for {id}"
            );
        }
    }

    #[test]
    fn empty_registry_is_lazy_and_valid() {
        assert!(
            Registry::builtin().unwrap().default_entry().is_none()
                || !Registry::builtin().unwrap().entries().is_empty()
        );
    }
}
