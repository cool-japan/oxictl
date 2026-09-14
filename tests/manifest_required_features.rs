//! Guard: every test, example and bench target that uses a feature-gated part
//! of `oxictl` must declare the matching `required-features` in `Cargo.toml`
//! (or `cfg`-gate that usage itself).
//!
//! Without it, `cargo test` / `cargo clippy --all-targets` fail to compile for
//! any feature set that leaves the module disabled (e.g. default features or
//! `--no-default-features`).  The gates are discovered from `src/` itself
//! (`#[cfg(feature = "...")] pub mod ...;` / `pub use ...;`), so new modules
//! are covered automatically.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Module path (e.g. `protocol::dds::api`) or item path → gating feature.
type Gates = HashMap<String, String>;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Extracts `x` from a line of the form `#[cfg(feature = "x")]`.
fn simple_cfg_feature(line: &str) -> Option<String> {
    let inner = line
        .trim()
        .strip_prefix("#[cfg(feature = \"")?
        .strip_suffix("\")]")?;
    Some(inner.to_string())
}

/// All `feature = "x"` names mentioned anywhere in `text`.
fn mentioned_features(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = text;
    while let Some(pos) = rest.find("feature = \"") {
        rest = &rest[pos + "feature = \"".len()..];
        if let Some(end) = rest.find('"') {
            out.insert(rest[..end].to_string());
            rest = &rest[end..];
        }
    }
    out
}

/// Expands every `oxictl::` use tree in `text` into flat paths
/// (without the `oxictl::` prefix).
fn oxictl_paths(text: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find("oxictl::") {
        let preceded_by_ident = rest[..pos]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        rest = &rest[pos + "oxictl::".len()..];
        if preceded_by_ident {
            continue;
        }
        let chars: Vec<char> = rest.chars().collect();
        let mut idx = 0;
        parse_use_tree(&chars, &mut idx, "", &mut paths);
    }
    paths
}

fn parse_use_tree(chars: &[char], idx: &mut usize, prefix: &str, out: &mut Vec<String>) {
    let mut path = prefix.to_string();
    loop {
        skip_ws(chars, idx);
        match chars.get(*idx) {
            Some('{') => {
                *idx += 1;
                loop {
                    skip_ws(chars, idx);
                    match chars.get(*idx) {
                        Some('}') => {
                            *idx += 1;
                            return;
                        }
                        Some(',') => *idx += 1,
                        Some(_) => {
                            let before = *idx;
                            parse_use_tree(chars, idx, &path, out);
                            if *idx == before {
                                // Unparseable token: skip it to guarantee progress.
                                *idx += 1;
                            }
                        }
                        None => return,
                    }
                }
            }
            Some('*') => {
                *idx += 1;
                out.push(path);
                return;
            }
            Some(c) if c.is_alphabetic() || *c == '_' => {
                let start = *idx;
                while chars
                    .get(*idx)
                    .is_some_and(|c| c.is_alphanumeric() || *c == '_')
                {
                    *idx += 1;
                }
                let ident: String = chars[start..*idx].iter().collect();
                if ident != "self" {
                    if !path.is_empty() {
                        path.push_str("::");
                    }
                    path.push_str(&ident);
                }
                if chars.get(*idx) == Some(&':') && chars.get(*idx + 1) == Some(&':') {
                    *idx += 2;
                    continue;
                }
                out.push(path);
                return;
            }
            _ => {
                if !path.is_empty() {
                    out.push(path);
                }
                return;
            }
        }
    }
}

fn skip_ws(chars: &[char], idx: &mut usize) {
    while chars.get(*idx).is_some_and(|c| c.is_whitespace()) {
        *idx += 1;
    }
}

/// Recursively records feature-gated `pub mod` / `pub use` items of the crate.
fn collect_gates(file: &Path, module: &str, child_dir: &Path, gates: &mut Gates) -> TestResult {
    let text = fs::read_to_string(file)?;
    let mut pending: Option<String> = None;
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if let Some(feature) = simple_cfg_feature(trimmed) {
            pending = Some(feature);
            continue;
        }
        let gate = pending.take();
        if let Some(name) = trimmed
            .strip_prefix("pub mod ")
            .and_then(|r| r.strip_suffix(';'))
        {
            let full = join(module, name);
            if let Some(feature) = gate {
                gates.insert(full.clone(), feature);
            }
            let flat = child_dir.join(format!("{name}.rs"));
            let nested = child_dir.join(name).join("mod.rs");
            if flat.is_file() {
                collect_gates(&flat, &full, &child_dir.join(name), gates)?;
            } else if nested.is_file() {
                collect_gates(&nested, &full, &child_dir.join(name), gates)?;
            }
        } else if let (Some(feature), Some(start)) = (gate, trimmed.strip_prefix("pub use ")) {
            let mut stmt = start.to_string();
            while !stmt.contains(';') {
                match lines.next() {
                    Some(next) => stmt.push_str(next),
                    None => break,
                }
            }
            let chars: Vec<char> = stmt.chars().collect();
            let mut idx = 0;
            let mut leaves = Vec::new();
            parse_use_tree(&chars, &mut idx, "", &mut leaves);
            for leaf in leaves {
                if let Some(item) = leaf.rsplit("::").next() {
                    gates.insert(join(module, item), feature.clone());
                }
            }
        }
    }
    Ok(())
}

fn join(module: &str, name: &str) -> String {
    if module.is_empty() {
        name.to_string()
    } else {
        format!("{module}::{name}")
    }
}

/// Features needed to name `path` (gates of every prefix).
fn needed_features(path: &str, gates: &Gates) -> BTreeSet<String> {
    let mut needed = BTreeSet::new();
    let mut prefix = String::new();
    for segment in path.split("::") {
        prefix = join(&prefix, segment);
        if let Some(feature) = gates.get(&prefix) {
            needed.insert(feature.clone());
        }
    }
    needed
}

/// Transitive closure of `features` under the `[features]` table.
fn feature_closure(
    features: &BTreeSet<String>,
    table: &BTreeMap<String, Vec<String>>,
) -> BTreeSet<String> {
    let mut closure = BTreeSet::new();
    let mut stack: Vec<String> = features.iter().cloned().collect();
    while let Some(feature) = stack.pop() {
        if !closure.insert(feature.clone()) {
            continue;
        }
        for implied in table.get(&feature).into_iter().flatten() {
            if !implied.starts_with("dep:") && !implied.contains('/') {
                stack.push(implied.clone());
            }
        }
    }
    closure
}

/// A compilation target: its root file, all its source files, and the
/// features declared in `required-features`.
struct Target {
    kind: &'static str,
    root: PathBuf,
    files: Vec<PathBuf>,
}

fn discover_targets(root: &Path) -> Result<Vec<Target>, Box<dyn std::error::Error>> {
    let mut targets = Vec::new();
    for kind in ["tests", "examples", "benches"] {
        let dir = root.join(kind);
        if !dir.is_dir() {
            continue;
        }
        let mut entries: Vec<PathBuf> = fs::read_dir(&dir)?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        entries.sort();
        for path in entries {
            if path.extension().is_some_and(|e| e == "rs") {
                targets.push(Target {
                    kind,
                    root: path.clone(),
                    files: vec![path],
                });
            } else if path.join("main.rs").is_file() {
                let mut files = Vec::new();
                for entry in fs::read_dir(&path)? {
                    let file = entry?.path();
                    if file.extension().is_some_and(|e| e == "rs") {
                        files.push(file);
                    }
                }
                files.sort();
                targets.push(Target {
                    kind,
                    root: path.join("main.rs"),
                    files,
                });
            }
        }
    }
    Ok(targets)
}

/// `path` (relative to the manifest, `/`-separated) → `required-features`.
fn declared_requirements(
    manifest: &toml::Table,
) -> Result<HashMap<String, BTreeSet<String>>, Box<dyn std::error::Error>> {
    let mut out = HashMap::new();
    for (section, default_dir) in [
        ("test", "tests"),
        ("example", "examples"),
        ("bench", "benches"),
    ] {
        let Some(entries) = manifest.get(section).and_then(|v| v.as_array()) else {
            continue;
        };
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("[[{section}]] entry without a name"))?;
            let path = entry
                .get("path")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("{default_dir}/{name}.rs"));
            let required: BTreeSet<String> = entry
                .get("required-features")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            out.insert(path, required);
        }
    }
    Ok(out)
}

#[test]
fn test_every_target_declares_required_features() -> TestResult {
    let root = manifest_dir();
    let manifest: toml::Table = fs::read_to_string(root.join("Cargo.toml"))?.parse()?;

    let feature_table: BTreeMap<String, Vec<String>> = manifest
        .get("features")
        .and_then(|v| v.as_table())
        .ok_or("Cargo.toml has no [features] table")?
        .iter()
        .map(|(k, v)| {
            let implied = v
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect();
            (k.clone(), implied)
        })
        .collect();

    let mut gates = Gates::new();
    let src = root.join("src");
    collect_gates(&src.join("lib.rs"), "", &src, &mut gates)?;
    assert!(
        gates.get("pid").is_some_and(|f| f == "pid"),
        "gate discovery failed: `pid` module gate not found"
    );

    let declared = declared_requirements(&manifest)?;
    let targets = discover_targets(&root)?;
    assert!(!targets.is_empty(), "no targets discovered");

    let mut problems = Vec::new();
    for target in &targets {
        let rel = target
            .root
            .strip_prefix(&root)?
            .to_string_lossy()
            .replace('\\', "/");
        if rel == file!().replace('\\', "/") {
            // This guard's own fixtures mention gated paths on purpose.
            continue;
        }
        let required = declared.get(&rel).cloned().unwrap_or_default();
        let enabled = feature_closure(&required, &feature_table);

        // `cfg` gates on a sub-module's `mod` declaration in main.rs.
        let root_text = fs::read_to_string(&target.root)?;
        let mut mod_gates: HashMap<String, BTreeSet<String>> = HashMap::new();
        let mut pending = BTreeSet::new();
        for line in root_text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("#[cfg(") {
                pending = mentioned_features(trimmed);
            } else if let Some(name) = trimmed
                .strip_prefix("mod ")
                .and_then(|r| r.strip_suffix(';'))
            {
                mod_gates.insert(name.to_string(), std::mem::take(&mut pending));
            } else if !trimmed.is_empty() && !trimmed.starts_with("//") {
                pending.clear();
            }
        }

        for file in &target.files {
            let text = fs::read_to_string(file)?;
            let mut locally_gated = mentioned_features(&text);
            if let Some(stem) = file.file_stem().and_then(|s| s.to_str()) {
                if let Some(extra) = mod_gates.get(stem) {
                    locally_gated.extend(extra.iter().cloned());
                }
            }
            for path in oxictl_paths(&text) {
                for feature in needed_features(&path, &gates) {
                    if !enabled.contains(&feature) && !locally_gated.contains(&feature) {
                        problems.push(format!(
                            "{} target `{rel}`: {} uses `oxictl::{path}` which needs feature \
                             `{feature}`, but required-features = {:?}",
                            target.kind,
                            file.strip_prefix(&root)?.display(),
                            required
                        ));
                    }
                }
            }
        }
    }

    assert!(
        problems.is_empty(),
        "missing required-features:\n{}",
        problems.join("\n")
    );
    Ok(())
}

#[test]
fn test_use_tree_expansion() {
    let text = "use oxictl::{core::matrix::Matrix, protocol::{dds::api::X, modbus}};\n\
                let _ = oxictl::pid::Pid::new(); // notoxictl::sim::Y";
    let paths = oxictl_paths(text);
    assert_eq!(
        paths,
        vec![
            "core::matrix::Matrix".to_string(),
            "protocol::dds::api::X".to_string(),
            "protocol::modbus".to_string(),
            "pid::Pid::new".to_string(),
        ]
    );
}
