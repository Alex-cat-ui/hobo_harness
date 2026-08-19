//! The workflow graph: model, validation, and the topology the engine walks.
//!
//! Pure by construction — no I/O, no clock. The validator carries a promise the
//! engine relies on, so it is the kind of code that earns property testing.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub type NodeId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Input,
    Output,
    Agent,
    Foreman,
    Tool,
    Gate,
    Branch,
    Loop,
    Merge,
}

impl NodeKind {
    /// Kinds that instantiate a role and therefore need a model and an output.
    pub fn needs_model(&self) -> bool {
        matches!(self, NodeKind::Agent | NodeKind::Foreman)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub slot: Option<String>,
    /// Document this node writes. Required of anything that produces one.
    #[serde(default)]
    pub output: Option<String>,
    /// Required of a Loop node: the engine refuses an unbounded one.
    #[serde(default)]
    pub loop_limit: Option<u32>,
    /// Required of a Tool node: program first, then arguments.
    #[serde(default)]
    pub command: Option<Vec<String>>,
    #[serde(default)]
    pub gate: bool,
}

/// A branch reads a field of a document's `results` block. Never the prose:
/// routing that depends on wording breaks the moment a model rephrases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Condition {
    pub document: String,
    pub field: String,
    pub op: Op,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Eq,
    Ne,
    Gt,
    Lt,
}

impl Condition {
    /// Numeric where both sides parse as numbers, textual otherwise.
    pub fn holds(&self, actual: &str) -> bool {
        match (actual.trim().parse::<f64>(), self.value.trim().parse::<f64>()) {
            (Ok(a), Ok(b)) => match self.op {
                Op::Eq => (a - b).abs() < f64::EPSILON,
                Op::Ne => (a - b).abs() >= f64::EPSILON,
                Op::Gt => a > b,
                Op::Lt => a < b,
            },
            _ => {
                let a = actual.trim();
                let b = self.value.trim();
                match self.op {
                    Op::Eq => a.eq_ignore_ascii_case(b),
                    Op::Ne => !a.eq_ignore_ascii_case(b),
                    Op::Gt | Op::Lt => false,
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    /// Unconditional when absent. On a Branch node the first edge whose
    /// condition holds is taken; an unconditional edge is the fallback.
    #[serde(default)]
    pub when: Option<Condition>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// Exactly the five failures the specification promises, each naming its node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    DanglingInput(NodeId),
    Unreachable(NodeId),
    NoPathToOutput(NodeId),
    UnboundedLoop(NodeId),
    InvalidNode { id: NodeId, why: &'static str },
    /// Structural preconditions, distinct from the five node-level checks.
    MissingInput,
    MissingOutput,
    DuplicateId(NodeId),
    UnknownEndpoint(NodeId),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use GraphError::*;
        match self {
            DanglingInput(id) => write!(f, "`{id}` has no incoming connection"),
            Unreachable(id) => write!(f, "`{id}` cannot be reached from the Input"),
            NoPathToOutput(id) => write!(f, "`{id}` has no path to the Output, so its work would be discarded"),
            UnboundedLoop(id) => write!(f, "the cycle through `{id}` has no iteration limit"),
            InvalidNode { id, why } => write!(f, "`{id}` is misconfigured: {why}"),
            MissingInput => write!(f, "the graph has no Input node"),
            MissingOutput => write!(f, "the graph has no Output node"),
            DuplicateId(id) => write!(f, "`{id}` is defined more than once"),
            UnknownEndpoint(id) => write!(f, "a connection names `{id}`, which does not exist"),
        }
    }
}

impl Graph {
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn successors(&self, id: &str) -> Vec<&NodeId> {
        self.edges.iter().filter(|e| e.from == id).map(|e| &e.to).collect()
    }

    pub fn predecessors(&self, id: &str) -> Vec<&NodeId> {
        self.edges.iter().filter(|e| e.to == id).map(|e| &e.from).collect()
    }

    fn of_kind(&self, kind: NodeKind) -> Vec<&Node> {
        self.nodes.iter().filter(|n| n.kind == kind).collect()
    }

    /// Edges that close a cycle, found by depth-first search. The engine
    /// excludes these when deciding readiness, or a review loop would wait on
    /// its own descendant for ever.
    pub fn back_edges(&self) -> BTreeSet<(NodeId, NodeId)> {
        let mut back = BTreeSet::new();
        let mut colour: BTreeMap<&str, u8> = BTreeMap::new(); // 0 white, 1 grey, 2 black

        fn visit<'a>(
            g: &'a Graph,
            id: &'a str,
            colour: &mut BTreeMap<&'a str, u8>,
            back: &mut BTreeSet<(NodeId, NodeId)>,
        ) {
            colour.insert(id, 1);
            for s in g.successors(id) {
                match colour.get(s.as_str()).copied().unwrap_or(0) {
                    1 => {
                        back.insert((id.to_string(), s.clone()));
                    }
                    0 => {
                        let key = g.nodes.iter().find(|n| &n.id == s).map(|n| n.id.as_str()).unwrap_or(s.as_str());
                        visit(g, key, colour, back);
                    }
                    _ => {}
                }
            }
            colour.insert(id, 2);
        }

        for n in &self.nodes {
            if colour.get(n.id.as_str()).copied().unwrap_or(0) == 0 {
                visit(self, &n.id, &mut colour, &mut back);
            }
        }
        back
    }

    fn reachable_from(&self, start: &str, forward: bool) -> BTreeSet<NodeId> {
        let mut seen: BTreeSet<NodeId> = BTreeSet::new();
        let mut q = VecDeque::new();
        q.push_back(start.to_string());
        seen.insert(start.to_string());
        while let Some(cur) = q.pop_front() {
            let next: Vec<NodeId> = if forward {
                self.successors(&cur).into_iter().cloned().collect()
            } else {
                self.predecessors(&cur).into_iter().cloned().collect()
            };
            for n in next {
                if seen.insert(n.clone()) {
                    q.push_back(n);
                }
            }
        }
        seen
    }

    /// Every failure, not just the first, so the builder can mark them all.
    pub fn validate(&self) -> Vec<GraphError> {
        let mut errors = Vec::new();

        let mut ids = BTreeSet::new();
        for n in &self.nodes {
            if !ids.insert(n.id.clone()) {
                errors.push(GraphError::DuplicateId(n.id.clone()));
            }
        }
        for e in &self.edges {
            for end in [&e.from, &e.to] {
                if !ids.contains(end) {
                    errors.push(GraphError::UnknownEndpoint(end.clone()));
                }
            }
        }
        if !errors.is_empty() {
            return errors;
        }

        let inputs = self.of_kind(NodeKind::Input);
        let outputs = self.of_kind(NodeKind::Output);
        if inputs.is_empty() {
            errors.push(GraphError::MissingInput);
        }
        if outputs.is_empty() {
            errors.push(GraphError::MissingOutput);
        }
        if !errors.is_empty() {
            return errors;
        }

        // 5. node configuration
        for n in &self.nodes {
            if n.kind.needs_model() {
                if n.slot.as_deref().unwrap_or("").is_empty() {
                    errors.push(GraphError::InvalidNode { id: n.id.clone(), why: "no model slot is bound" });
                }
                if n.output.as_deref().unwrap_or("").is_empty() {
                    errors.push(GraphError::InvalidNode { id: n.id.clone(), why: "no output document is declared" });
                }
            }
            if n.kind == NodeKind::Tool {
                if n.output.as_deref().unwrap_or("").is_empty() {
                    errors.push(GraphError::InvalidNode { id: n.id.clone(), why: "no output document is declared" });
                }
                if n.command.as_ref().map(|c| c.is_empty()).unwrap_or(true) {
                    errors.push(GraphError::InvalidNode { id: n.id.clone(), why: "no command is set" });
                }
            }
        }

        // 1. dangling input
        for n in &self.nodes {
            if n.kind != NodeKind::Input && self.predecessors(&n.id).is_empty() {
                errors.push(GraphError::DanglingInput(n.id.clone()));
            }
        }

        // 2. reachability from any Input
        let mut forward: BTreeSet<NodeId> = BTreeSet::new();
        for i in &inputs {
            forward.extend(self.reachable_from(&i.id, true));
        }
        for n in &self.nodes {
            if !forward.contains(&n.id) {
                errors.push(GraphError::Unreachable(n.id.clone()));
            }
        }

        // 3. a path onward to some Output
        let mut backward: BTreeSet<NodeId> = BTreeSet::new();
        for o in &outputs {
            backward.extend(self.reachable_from(&o.id, false));
        }
        for n in &self.nodes {
            if !backward.contains(&n.id) {
                errors.push(GraphError::NoPathToOutput(n.id.clone()));
            }
        }

        // 4. every cycle bounded by a Loop that declares a limit
        for (from, to) in self.back_edges() {
            // The cycle is what a walk forward from `to` and a walk backward
            // from `from` have in common. Both walks include their own start,
            // so `from` and `to` are already members — naming them again would
            // be dead code.
            let onward = self.reachable_from(&to, true);
            let ancestors = self.reachable_from(&from, false);
            let bounded = onward
                .intersection(&ancestors)
                .filter_map(|id| self.node(id))
                .any(|n| n.kind == NodeKind::Loop && n.loop_limit.is_some_and(|l| l > 0));
            if !bounded {
                errors.push(GraphError::UnboundedLoop(to.clone()));
            }
        }

        errors
    }

    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}
