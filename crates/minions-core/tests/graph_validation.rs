//! The validator makes a promise the engine depends on, so acceptance is
//! re-checked by independent graph walks rather than by the same rule.

use minions_core::graph::*;
use proptest::prelude::*;
use std::collections::{BTreeSet, VecDeque};

fn n(id: &str, kind: NodeKind) -> Node {
    Node { id: id.into(), kind, role: None, slot: None, output: None, loop_limit: None, command: None, gate: false }
}
fn agent(id: &str) -> Node {
    Node {
        id: id.into(),
        kind: NodeKind::Agent,
        role: Some("analyst".into()),
        slot: Some("reasoning".into()),
        output: Some(format!("{id}.md")),
        loop_limit: None,
        command: None,
        gate: false,
    }
}
fn e(a: &str, b: &str) -> Edge {
    Edge { from: a.into(), to: b.into(), when: None }
}

fn linear() -> Graph {
    Graph {
        nodes: vec![n("in", NodeKind::Input), agent("a"), agent("b"), n("out", NodeKind::Output)],
        edges: vec![e("in", "a"), e("a", "b"), e("b", "out")],
    }
}

// --- independent re-checks, deliberately not sharing code with the validator ---

fn walk(g: &Graph, start: &str, forward: bool) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut q = VecDeque::from(vec![start.to_string()]);
    seen.insert(start.to_string());
    while let Some(c) = q.pop_front() {
        for edge in &g.edges {
            let next = if forward {
                if edge.from == c { Some(&edge.to) } else { None }
            } else if edge.to == c {
                Some(&edge.from)
            } else {
                None
            };
            if let Some(nx) = next {
                if seen.insert(nx.clone()) {
                    q.push_back(nx.clone());
                }
            }
        }
    }
    seen
}

fn every_non_input_has_a_predecessor(g: &Graph) -> bool {
    g.nodes
        .iter()
        .filter(|x| x.kind != NodeKind::Input)
        .all(|x| g.edges.iter().any(|edge| edge.to == x.id))
}

fn all_reachable_from_input(g: &Graph) -> bool {
    let Some(input) = g.nodes.iter().find(|x| x.kind == NodeKind::Input) else { return false };
    let seen = walk(g, &input.id, true);
    g.nodes.iter().all(|x| seen.contains(&x.id))
}

fn all_reach_output(g: &Graph) -> bool {
    let Some(out) = g.nodes.iter().find(|x| x.kind == NodeKind::Output) else { return false };
    let seen = walk(g, &out.id, false);
    g.nodes.iter().all(|x| seen.contains(&x.id))
}

#[test]
fn a_linear_graph_is_accepted() {
    assert_eq!(linear().validate(), vec![], "a plain chain must validate");
}

#[test]
fn a_node_with_no_incoming_connection_is_named() {
    let mut g = linear();
    g.nodes.push(agent("orphan"));
    g.edges.push(e("orphan", "out"));
    assert!(g.validate().contains(&GraphError::DanglingInput("orphan".into())));
}

#[test]
fn a_node_that_cannot_be_reached_is_named() {
    let mut g = linear();
    g.nodes.push(agent("island"));
    g.edges.push(e("b", "island"));
    g.edges.retain(|x| !(x.from == "b" && x.to == "out"));
    g.edges.push(e("island", "out"));
    // now make it genuinely unreachable by cutting the only way in
    g.edges.retain(|x| !(x.from == "b" && x.to == "island"));
    g.edges.push(e("island", "island"));
    let errs = g.validate();
    assert!(errs.iter().any(|x| matches!(x, GraphError::Unreachable(id) if id == "island")), "{errs:?}");
}

#[test]
fn work_that_cannot_reach_the_output_is_named() {
    let mut g = linear();
    g.nodes.push(agent("deadend"));
    g.edges.push(e("a", "deadend"));
    assert!(g.validate().contains(&GraphError::NoPathToOutput("deadend".into())));
}

#[test]
fn a_cycle_without_a_loop_node_is_refused_and_with_one_is_accepted() {
    let mut g = linear();
    g.edges.push(e("b", "a")); // review loop, unbounded
    let errs = g.validate();
    assert!(errs.iter().any(|x| matches!(x, GraphError::UnboundedLoop(_))), "{errs:?}");

    let mut bounded = linear();
    let mut lp = n("loop", NodeKind::Loop);
    lp.loop_limit = Some(3);
    bounded.nodes.push(lp);
    bounded.edges.push(e("b", "loop"));
    bounded.edges.push(e("loop", "a"));
    assert_eq!(bounded.validate(), vec![], "a loop with a limit must be accepted");
}

#[test]
fn an_agent_without_a_slot_or_an_output_is_named() {
    let mut g = linear();
    g.nodes.push(Node { id: "bare".into(), kind: NodeKind::Agent, role: None, slot: None, output: None, loop_limit: None, command: None, gate: false });
    g.edges.push(e("a", "bare"));
    g.edges.push(e("bare", "out"));
    let errs = g.validate();
    let whys: Vec<&str> = errs
        .iter()
        .filter_map(|x| match x {
            GraphError::InvalidNode { id, why } if id == "bare" => Some(*why),
            _ => None,
        })
        .collect();
    assert!(whys.contains(&"no model slot is bound"), "{errs:?}");
    assert!(whys.contains(&"no output document is declared"), "{errs:?}");
}

#[test]
fn back_edges_are_found_so_the_engine_can_ignore_them() {
    let mut g = linear();
    g.edges.push(e("b", "a"));
    let back = g.back_edges();
    assert_eq!(back.len(), 1);
    assert!(back.contains(&("b".to_string(), "a".to_string())));
}

#[test]
fn every_error_names_a_node_a_human_can_find() {
    let mut g = linear();
    g.edges.push(e("b", "a"));
    g.nodes.push(agent("orphan"));
    for err in g.validate() {
        let text = err.to_string();
        assert!(text.contains('`'), "error does not quote a node: {text}");
        assert!(text.len() > 20, "error is too thin to act on: {text}");
    }
}

// ---- gaps found by mutation testing, 2026-08-16 ----

fn bounded_loop(id: &str, limit: Option<u32>) -> Node {
    let mut l = n(id, NodeKind::Loop);
    l.loop_limit = limit;
    l
}

#[test]
fn a_loop_node_off_the_cycle_does_not_bound_it() {
    // The membership test must be the intersection of the two walks. A Loop
    // sitting elsewhere in the graph must not license an unbounded cycle.
    let mut g = linear();
    g.edges.push(e("b", "a")); // unbounded cycle a -> b -> a
    g.nodes.push(bounded_loop("far", Some(5)));
    g.edges.push(e("b", "far"));
    g.edges.push(e("far", "out"));

    let errs = g.validate();
    assert!(
        errs.iter().any(|x| matches!(x, GraphError::UnboundedLoop(_))),
        "a loop outside the cycle wrongly bounded it: {errs:?}"
    );
}

#[test]
fn a_loop_without_a_limit_does_not_bound_a_cycle() {
    let mut g = linear();
    g.nodes.push(bounded_loop("lp", None));
    g.edges.push(e("b", "lp"));
    g.edges.push(e("lp", "a"));
    let errs = g.validate();
    assert!(errs.iter().any(|x| matches!(x, GraphError::UnboundedLoop(_))), "{errs:?}");
}

#[test]
fn a_loop_limit_of_zero_does_not_bound_a_cycle() {
    // Zero iterations is not a bound, it is a graph that cannot make progress.
    let mut g = linear();
    g.nodes.push(bounded_loop("lp", Some(0)));
    g.edges.push(e("b", "lp"));
    g.edges.push(e("lp", "a"));
    let errs = g.validate();
    assert!(errs.iter().any(|x| matches!(x, GraphError::UnboundedLoop(_))), "{errs:?}");
}

#[test]
fn is_valid_agrees_with_validate() {
    assert!(linear().is_valid(), "a valid graph must report valid");
    let mut broken = linear();
    broken.nodes.push(agent("orphan"));
    assert!(!broken.is_valid(), "a broken graph must not report valid");
}

proptest! {
    /// Anything the validator accepts satisfies the five claims, checked by
    /// independent walks. Sharing the validator's own code here would prove
    /// nothing.
    #[test]
    fn acceptance_implies_the_promises(
        extra_nodes in 0usize..5,
        extra_edges in prop::collection::vec((0usize..7, 0usize..7), 0..10),
    ) {
        let mut g = linear();
        for i in 0..extra_nodes {
            g.nodes.push(agent(&format!("x{i}")));
        }
        let ids: Vec<String> = g.nodes.iter().map(|x| x.id.clone()).collect();
        for (a, b) in extra_edges {
            if a < ids.len() && b < ids.len() && a != b {
                g.edges.push(Edge { from: ids[a].clone(), to: ids[b].clone(), when: None });
            }
        }

        if g.is_valid() {
            prop_assert!(every_non_input_has_a_predecessor(&g), "accepted a dangling input");
            prop_assert!(all_reachable_from_input(&g), "accepted an unreachable node");
            prop_assert!(all_reach_output(&g), "accepted work that cannot reach the output");
            prop_assert!(g.back_edges().is_empty() || g.nodes.iter().any(|x| x.kind == NodeKind::Loop),
                "accepted a cycle with no loop node");
            for node in &g.nodes {
                if node.kind.needs_model() {
                    prop_assert!(node.slot.is_some(), "accepted an agent with no slot");
                    prop_assert!(node.output.is_some(), "accepted an agent with no output");
                }
            }
        }
    }

    /// Validation is total: no graph, however malformed, makes it panic.
    #[test]
    fn validation_never_panics(
        kinds in prop::collection::vec(0usize..9, 0..8),
        edges in prop::collection::vec((0usize..8, 0usize..8), 0..14),
    ) {
        let all = [NodeKind::Input, NodeKind::Output, NodeKind::Agent, NodeKind::Foreman,
                   NodeKind::Tool, NodeKind::Gate, NodeKind::Branch, NodeKind::Loop, NodeKind::Merge];
        let nodes: Vec<Node> = kinds.iter().enumerate().map(|(i, k)| n(&format!("n{i}"), all[*k])).collect();
        let ids: Vec<String> = nodes.iter().map(|x| x.id.clone()).collect();
        let edges: Vec<Edge> = edges
            .into_iter()
            .filter(|(a, b)| *a < ids.len() && *b < ids.len())
            .map(|(a, b)| Edge { from: ids[a].clone(), to: ids[b].clone(), when: None })
            .collect();
        let g = Graph { nodes, edges };
        let _ = g.validate();
        let _ = g.back_edges();
    }
}
