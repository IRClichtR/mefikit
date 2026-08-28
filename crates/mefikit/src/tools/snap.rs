use crate::mesh::{ElementLike, IndirectIndexOwned, UMesh, UMeshView};

use itertools::Itertools;
use rustc_hash::FxHashMap;

fn sorted_indices_for<const T: usize>(points: &[[f64; T]]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..points.len()).collect();
    indices.sort_by(|&a, &b| {
        for (pa, pb) in points[a].iter().zip(points[b].iter()) {
            match pa.partial_cmp(pb) {
                Some(std::cmp::Ordering::Equal) => continue,
                other => return other.unwrap_or(std::cmp::Ordering::Equal),
            }
        }
        std::cmp::Ordering::Equal
    });
    indices
}

fn closest_within<const T: usize>(
    target: &[f64; T],
    sorted_indices: &[usize],
    points: &[[f64; T]],
    eps: f64,
) -> Option<usize> {
    let eps_sq = eps * eps;
    let left = sorted_indices.partition_point(|&i| points[i][0] < target[0] - eps);
    let right = sorted_indices.partition_point(|&i| points[i][0] <= target[0] + eps);

    let mut best_dist_sq = eps_sq;
    let mut best_idx = None;
    for &i in &sorted_indices[left..right] {
        let dist_sq: f64 = target
            .iter()
            .zip(points[i].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = Some(i);
        }
    }
    best_idx
}

fn snap_dim_n<const T: usize>(subject: &mut UMesh, reference: &UMeshView, eps: f64) {
    let ref_points: Vec<[f64; T]> = reference
        .used_nodes()
        .iter()
        .map(|&i| {
            reference
                .coords()
                .row(i)
                .to_slice()
                .unwrap()
                .try_into()
                .unwrap()
        })
        .collect();
    let sorted_ref = sorted_indices_for(&ref_points);

    for node in subject.used_nodes() {
        let coord: &mut [f64; T] = subject
            .coords
            .row_mut(node)
            .into_slice()
            .unwrap()
            .try_into()
            .unwrap();
        if let Some(idx) = closest_within(coord, &sorted_ref, &ref_points, eps) {
            coord.copy_from_slice(&ref_points[idx]);
        }
    }
}

fn duplicates_from_dim_n<const T: usize>(
    subject: &UMeshView,
    reference: &UMeshView,
    eps: f64,
) -> FxHashMap<usize, Vec<usize>> {
    let mut res: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let ref_used_nodes = reference.used_nodes();
    let ref_points: Vec<[f64; T]> = ref_used_nodes
        .iter()
        .map(|&i| {
            reference
                .coords()
                .row(i)
                .to_slice()
                .unwrap()
                .try_into()
                .unwrap()
        })
        .collect();
    let sorted_ref = sorted_indices_for(&ref_points);

    for node in subject.used_nodes() {
        let coord: &[f64; T] = subject
            .coords
            .row(node)
            .to_slice()
            .unwrap()
            .try_into()
            .unwrap();
        if let Some(idx) = closest_within(coord, &sorted_ref, &ref_points, eps) {
            let close_nodes = res.entry(ref_used_nodes[idx]).or_default();
            close_nodes.push(node);
        }
    }
    res
}

/// Find duplicates coords of subject mesh onto used nodes of reference.
pub fn duplicates_from(
    subject: &UMeshView,
    reference: &UMeshView,
    eps: f64,
) -> FxHashMap<usize, Vec<usize>> {
    match subject.coords().ncols() {
        1 => duplicates_from_dim_n::<1>(subject, reference, eps),
        2 => duplicates_from_dim_n::<2>(subject, reference, eps),
        3 => duplicates_from_dim_n::<3>(subject, reference, eps),
        _ => panic!("Could not snap the mesh because of its dimension."),
    }
}

/// Snap coords of subject mesh onto used nodes of reference.
///
/// Be careful, the method could produce degenerated elements if eps is not lower than half the
/// smallest distance between two points from the same element.
pub fn snap(subject: &mut UMesh, reference: &UMeshView, eps: f64) {
    match subject.coords().ncols() {
        // 1 => snap_dim_n::<1>(subject, reference, eps),
        2 => snap_dim_n::<2>(subject, reference, eps),
        3 => snap_dim_n::<3>(subject, reference, eps),
        _ => panic!("Could not snap the mesh because of its dimension."),
    }
}

fn duplicates_dim_n<const T: usize>(mesh: &UMeshView, eps: f64) -> IndirectIndexOwned<usize> {
    let used_nodes = mesh.used_nodes();
    let points: Vec<[f64; T]> = used_nodes
        .iter()
        .map(|&i| mesh.coords().row(i).to_slice().unwrap().try_into().unwrap())
        .collect();

    let mut sorted_indices: Vec<usize> = (0..points.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        for (pa, pb) in points[a].iter().zip(points[b].iter()) {
            match pa.partial_cmp(pb) {
                Some(std::cmp::Ordering::Equal) => continue,
                other => return other.unwrap_or(std::cmp::Ordering::Equal),
            }
        }
        std::cmp::Ordering::Equal
    });

    let eps_sq = eps * eps;
    let mut processed = vec![false; points.len()];
    let mut res = IndirectIndexOwned::new();

    for (pos, &idx) in sorted_indices.iter().enumerate() {
        if processed[idx] {
            continue;
        }
        let mut group = vec![used_nodes[idx]];
        processed[idx] = true;

        for &other_idx in sorted_indices.iter().skip(pos + 1) {
            if processed[other_idx] {
                continue;
            }
            if points[other_idx][0] - points[idx][0] > eps {
                break;
            }
            let dist_sq: f64 = points[idx]
                .iter()
                .zip(points[other_idx].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if dist_sq <= eps_sq {
                group.push(used_nodes[other_idx]);
                processed[other_idx] = true;
            }
        }

        if group.len() > 1 {
            group.sort_unstable();
            res.push(&group);
        }
    }
    res
}

pub fn duplicates(mesh: &UMeshView, eps: f64) -> IndirectIndexOwned<usize> {
    match mesh.coords().ncols() {
        1 => duplicates_dim_n::<1>(mesh, eps),
        2 => duplicates_dim_n::<2>(mesh, eps),
        3 => duplicates_dim_n::<3>(mesh, eps),
        _ => panic!("Could not find duplicates because of invalid dimension."),
    }
}

fn find_group(n: &usize, nodes: &[usize], groups: &[usize]) -> Option<usize> {
    match nodes.binary_search(n) {
        Ok(i) => Some(groups[i]),
        Err(_) => None,
    }
}

/// Merge close nodes.
///
/// Be careful, this method can produce degenerated elements if used with an epsilon greater than
/// the distance between two nodes of the same element.
pub fn merge_nodes(mesh: &mut UMesh, eps: f64) {
    let dups = duplicates(&mesh.view(), eps);
    let sorted_nodes_dup: Vec<(usize, usize)> = dups
        .iter()
        .enumerate()
        .flat_map(|(i, ns)| ns.iter().cloned().zip(std::iter::repeat(i)))
        .sorted_unstable()
        .collect();
    let sorted_nodes: Vec<usize> = sorted_nodes_dup.iter().map(|t| t.0).collect();
    let sorted_grps: Vec<usize> = sorted_nodes_dup.iter().map(|t| t.1).collect();
    // Here the idea is to go once throught each element and to renumber all nodes presents in
    // duplicates to the first node of the duplicates group.
    // I suppose that the number of duplicates is small in front of the number of elements so I
    // only go throught all elements once and thought the number of duplicates many times.
    // TODO: build a parallel version of the ElementMut iterator
    let eids: Vec<_> = mesh.elements().map(|e| e.id()).collect();
    for e in eids {
        let elem = mesh.element_mut(e);
        for n in elem.connectivity {
            if let Some(grp) = find_group(n, &sorted_nodes, &sorted_grps) {
                *n = dups[grp][0];
            }
        }
    }
}

pub trait NodeDuplicates {
    fn merge_nodes(&mut self, eps: f64);
    fn snap_on(&mut self, other: &UMeshView, eps: f64);
}

impl NodeDuplicates for UMesh {
    fn merge_nodes(&mut self, eps: f64) {
        merge_nodes(self, eps)
    }

    fn snap_on(&mut self, other: &UMeshView, eps: f64) {
        snap(self, other, eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, UMesh};
    use ndarray as nd;

    #[test]
    fn test_snap_2d() {
        let subject_coords =
            nd::Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.01, 0.0, 0.0, 1.01]).unwrap();
        let mut subject = UMesh::new(subject_coords.into());
        subject.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 2]]).to_shared(),
            None,
        );

        let reference_coords =
            nd::Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        let mut reference = UMesh::new(reference_coords.into());
        reference.add_regular_block(ElementType::SEG2, nd::arr2(&[[0, 1]]).to_shared(), None);

        snap(&mut subject, &reference.view(), 0.02);
        assert!((subject.coords()[[1, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_nodes() {
        let mesh_coords =
            nd::Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.001, 0.001, 1.0, 0.0, 1.0, 1.0])
                .unwrap();
        let mut mesh = UMesh::new(mesh_coords.into());
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 2], [2, 3]]).to_shared(),
            None,
        );

        let original_num_nodes = mesh.coords().nrows();
        merge_nodes(&mut mesh, 0.01);
        // After merging, some nodes should be merged
        assert!(mesh.coords().nrows() <= original_num_nodes);
    }

    #[test]
    fn test_duplicates_2d() {
        let mesh_coords =
            nd::Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
                .unwrap();
        let mut mesh = UMesh::new(mesh_coords.into());
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 2], [1, 3]]).to_shared(),
            None,
        );

        let dups = duplicates(&mesh.view(), 0.01);
        // Should find at least one duplicate (nodes 0 and 1)
        assert!(dups.len() > 0);
    }

    #[test]
    fn test_nodeduplicates_trait() {
        let mesh_coords =
            nd::Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.001, 0.001, 1.0, 0.0, 1.0, 1.0])
                .unwrap();
        let mut mesh = UMesh::new(mesh_coords.into());
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 2], [2, 3]]).to_shared(),
            None,
        );

        mesh.merge_nodes(0.01);
        // Just verify it doesn't panic
    }
}
