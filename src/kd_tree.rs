/// A k-d tree implementation supporting the following operations:
///
/// Main functions:
///
/// new() -> Create an empty k-d tree
/// build() -> Generate a balance k-d tree from a vector of points
/// insert() -> Add a point to a k-d tree
/// delete() -> Remove a point from a k-d tree
/// contains() -> Search for a point in a k-d tree
/// n_nearest_neighbors -> Search the nearest neighbors of a given point from a k-d tree with their respective distances
/// len() -> Determine the number of points stored in a kd-tree
/// is_empty() -> Determine whether or not there are points in a k-d tree
///
/// Helper functions:
///
/// distance() -> Calculate the Euclidean distance between two points
/// min_node() -> Determine the minimum node from a given k-d tree with respect to a given axis
/// min_node_on_axis() -> Determine the minimum node among three nodes on a given axis
///
/// Check each function's definition for more details
///
/// TODO: Implement a `range_search` function to return a set of points found within a given boundary
use num_traits::{abs, real::Real, Signed};
use std::iter::Sum;

/// Internal node of a `KDTree`
///
/// `point` contains the data of the node
/// `left` and `right` for branching
struct KDNode<T: PartialOrd + Copy, const K: usize> {
    point: [T; K],
    left: Option<Box<KDNode<T, K>>>,
    right: Option<Box<KDNode<T, K>>>,
}

impl<T: PartialOrd + Copy, const K: usize> KDNode<T, K> {
    // Create a new node `KDNode` from a given point
    fn new(point: [T; K]) -> Self {
        KDNode {
            point,
            left: None,
            right: None,
        }
    }
}

// A `KDTree` with a `root` node and a `size` to represent the number of points in the kd-tree
pub struct KDTree<T: PartialOrd + Copy, const K: usize> {
    root: Option<Box<KDNode<T, K>>>,
    size: usize,
}

impl<T: PartialOrd + Copy, const K: usize> Default for KDTree<T, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PartialOrd + Copy, const K: usize> KDTree<T, K> {
    // Create and empty kd-tree
    pub fn new() -> Self {
        KDTree {
            root: None,
            size: 0,
        }
    }

    // Returns true if point found, false otherwise
    pub fn contains(&self, point: &[T; K]) -> bool {
        search_rec(&self.root, point, 0)
    }

    // Returns true if successfully insert a point, false otherwise
    pub fn insert(&mut self, point: [T; K]) -> bool {
        let inserted: bool = insert_rec(&mut self.root, point, 0);
        if inserted {
            self.size += 1;
        }
        inserted
    }

    // Returns true if successfully delete a point, false otherwise
    pub fn delete(&mut self, point: &[T; K]) -> bool {
        let deleted = delete_rec(&mut self.root, point, 0);
        if deleted {
            self.size -= 1;
        }
        deleted
    }

    // Returns the nearest neighbors of a given point with their respective distances
    pub fn nearest_neighbors(&self, point: &[T; K], n: usize) -> Vec<(T, [T; K])>
    where
        T: Sum + Signed + Real,
    {
        let mut neighbors: Vec<(T, [T; K])> = Vec::new();
        n_nearest_neighbors(&self.root, point, n, 0, &mut neighbors);
        neighbors
    }

    // Returns the number of points in a kd-tree
    pub fn len(&self) -> usize {
        self.size
    }

    // Determine whether there exist points in a kd-tree or not
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    // Returns a kd-tree built from a vector points
    pub fn build(points: Vec<[T; K]>) -> KDTree<T, K> {
        let mut tree: KDTree<T, K> = KDTree::new();
        if points.is_empty() {
            tree
        } else {
            tree.size = points.len();
            tree.root = build_rec(points, 0);

            tree
        }
    }
}

// Helper functions ............................................................................

// Recursively insert a point in a kd-tree
fn insert_rec<T: PartialOrd + Copy, const K: usize>(
    kd_tree: &mut Option<Box<KDNode<T, K>>>,
    point: [T; K],
    depth: usize,
) -> bool {
    if let Some(ref mut kd_node) = kd_tree {
        let axis: usize = depth % K;
        if point[axis] < kd_node.point[axis] {
            insert_rec(&mut kd_node.left, point, depth + 1)
        } else if point == kd_node.point {
            false
        } else {
            insert_rec(&mut kd_node.right, point, depth + 1)
        }
    } else {
        let nde: KDNode<T, K> = KDNode::new(point);
        *kd_tree = Some(Box::new(nde));
        true
    }
}

// Recursively search for a given point in a kd-tree
fn search_rec<T: PartialOrd + Copy, const K: usize>(
    kd_tree: &Option<Box<KDNode<T, K>>>,
    point: &[T; K],
    depth: usize,
) -> bool {
    if let Some(kd_node) = kd_tree {
        if point == &kd_node.point {
            true
        } else {
            let axis: usize = depth % K;
            if point[axis] < kd_node.point[axis] {
                search_rec(&kd_node.left, point, depth + 1)
            } else {
                search_rec(&kd_node.right, point, depth + 1)
            }
        }
    } else {
        false
    }
}

// Recursively delete a point from a kd-tree
fn delete_rec<T: PartialOrd + Copy, const K: usize>(
    kd_node: &mut Option<Box<KDNode<T, K>>>,
    point: &[T; K],
    depth: usize,
) -> bool {
    if let Some(current_node) = kd_node {
        let axis: usize = depth % K;
        if current_node.point == *point {
            if current_node.right.is_some() {
                // safe to use `unwrap()` since we know for sure there exist a node
                let min_point = min_node(current_node.right.as_deref(), axis, 0)
                    .unwrap()
                    .point;

                current_node.point = min_point;
                delete_rec(&mut current_node.right, &current_node.point, depth + 1)
            } else if current_node.left.is_some() {
                let min_point: [T; K] = min_node(current_node.left.as_deref(), axis, 0)
                    .unwrap()
                    .point;

                current_node.point = min_point;
                current_node.right = current_node.left.take();
                delete_rec(&mut current_node.right, &current_node.point, depth + 1)
            } else {
                *kd_node = None;
                true
            }
        } else if point[axis].lt(&current_node.point[axis]) {
            delete_rec(&mut current_node.left, point, depth + 1)
        } else {
            delete_rec(&mut current_node.right, point, depth + 1)
        }
    } else {
        false
    }
}

/// Recursively build a kd-tree from a vector of points by taking the median of a sorted
/// vector of points as the root to maintain a balance kd-tree
fn build_rec<T: PartialOrd + Copy, const K: usize>(
    mut points: Vec<[T; K]>,
    depth: usize,
) -> Option<Box<KDNode<T, K>>> {
    if points.is_empty() {
        None
    } else {
        let axis = depth % K;
        points.sort_by(|a, b| {
            a[axis]
                .partial_cmp(&b[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let median: usize = points.len() / 2;
        let mut node: KDNode<T, K> = KDNode::new(points[median]);

        node.left = build_rec(points[..median].to_vec(), depth + 1);
        node.right = build_rec(points[median + 1..].to_vec(), depth + 1);

        Some(Box::new(node))
    }
}

// Calculate the distance between two points
fn distance<T, const K: usize>(point_1: &[T; K], point_2: &[T; K]) -> T
where
    T: PartialOrd + Copy + Sum + Real,
{
    point_1
        .iter()
        .zip(point_2.iter())
        .map(|(&a, &b)| {
            let diff: T = a - b;
            diff * diff
        })
        .sum::<T>()
        .sqrt()
}

// Returns the minimum nodes among three kd-nodes on a given axis
fn min_node_on_axis<'a, T: PartialOrd + Copy, const K: usize>(
    kd_node: &'a KDNode<T, K>,
    left: Option<&'a KDNode<T, K>>,
    right: Option<&'a KDNode<T, K>>,
    axis: usize,
) -> &'a KDNode<T, K> {
    let mut current_min_node = kd_node;
    if let Some(left_node) = left {
        if left_node.point[axis].lt(&current_min_node.point[axis]) {
            current_min_node = left_node;
        }
    }
    if let Some(right_node) = right {
        if right_node.point[axis].lt(&current_min_node.point[axis]) {
            current_min_node = right_node;
        }
    }
    current_min_node
}

// Returns the minimum node of a kd-tree with respect to an axis
fn min_node<T: PartialOrd + Copy, const K: usize>(
    kd_node: Option<&KDNode<T, K>>,
    axis: usize,
    depth: usize,
) -> Option<&KDNode<T, K>> {
    if let Some(current_node) = kd_node {
        let current_axis = depth % K;
        if current_axis == axis {
            if current_node.left.is_some() {
                min_node(current_node.left.as_deref(), axis, depth + 1)
            } else {
                Some(current_node)
            }
        } else {
            let (left_min, right_min): (Option<&KDNode<T, K>>, Option<&KDNode<T, K>>) = (
                min_node(current_node.left.as_deref(), axis, depth + 1),
                min_node(current_node.right.as_deref(), axis, depth + 1),
            );
            Some(min_node_on_axis(current_node, left_min, right_min, axis))
        }
    } else {
        None
    }
}

// Find the nearest neighbors of a given point. The number neighbors is determine by the variable `n`.
fn n_nearest_neighbors<T, const K: usize>(
    kd_tree: &Option<Box<KDNode<T, K>>>,
    point: &[T; K],
    n: usize,
    depth: usize,
    neighbors: &mut Vec<(T, [T; K])>,
) where
    T: PartialOrd + Copy + Sum + Real + Signed,
{
    if let Some(kd_node) = kd_tree {
        let distance: T = distance(&kd_node.point, point);
        if neighbors.len() < n {
            neighbors.push((distance, kd_node.point));
        } else {
            // safe to call unwrap() since we know our neighbors is ont empty in this scope
            let max_distance = neighbors
                .iter()
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap()
                .0;
            if distance < max_distance {
                if let Some(pos) = neighbors.iter().position(|x| x.0 == max_distance) {
                    neighbors[pos] = (distance, kd_node.point);
                }
            }
        }

        let axis: usize = depth % K;
        let target_axis: T = point[axis];
        let split_axis: T = kd_node.point[axis];

        let (look_first, look_second) = if target_axis < split_axis {
            (&kd_node.left, &kd_node.right)
        } else {
            (&kd_node.right, &kd_node.left)
        };

        if look_first.is_some() {
            n_nearest_neighbors(look_first, point, n, depth + 1, neighbors);
        }

        // Check if it's necessary to look on the other branch by computing the distance between our current point with the nearest point on the other branch
        if look_second.is_some() {
            let max_distance = neighbors
                .iter()
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap()
                .0;
            if neighbors.len() < n || abs(target_axis - split_axis) < max_distance {
                n_nearest_neighbors(look_second, point, n, depth + 1, neighbors);
            }
        }
    }
}
