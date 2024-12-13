pub mod kd_tree;

#[cfg(test)]
mod test {
    /// Tests for the following operations:
    ///
    /// insert(), contains(), delete(), n_nearest_neighbors(), len(), is_empty()
    /// This test uses a 2-Dimensional point
    ///
    /// TODO: Create a global constant(K for example) to hold the dimension to be tested and adjust each test case to make use of K for points allocation.
    use super::kd_tree::KDTree;

    #[test]
    fn insert() {
        let points: Vec<[f64; 2]> = (0..100)
            .map(|_| {
                [
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                ]
            })
            .collect::<Vec<[f64; 2]>>();
        let mut kd_tree = KDTree::build(points);
        let point = [
            (rand::random::<f64>() * 1000.0).round() / 10.0,
            (rand::random::<f64>() * 1000.0).round() / 10.0,
        ];

        assert!(kd_tree.insert(point));
        // Cannot insert twice
        assert!(!kd_tree.insert(point));
    }

    #[test]
    fn contains() {
        let points: Vec<[f64; 2]> = (0..100)
            .map(|_| {
                [
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                ]
            })
            .collect::<Vec<[f64; 2]>>();
        let mut kd_tree = KDTree::build(points);
        let point = [
            (rand::random::<f64>() * 1000.0).round() / 10.0,
            (rand::random::<f64>() * 1000.0).round() / 10.0,
        ];
        kd_tree.insert(point);

        assert!(kd_tree.contains(&point));
    }

    #[test]
    fn delete() {
        let points: Vec<[f64; 2]> = (0..100)
            .map(|_| {
                [
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                ]
            })
            .collect::<Vec<[f64; 2]>>();
        let point = points[(rand::random::<f64>() * 100.0).round() as usize];
        let mut kd_tree = KDTree::build(points);

        assert!(kd_tree.delete(&point));
        // Cannot delete twice
        assert!(!kd_tree.delete(&point));
        // Ensure point is no longer present in k-d tree
        assert!(!kd_tree.contains(&point));
    }

    #[test]
    fn nearest_neighbors() {
        // Test with large data set
        let points_1: Vec<[f64; 2]> = (0..1000)
            .map(|_| {
                [
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                ]
            })
            .collect::<Vec<[f64; 2]>>();
        let kd_tree_1 = KDTree::build(points_1);
        let target = [50.0, 50.0];
        let neighbors_1 = kd_tree_1.nearest_neighbors(&target, 10);

        // Confirm we have exactly 10 nearest neighbors
        assert_eq!(neighbors_1.len(), 10);

        // `14.14` is the approximate distance between [40.0, 40.0] and [50.0, 50.0] &
        // [50.0, 50.0] and [60.0, 60.0]
        // so our closest neighbors are expected to be found between the bounding box [40.0, 40.0] - [60.0, 60.0]
        // with a distance from [50.0, 50.0] less than or equal 14.14
        for neighbor in neighbors_1 {
            assert!(neighbor.0 <= 14.14);
        }

        // Test with small data set
        let points_2: Vec<[f64; 2]> = vec![
            [2.0, 3.0],
            [5.0, 4.0],
            [9.0, 6.0],
            [4.0, 7.0],
            [8.0, 1.0],
            [7.0, 2.0],
        ];
        let kd_tree_2 = KDTree::build(points_2);
        let neighbors_2 = kd_tree_2.nearest_neighbors(&[6.0, 3.0], 3);
        let expected_neighbors = vec![[7.0, 2.0], [5.0, 4.0], [8.0, 1.0]];
        let neighbors = neighbors_2.iter().map(|a| a.1).collect::<Vec<[f64; 2]>>();

        // Confirm we have exactly 10 nearest neighbors
        assert_eq!(neighbors.len(), 3);

        // With a small set of data, we can manually calculate our 3 nearest neighbors
        // and compare with those obtained from our method
        assert_eq!(neighbors, expected_neighbors);
    }

    #[test]
    fn is_empty() {
        let mut kd_tree: KDTree<f64, 2> = KDTree::new();

        assert!(kd_tree.is_empty());

        kd_tree.insert([1.5, 3.0]);

        assert!(!kd_tree.is_empty());
    }

    #[test]
    fn len() {
        let points: Vec<[f64; 2]> = (0..1000)
            .map(|_| {
                [
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                    (rand::random::<f64>() * 1000.0).round() / 10.0,
                ]
            })
            .collect::<Vec<[f64; 2]>>();
        let kd_tree = KDTree::build(points);

        assert_eq!(kd_tree.len(), 1000);
    }
}
