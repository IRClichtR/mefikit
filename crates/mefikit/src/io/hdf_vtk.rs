use hdf5_metno::{File, types::FixedAscii};
use ndarray::{Array2, Array1};
use std::path::Path;

use crate::{io::to_element_type, mesh::UMesh};

// Unstructured grid

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let vtkhdf_group = file
        .group("VTKHDF").map_err(|_| "Not a VTKHDF file")?;
    let groups = vtkhdf_group.member_names()?;

    for group in groups {
        let block = vtkhdf_group.group(group.as_str())?;
        let kind: FixedAscii<64> = block.attr("Type")?.read_scalar()?;
        // Match over block type
        match kind.as_str().trim_end_matches('\0') {
            "UnstructuredGrid" => {
                // read data from file
                let points: Array2<f64> = block.dataset("Points")?.read()?;
                let offsets: Array1<usize> = block.dataset("Offsets")?.read()?;
                let conn: Array1<i64> = block.dataset("Connectivity")?.read()?;
                let types: Array1<usize> = block.dataset("Types")?.read()?;

                // transform data into mesh
                let mut mesh = UMesh::new(Array2::from_shape_vec((points.len() / 3, 3), points)?.into());
                // mesh creation ?
            },
            _ => {
                eprintln!("Unsupported kind: {kind}");
            },
        }
        
    }
    Ok(mesh)
}

pub fn write() {}